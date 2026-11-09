# Bharat-EarthFM — Data-Engineering DAG (Bharat-EO-Corpus build)

End-to-end pipeline that turns raw multi-sensor archives into the **analysis-ready
datacube (ARD)** and the shuffled **WebDataset shards** the trainer consumes.
Target: ~0.75 PB curated ARD over India + buffer (see blueprint Part IV).

```
                 ┌──────────────────────────────────────────────────────────┐
 [0] REGISTRIES  │ Copernicus DS · NASA Earthdata · ISRO NOEDA/MOSDAC ·       │
                 │ ECMWF CDS · India-WRIS · JAXA G-Portal                     │
                 └──────────────────────────────────────────────────────────┘
                          │ (per-sensor pull, incremental, checksummed)
 [1] INGEST  ────────────▼────────────────────────────────────────────────────
        landing/  (raw SAFE/COG/NetCDF/HDF, immutable, object store)
                          │
 [2] CALIBRATE / PREPROCESS  (per modality)
        optical: TOA→BOA, cloud/shadow mask        SAR: GRD→RTC σ0(dB), speckle,
        geostationary: rectify+calibrate           ortho; SLC→InSAR (HyP3/ISCE, side)
        GRACE/weather: regrid                       DEM: void-fill, slope/HAND
                          │
 [3] HARMONIZE  → common grid (India Albers), multi-res pyramid 10/20/30/60/250 m,
        timestamp alignment, per-sensor radiometric harmonization
                          │
 [4] QA & MASK  → per-pixel quality flags; tropical-monsoon cloud/shadow/haze
        probability; per-token CONFIDENCE WEIGHTS (feed training, Patent P10/P11)
                          │
 [5] MPATE PHASE LABELER  → monsoon phase {pre,onset,active,break,withdrawal}
        from IMD 0.25° rainfall; agro-calendar day (kharif/rabi/zaid) (Patent P9)
                          │
 [6] TEMPORAL STACK  → per-tile time series; gap policy = keep+flag (no infill)
                          │
 [7] ARD DATACUBE  → Zarr (dims time×band×y×x, per-sensor groups, chunked), with
        confidence + phase + geo metadata
                          │
 [8] LABEL JOIN  → rasterize/align FASAL crop, CGWB wells(→kriged), CWC reservoir,
        flood masks, Bhuvan LULC, WorldCover, GEDI → label cubes
                          │
 [9] TILE & SHARD  → 224×224×T×G chips, QA-filtered, shuffled WebDataset .tar
        (~1 GB each) + Parquet manifest/index
                          │
 [10] CATALOG & VERSION → STAC catalog + DVC/lakeFS snapshot + data card
```

---

## Node specifications

| # | Node | Tooling | Parallelism | Key failure modes & handling |
|---|------|---------|-------------|------------------------------|
| 1 | Ingest | `cdsapi`, `earthaccess`/ASF, `pystac-client`+`stackstac`, NOEDA/MOSDAC clients, India-WRIS API | Ray actors per sensor; rate-limit aware | partial downloads → checksum + resume; quota → backoff; ISRO **direct downlink** removes WAN for Indian sensors |
| 2 | Calibrate | Sen2Cor/Fmask, SNAP/`pyroSAR` (RTC), ISCE2+MintPy (InSAR), GDAL | Dask per scene | bad geolocation → reject; SAR thermal-noise → mask |
| 3 | Harmonize | `rioxarray`, `pyproj`, `odc-geo` | Dask grid-blocks | reprojection seams → overlap-blend; res mismatch → pyramid |
| 4 | QA/mask | s2cloudless (monsoon-tuned), custom CNN haze | Dask per tile | over-masking → confidence-weighted soft mask, not hard drop |
| 5 | MPATE label | rule+HMM on IMD grids | per agro-zone | data gaps in IMD → climatological prior |
| 6 | Temporal stack | `xarray` resample/groupby | per tile | irregular revisit → continuous-time index, gap flag |
| 7 | ARD assemble | `zarr`, `xarray` | sharded writes | chunk skew → rechunk; metadata drift → schema validation |
| 8 | Label join | `rasterio`, `geopandas`, kriging (`pykrige`) | per region | sparse wells → kriging variance kept as uncertainty band |
| 9 | Tile/shard | WebDataset writer | Ray map | class imbalance → agro-stratified sampler (P14); leakage → spatial-block tagging in manifest |
| 10 | Catalog | STAC, DVC/lakeFS | — | reproducibility → immutable snapshots + data card |

---

## ARD datacube schema (Zarr)

```
bharat_eo_corpus.zarr/
  /optical_vnir   dims (time, band[B2,B3,B4,B8], y, x)   chunk (1, 1, 256, 256)  int16
  /optical_swir   dims (time, band[B11,B12], y, x)
  /sar_c          dims (time, pol[VV,VH], y, x)          float32 (σ0 dB)
  /sar_l          dims (time, pol[HH,HV], y, x)          # NISAR
  /geostationary  dims (time, channel, y, x)             # INSAT, high cadence
  /weather        dims (time, var[precip,t2m,...], y, x) # IMD/ERA5 regridded
  /static         dims (band[DEM,slope,HAND,GRACE_tws], y, x)
  /qa             dims (time, sensor, y, x)  uint8        # quality flags
  /confidence     dims (time, sensor, y, x)  float16      # per-pixel weight
  /phase          dims (time, y, x)  uint8                # MPATE monsoon phase
  attrs: crs=India-Albers, gsd_base=10m, time=continuous(days since 2017-01-01)
```

---

## Monsoon-phase labeler (node 5) — algorithm

1. Input: IMD 0.25° daily gridded rainfall; area-average per agro-climatic zone.
2. **Onset:** local rainfall crosses zone threshold for ≥3 days after the
   climatological window (anchored to IMD's Kerala-onset date, propagated north).
3. **Active vs break:** standardized rainfall anomaly Z over a 7-day window;
   Z>+0.5 sustained → *active*; Z<−0.5 with a dry spell ≥5 days → *break*.
4. **Withdrawal:** sustained Z<−1 after Sept + falling soil moisture.
5. Output: daily per-pixel phase raster (5 classes) + agro-calendar day index
   (kharif Jun–Oct, rabi Nov–Mar, zaid Apr–May). Feeds MPATE temporal encoding.

---

## Orchestration, storage tiers, sizing

- **Orchestrator:** Prefect (or Airflow) DAG; idempotent tasks; incremental
  (only new scenes); Ray/Dask for the heavy map stages.
- **Storage tiers:** `landing/` (raw mirror, cold object ~1.5 PB) → `ard/`
  (Zarr, warm ~0.75 PB) → `shards/` (hot parallel-FS ~0.75 PB, ×3 replicated for
  the active subset) → `checkpoints/` + `embeddings/` (product output).
- **Throughput target at node 9→trainer:** ≥30–50 GB/s sustained (matches the
  pretraining data-loader requirement).
- **Governance:** all data resident in India (NGP-2022); STAC catalog public for
  open layers, restricted for licensed (Cartosat, FASAL).
```
