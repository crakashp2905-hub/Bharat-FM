"""Evidence levels, independent-reference overlap and source-provenance records.

The benchmark's *only* dense label is ESA WorldCover v200 (2021). Dynamic World, ESRI 10 m LULC
and the Dynamic World probabilities are used for per-pixel **agreement counts** (`conf`) and soft
targets, not as independent labels; all four are model products. The single genuinely independent
reference we hold is the set of 297 Google-street-view expert crop annotations (239 crop / 58
other) in data/bench_v1/t2_fieldtruth_india.csv.

    python -m src.ground_truth  [--root data/chips_dsv2_sp] [--out data/provenance]

writes
    chip_provenance.csv        one row per manifest chip: split, coordinates, per-source agreement
    label_noise_by_class.csv   per WorldCover class: agreement with DW / ESRI, confident fraction
    reference_overlap.csv      each expert point -> nearest chip, distance, inside-chip flag
    gt_target_chips.csv        the 297 expert points to export as an eval-only ground-truth subset
    PROVENANCE.json            sources, asset ids, licences, band layout, hashes, evidence levels
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_integrity as bi                                      # noqa: E402

LAB = Path(__file__).resolve().parents[1]
HALF_CHIP_KM = 224 * 10 / 2 / 1000.0                              # 1.12 km (10 m pixels)
# aux channel layout written by export_dataset_v2._bandlists (aux = label bands minus 'label')
AUX = dict(dw=0, esri=1, dwp0=2, wc_crop=11, conf=12, zone=13)

EVIDENCE_LEVELS = {
    "L0": "single-source: ESA WorldCover label only (a model product)",
    "L1": "multi-source: >=2 of {WorldCover, Dynamic World, ESRI} agree (conf>=2). Still model products.",
    "L2": "triple agreement: all three agree (conf==3) AND the class is not force-confident "
          "(mangrove/moss are set to conf=3 by construction, so they never reach L2).",
    "L3": "independent expert reference (street-view annotations / field data). Currently covers "
          "NO pixel of any released chip: the 297 reference points lie >=2.4 km from every chip centre.",
}

SOURCES = [
    dict(role="primary label", name="ESA WorldCover v200 (2021)", asset="ESA/WorldCover/v200",
         licence="CC BY 4.0", remap="10,20,...,100 -> 0..10 (see config), 255=nodata"),
    dict(role="agreement source / soft target", name="Dynamic World v1 (Jun-Nov mode label, mean probabilities)",
         asset="GOOGLE/DYNAMICWORLD/V1", licence="CC BY 4.0", remap="DW class -> our class via DW_TO_IDX=[7,0,2,8,3,1,4,5,6]"),
    dict(role="agreement source", name="Esri/Impact Observatory 10 m LULC time series (community asset)",
         asset="projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS", licence="CC BY 4.0 (check asset page)",
         remap="ESRI_FROM=[1,2,4,5,7,8,9,11] -> ESRI_TO=[7,0,8,3,4,5,6,2]"),
    dict(role="auxiliary feature", name="ESA WorldCereal 2021 temporary-crops model", asset="ESA/WorldCereal/2021/MODELS/v100",
         licence="CC BY 4.0", remap="classification>0 -> 1"),
    dict(role="input (optical)", name="Sentinel-2 SR Harmonized, SCL-masked mosaic", asset="COPERNICUS/S2_SR_HARMONIZED",
         licence="Copernicus open data"),
    dict(role="input (SAR)", name="Sentinel-1 GRD IW VV/VH mosaic (x100, int16)", asset="COPERNICUS/S1_GRD",
         licence="Copernicus open data"),
    dict(role="input (terrain)", name="SRTM 30 m elevation/slope/aspect", asset="USGS/SRTMGL1_003", licence="public domain"),
    dict(role="input (climate)", name="WorldClim v1 bio1/bio12", asset="WORLDCLIM/V1/BIO", licence="CC BY-SA 4.0 (check)"),
    dict(role="input (lights)", name="VIIRS DNB annual nighttime lights v21", asset="NOAA/VIIRS/DNB/ANNUAL_V21",
         licence="public (NOAA/EOG terms)"),
    dict(role="independent reference (not in chips)", name="street-view expert crop annotations (297 pts: 239 crop, 58 other)",
         asset="data/bench_v1/t2_fieldtruth_india.csv (origin: WorldCereal reference data, Zenodo 7825628)",
         licence="CC BY 4.0 (verify on the Zenodo record before release)"),
]


def _md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def chip_stats(root, man):
    """Per-chip and per-class source-agreement statistics, from the arrays themselves."""
    ad = bi.aux_dir(root)
    rows, cls = [], np.zeros((bi.N_CLASSES, 6), np.int64)   # px, agree_dw, agree_esri, conf>=2, conf==3, forced
    for r in man.itertuples():
        lbl = np.load(Path(root) / "lbl" / f"{r.chip}.npy")
        aux = np.load(Path(root) / ad / f"{r.chip}.npy", mmap_mode="r")
        dw, es, cf, wcc = (np.asarray(aux[:, :, AUX[k]]) for k in ("dw", "esri", "conf", "wc_crop"))
        v = lbl != bi.IGNORE
        n = int(v.sum())
        a_dw = (lbl == dw) & v
        a_es = (lbl == es) & v & (es != 255)
        a_de = (dw == es) & v & (es != 255)
        forced = v & np.isin(lbl, (9, 10))
        for k in range(bi.N_CLASSES):
            m = v & (lbl == k)
            if m.any():
                cls[k] += [m.sum(), (a_dw & m).sum(), (a_es & m).sum(), ((cf >= 2) & m).sum(), ((cf == 3) & m).sum(), (forced & m).sum()]
        rows.append(dict(chip=r.chip, split=r.split, lon=r.lon, lat=r.lat, zone=r.zone, valid_px=n,
                         agree_wc_dw=a_dw.sum() / max(n, 1), agree_wc_esri=a_es.sum() / max(n, 1),
                         agree_dw_esri=a_de.sum() / max(n, 1), frac_conf_ge2=((cf >= 2) & v).sum() / max(n, 1),
                         frac_conf_eq3=((cf == 3) & v).sum() / max(n, 1), frac_forced_conf=forced.sum() / max(n, 1),
                         wc_crop_frac_of_crop=float(((wcc > 0) & (lbl == 3)).sum() / max((lbl == 3).sum(), 1)),
                         n_rare_px=int(np.isin(lbl, bi.RARE_CLASSES).sum()), dominant=int(np.bincount(lbl[v], minlength=bi.N_CLASSES).argmax()) if n else -1))
    c = pd.DataFrame(cls, columns=["px", "agree_dw", "agree_esri", "conf_ge2", "conf_eq3", "forced_conf"])
    c.insert(0, "name", bi.CLASS_NAMES)
    for k in ("agree_dw", "agree_esri", "conf_ge2", "conf_eq3", "forced_conf"):
        c[k + "_frac"] = (c[k] / c.px.clip(lower=1)).round(4)
    return pd.DataFrame(rows), c


def reference_overlap(man, gt):
    d = bi.haversine_km(gt.x.values, gt.y.values, man.lon.values, man.lat.values)
    j = d.argmin(1)
    out = gt[["id", "x", "y", "cropType", "irrType", "is_crop_ref", "wc"]].copy()
    out["nearest_chip"] = man.chip.values[j]; out["nearest_split"] = man.split.values[j]
    out["dist_km"] = d.min(1).round(3); out["inside_chip"] = out.dist_km <= HALF_CHIP_KM * 1.42   # corner
    return out


def reference_audit(gt):
    """WorldCover value at each expert point vs the expert's crop / non-crop call."""
    crop = gt.is_crop_ref.astype(bool); wc_crop = gt.wc == 40
    return dict(n=int(len(gt)), n_crop=int(crop.sum()), n_other=int((~crop).sum()),
                worldcover_cropland_recall=float((wc_crop & crop).sum() / max(crop.sum(), 1)),
                worldcover_cropland_on_expert_other=float((wc_crop & ~crop).sum() / max((~crop).sum(), 1)),
                note="point-level audit of the label PRODUCT against expert annotations; not a model score")


def main(root, out):
    root, out = Path(root), Path(out); out.mkdir(parents=True, exist_ok=True)
    man = bi.load_manifest(root)
    gt = pd.read_csv(LAB / "data" / "bench_v1" / "t2_fieldtruth_india.csv")
    per_chip, per_cls = chip_stats(root, man)
    per_chip.round(5).to_csv(out / "chip_provenance.csv", index=False)
    per_cls.to_csv(out / "label_noise_by_class.csv", index=False)
    ov = reference_overlap(man, gt); ov.to_csv(out / "reference_overlap.csv", index=False)
    tgt = gt[["id", "x", "y", "cropType", "irrType", "is_crop_ref"]].rename(columns={"x": "lon", "y": "lat"})
    tgt = tgt.assign(year=2021, role="eval_only", never_in_train=True, source="street-view expert annotation")
    tgt.to_csv(out / "gt_target_chips.csv", index=False)
    sup = bi.class_support(root, man)
    prov = dict(
        dataset="bharat-eo-chips-dsv2-sp", n_chips_manifest=int(len(man)), n_chips_on_disk=int(len(list((root / "img").glob("*.npy")))),
        manifest_md5=_md5(root / "manifest.csv"), split_type=str(man.split_type.iloc[0]),
        splits=man.split.value_counts().to_dict(),
        min_dist_km=dict(train_val=bi.min_distance_km(man, "train", "val"), train_test=bi.min_distance_km(man, "train", "test")),
        chip_spec=dict(px=224, gsd_m=10, img_bands=bi.IMG_BANDS, aux_bands=bi.AUX_BANDS, aux_layout=AUX, conf_definition=
                       "1 + [WC==DW] + [WC==ESRI], capped at 3; mangrove/moss (WC classes 9,10) forced to 3"),
        sources=SOURCES, evidence_levels=EVIDENCE_LEVELS,
        independent_reference=dict(audit=reference_audit(gt), points_inside_any_chip=int(ov.inside_chip.sum()),
                                   nearest_point_to_chip_km=float(ov.dist_km.min())),
        class_support={f"{r.split}/{r.name}": dict(chips=int(r.chips), px=int(r.px)) for r in sup.itertuples()},
        exporter="src/export_dataset_v2.py", exporter_md5=_md5(LAB / "src" / "export_dataset_v2.py"),
        split_code="src/finalize_dsv2_spatial.py", split_code_md5=_md5(LAB / "src" / "finalize_dsv2_spatial.py"),
    )
    with open(out / "PROVENANCE.json", "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2, default=str)
    print("wrote", [p.name for p in sorted(out.iterdir())])
    return prov


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(LAB / "data" / "chips_dsv2_sp"))
    ap.add_argument("--out", default=str(LAB / "data" / "provenance"))
    a = ap.parse_args(); main(a.root, a.out)
