# %% [markdown]
# # Real InSAR subsidence for Punjab via ASF HyP3 (the genuine InSAR path)
#
# Earth Engine can't do interferometry (no SLC phase). **ASF HyP3** processes
# Sentinel-1 **SLC** into InSAR displacement on ASF's cloud, driven by a Python
# SDK -- your laptop just orchestrates (control-plane pattern). Land subsidence
# from aquifer compaction is a direct, high-resolution groundwater-loss signal,
# and fuses with the GRACE downscaling in `src/gw_downscale.py`.
#
# **Prereqs (free):** a NASA Earthdata login (https://urs.earthdata.nasa.gov),
# then `pip install hyp3_sdk asf_search rasterio numpy`.
#
# NOTE: HyP3 jobs take ~20-40 min each on ASF's side and you get a monthly
# processing quota. Start with a few pairs over a small AOI.

# %%
import asf_search as asf
from hyp3_sdk import HyP3
import datetime as dt

# --- 1. AOI: a depleted Punjab/Haryana box (edit to taste) ------------------
# (south Haryana / Patiala-Sangrur belt are strong depletion zones)
AOI_WKT = "POLYGON((75.6 30.0, 76.4 30.0, 76.4 30.6, 75.6 30.6, 75.6 30.0))"

# %%
# --- 2. find Sentinel-1 SLC scenes over the AOI -----------------------------
results = asf.geo_search(
    platform=asf.PLATFORM.SENTINEL1,
    processingLevel=asf.PRODUCT_TYPE.SLC,
    beamMode=asf.BEAMMODE.IW,
    intersectsWith=AOI_WKT,
    start=dt.date(2019, 1, 1), end=dt.date(2023, 12, 31),
    flightDirection=asf.FLIGHT_DIRECTION.DESCENDING,
)
scenes = sorted({r.properties["sceneName"]: r for r in results}.items())
print(f"{len(scenes)} SLC scenes found")

# Build ~yearly reference->secondary pairs (long temporal baseline captures the
# slow, cumulative subsidence that tracks multi-year groundwater loss).
names = [n for n, _ in scenes]
pairs = [(names[i], names[i + 12]) for i in range(0, len(names) - 12, 12)][:6]
print("pairs:", pairs)

# %%
# --- 3. submit InSAR jobs to HyP3 (runs on ASF cloud) -----------------------
hyp3 = HyP3()  # prompts Earthdata login the first time
batch = []
for ref, sec in pairs:
    batch.append(hyp3.submit_insar_job(
        ref, sec, name="punjab_gw_insar",
        include_displacement_maps=True,   # vertical displacement (m)
        include_look_vectors=True))
batch = hyp3.watch(hyp3.Batch(batch))    # blocks until jobs finish
batch.download_files(location="insar_products")
print("downloaded InSAR products -> insar_products/")

# %% [markdown]
# ## 4. From interferograms to a subsidence-rate map
# Each product contains a vertical-displacement GeoTIFF. Stack them, divide by
# the time spans, and take the median -> a line-of-sight / vertical subsidence
# **rate (cm/yr)**. (For research-grade results use a proper SBAS/MintPy
# time-series with atmospheric and DEM-error corrections; this is the quick-look.)

# %%
import glob, numpy as np, rasterio
disp_files = glob.glob("insar_products/**/*displacement*.tif", recursive=True)
rates = []
for f in disp_files:
    with rasterio.open(f) as ds:
        rates.append(ds.read(1))     # metres of displacement per pair (~1 yr)
rate = np.nanmedian(np.stack(rates), axis=0) * 100.0   # cm/yr (negative = sinking)
print("median subsidence rate map shape:", rate.shape,
      "| 5th pct (cm/yr):", round(float(np.nanpercentile(rate, 5)), 2))

# %% [markdown]
# ## 5. Fuse with the GRACE downscaling
# InSAR subsidence is an *independent, high-resolution* groundwater-loss proxy.
# Combine with the downscaled GRACE trend from `gw_downscale.py`:
#  - **Validation:** correlate InSAR subsidence rate vs downscaled GRACE trend
#    per district -- agreement is real cross-sensor evidence of depletion.
#  - **Fusion:** use InSAR as an extra predictor (or a constraint) to sharpen
#    the downscaled map where wells are absent. This is patent concept **P4**
#    (multimodal groundwater estimator: InSAR + GRACE + [CGWB wells]).
#
# This notebook + the laptop's GRACE downscaling together realise the blueprint's
# groundwater module end-to-end, with the heavy InSAR compute on ASF's cloud.
