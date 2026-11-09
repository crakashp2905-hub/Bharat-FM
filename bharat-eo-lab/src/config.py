"""Central configuration for bharat-eo-lab.

Everything tunable lives here so the rest of the code stays declarative.
Override any value from the CLI (see run.py) without editing this file.
"""
from pathlib import Path

# ---- paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
OUT_FIG = ROOT / "outputs" / "figures"
OUT_REP = ROOT / "outputs" / "reports"
for _p in (DATA_RAW, DATA_PROC, OUT_FIG, OUT_REP):
    _p.mkdir(parents=True, exist_ok=True)

# ---- Earth Engine ----------------------------------------------------------
# You must set this to YOUR registered Google Cloud / Earth Engine project id.
EE_PROJECT = "bharat-fm-500317"

# AlphaEarth Foundations annual satellite embeddings (64-d, 10 m), free in GEE.
EMBED_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
# Google Dynamic World near-real-time land cover (kept for reference).
DW_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"
# ESA WorldCover v200 (2021): single static 10 m global land-cover map -> free,
# standard labels that are cheap to sample (no temporal reduction needed).
WORLDCOVER = "ESA/WorldCover/v200"
# FAO GAUL admin boundaries (country + state) for the AOI and spatial splits.
GAUL0 = "FAO/GAUL/2015/level0"
GAUL1 = "FAO/GAUL/2015/level1"
COUNTRY = "India"

YEAR = 2021            # analysis year (aligned with WorldCover v200 = 2021)
SCALE = 10             # metres (native AlphaEarth / WorldCover resolution)

# ---- sampling --------------------------------------------------------------
# randomPoints over India -> natural class distribution; cheap (no region scan).
N_POINTS_LOCAL = 12000    # total points (getInfo path, auto-batched <5k/call)
N_POINTS_DRIVE = 40000    # total points (Export-to-Drive path) -> robust/large
SEED = 42

# ---- features --------------------------------------------------------------
EMB_COLS = [f"A{i:02d}" for i in range(64)]   # AlphaEarth band names A00..A63

# ---- labels: ESA WorldCover v200 classes -----------------------------------
CLASS_NAMES = {
    10: "tree_cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built_up",
    60: "bare_sparse",
    70: "snow_ice",
    80: "water",
    90: "herbaceous_wetland",
    95: "mangroves",
    100: "moss_lichen",
}
CLASS_COLORS = {
    10: "#006400", 20: "#ffbb22", 30: "#ffff4c", 40: "#f096ff", 50: "#fa0000",
    60: "#b4b4b4", 70: "#f0f0f0", 80: "#0064c8", 90: "#0096a0", 95: "#00cf75",
    100: "#fae6a0",
}

# ---- spatial split (honest, leakage-free evaluation) -----------------------
BLOCK_DEG = 1.0        # size of spatial blocks (degrees) held out whole
TEST_FRAC = 0.2        # fraction of blocks reserved for the test set
MIN_SUPPORT = 10       # min test samples for a class to count in macro_f1_supported

# ---- model -----------------------------------------------------------------
XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,            # use all CPU threads
    tree_method="hist",  # fast histogram method, CPU-friendly
    random_state=SEED,
)

DEFAULT_CSV = DATA_RAW / f"india_landcover_{YEAR}.csv"
SMOKE_CSV = DATA_RAW / "_synthetic_smoke.csv"
BOOST_CSV = DATA_RAW / "india_rare_boost.csv"
COMBINED_CSV = DATA_RAW / f"india_landcover_{YEAR}_boosted.csv"

# ---- targeted rare-class sampling ------------------------------------------
# For spatially-clustered rare classes, randomPoints inside the regions where
# they actually occur. Multiple boxes per class -> points land in several 1-deg
# blocks so the spatial split keeps train+test coverage. Boxes = [W, S, E, N].
RARE_BOXES = {
    95: [  # mangroves
        [88.00, 21.50, 89.20, 22.60],   # Sundarbans (WB)
        [86.75, 20.55, 87.15, 20.95],   # Bhitarkanika (Odisha)
        [82.20, 16.20, 82.60, 16.95],   # Godavari-Krishna delta (AP)
        [79.70, 11.35, 79.95, 11.55],   # Pichavaram (TN)
        [69.30, 22.20, 70.60, 23.00],   # Gulf of Kutch (Gujarat)
        [92.50, 11.40, 93.20, 13.60],   # Andaman
    ],
    90: [  # herbaceous_wetland
        [85.00, 19.45, 85.75, 20.00],   # Chilika (Odisha)
        [68.50, 23.30, 71.80, 24.80],   # Rann of Kutch
        [93.70, 24.35, 93.95, 24.75],   # Loktak (Manipur)
        [81.05, 16.45, 81.45, 16.85],   # Kolleru (AP)
        [89.90, 25.90, 94.60, 27.20],   # Brahmaputra floodplain (Assam)
    ],
    100: [  # moss_lichen (high Himalaya)
        [76.80, 33.40, 78.60, 34.80],   # Ladakh
        [88.45, 27.65, 88.95, 28.20],   # Sikkim high
        [78.90, 30.45, 80.10, 31.40],   # Uttarakhand high
        [77.40, 31.90, 78.70, 32.70],   # Spiti (HP)
    ],
    70: [  # snow_ice (high Himalaya)
        [76.30, 33.90, 78.20, 35.60],   # Karakoram / Ladakh
        [78.40, 30.70, 79.90, 31.70],   # HP / Uttarakhand high
        [88.10, 27.75, 88.85, 28.25],   # Sikkim
    ],
}
PER_BOX_POINTS = 2500    # random points per box (target class kept after filter)
