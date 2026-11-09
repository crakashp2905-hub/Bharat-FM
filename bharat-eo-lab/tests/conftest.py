import os, sys
from pathlib import Path
import numpy as np, pandas as pd, pytest

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))
import bench_integrity as bi  # noqa: E402,F401

DS_ROOT = Path(os.environ.get("BHARAT_DS", LAB / "data" / "chips_dsv2_sp"))


@pytest.fixture(scope="session")
def real_root():
    if not (DS_ROOT / "manifest.csv").exists():
        pytest.skip(f"dataset not present at {DS_ROOT} (set BHARAT_DS)")
    return DS_ROOT


@pytest.fixture(scope="session")
def real_man(real_root):
    return bi.load_manifest(real_root)


def _chip(root, name, lbl_vals, conf=3, rng=None):
    rng = rng or np.random.RandomState(0)
    for sub in ("img", "lbl", "auxb"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    img = rng.randint(0, 3000, (224, 224, 17)).astype(np.int16)
    lbl = np.full((224, 224), lbl_vals[0], np.uint8)
    for i, v in enumerate(lbl_vals[1:], 1):
        lbl[i * 20:(i * 20) + 40, :] = v
    aux = np.zeros((224, 224, 14), np.int16)
    aux[:, :, 12] = conf
    np.save(root / "img" / f"{name}.npy", img)
    np.save(root / "lbl" / f"{name}.npy", lbl)
    np.save(root / "auxb" / f"{name}.npy", aux)


@pytest.fixture()
def toy(tmp_path):
    """A tiny, valid spatial-block dataset: train near lon 70, val near 80, test near 90."""
    rows, n = [], 0
    rng = np.random.RandomState(1)
    layout = [("train", 70.0, 3, [3, 8, 9, 10]), ("val", 80.0, 2, [3, 6]), ("test", 90.0, 2, [3, 6])]
    for split, lon0, k, classes in layout:
        for i in range(k):
            name = f"c{n:03d}"
            n += 1
            _chip(tmp_path, name, [classes[i % len(classes)], 3] + classes, rng=rng)
            lon = lon0 + i * 2.0
            rows.append(dict(chip=name, lon=lon, lat=20.0, year=2021, zone=0, target=-1, dominant=3,
                             n_classes=3, valid_frac=1.0, conf_frac=1.0, block=f"{int(lon)}_20",
                             split=split, split_type="spatial_block_1.0deg_buf25.0km"))
    man = pd.DataFrame(rows)
    man.to_csv(tmp_path / "manifest.csv", index=False)
    return tmp_path, man
