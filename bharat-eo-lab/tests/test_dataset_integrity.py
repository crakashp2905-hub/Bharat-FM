"""Dataset-integrity tests: the manifest and chip files are what the benchmark claims."""
import numpy as np
import pandas as pd
import bench_integrity as bi


# ---- the released dataset ------------------------------------------------------------
def test_real_manifest_is_valid(real_man):
    assert bi.check_manifest(real_man) == []


def test_real_split_sizes_are_sane(real_man):
    n = real_man.split.value_counts()
    assert set(n.index) == {"train", "val", "test"}
    assert n["train"] > n["val"] and n["train"] > n["test"]
    assert n.min() >= 100


def test_real_manifest_rows_exist_on_disk(real_root, real_man):
    assert bi.check_no_orphans_in_manifest(real_root, real_man) == []


def test_real_chip_files_well_formed(real_root, real_man):
    assert bi.check_files(real_root, real_man, sample=150, seed=1) == []


def test_real_every_chip_has_valid_coordinates(real_man):
    assert real_man.lon.between(68, 98).all() and real_man.lat.between(6, 38).all()


def test_real_zones_cover_all_15(real_man):
    assert real_man.zone.nunique() == 15


def test_real_no_cross_split_duplicate_imagery(real_root, real_man):
    assert bi.check_cross_split_duplicates(real_root, real_man) == []


# ---- the checks themselves must detect corruption (guard against vacuous tests) -------
def test_toy_is_clean(toy):
    root, man = toy
    res = bi.run_all(root, sample_files=None, full_duplicates=True)
    assert all(not v for v in res.values()), res


def test_detects_duplicate_chip_ids(toy):
    _, man = toy
    bad = pd.concat([man, man.iloc[[0]]])
    assert any("duplicated" in x for x in bi.check_manifest(bad))


def test_detects_unknown_split_label(toy):
    _, man = toy
    bad = man.copy()
    bad.loc[0, "split"] = "dev"
    assert any("unknown split" in x for x in bi.check_manifest(bad))


def test_detects_non_spatial_split_type(toy):
    _, man = toy
    bad = man.copy()
    bad["split_type"] = "zone_stratified_noncoord"
    assert any("not a spatial_block" in x for x in bi.check_manifest(bad))


def test_detects_block_straddling_splits(toy):
    _, man = toy
    bad = man.copy()
    bad.loc[1, ["lon", "lat"]] = [bad.loc[0, "lon"] + 0.1, bad.loc[0, "lat"]]
    bad.loc[1, "split"] = "val"
    assert any("straddle" in x for x in bi.check_manifest(bad))


def test_detects_missing_file(toy):
    root, man = toy
    (root / "lbl" / f"{man.chip[0]}.npy").unlink()
    assert any("missing lbl" in x for x in bi.check_files(root, man))


def test_detects_bad_label_value(toy):
    root, man = toy
    f = root / "lbl" / f"{man.chip[0]}.npy"
    l = np.load(f)
    l[0, 0] = 42
    np.save(f, l)
    assert any("outside" in x for x in bi.check_files(root, man))


def test_detects_wrong_shape(toy):
    root, man = toy
    np.save(root / "img" / f"{man.chip[0]}.npy", np.zeros((10, 10, 17), np.int16))
    assert any("img shape" in x for x in bi.check_files(root, man))


def test_detects_cross_split_duplicate_image(toy):
    root, man = toy
    tr, ev = man.chip[0], man[man.split == "test"].chip.iloc[0]
    for sub in ("img", "lbl"):
        np.save(root / sub / f"{ev}.npy", np.load(root / sub / f"{tr}.npy"))
    assert bi.check_cross_split_duplicates(root, man) != []
