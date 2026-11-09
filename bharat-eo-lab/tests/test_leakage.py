"""Leakage tests: nothing from val/test may reach training -- in data, pools or statistics."""
import pytest
import bench_integrity as bi


# ---- real data -------------------------------------------------------------------------
def test_real_splits_are_disjoint(real_man):
    assert (real_man.groupby("chip").split.nunique() == 1).all()


def test_real_rare_pool_is_train_only(real_root, real_man):
    pool = bi.build_rare_pool(real_root, real_man)
    bi.assert_pool_train_only(list(pool), real_man)


def test_real_target_column_cannot_drive_a_pool(real_man):
    """Regression: `target` is -1 for every chip, so a target-based pool is empty and
    the original copy-paste augmentation was a silent no-op in every reported run."""
    assert (real_man.target == -1).all()
    assert real_man.target.isin(bi.RARE_CLASSES).sum() == 0


def test_real_label_derived_pool_is_not_empty(real_root, real_man):
    assert len(bi.build_rare_pool(real_root, real_man)) > 0


def test_real_pool_reports_rare_classes_it_cannot_serve(real_root, real_man):
    pool = bi.build_rare_pool(real_root, real_man)
    missing = bi.check_pool_covers_rare(pool, None)
    print("pool cannot serve:", missing)
    assert isinstance(missing, list)


def test_real_eval_split_pool_would_be_rejected(real_root, real_man):
    ev_pool = bi.build_rare_pool(real_root, real_man, split="test")
    if ev_pool:
        with pytest.raises(bi.IntegrityError):
            bi.assert_pool_train_only(list(ev_pool), real_man)


def test_real_unmeasurable_classes_are_flagged(real_root, real_man):
    sup = bi.class_support(real_root, real_man)
    um = bi.unmeasurable_classes(sup, "test", min_chips=5)
    print("unmeasurable on test (<5 chips):", um)
    assert isinstance(um, list)


# ---- the guards must actually fire ----------------------------------------------------------
def test_toy_pool_is_train_only(toy):
    root, man = toy
    pool = bi.build_rare_pool(root, man)
    assert set(pool) <= set(man.loc[man.split == "train", "chip"])
    bi.assert_pool_train_only(list(pool), man)


def test_pool_built_from_whole_manifest_is_caught(toy):
    """The original bug: a pool drawn from every split."""
    root, man = toy
    leaked = []
    for s in ("train", "val", "test"):
        leaked += list(bi.build_rare_pool(root, man, classes=(6, 8, 9, 10), split=s))
    assert any(c in set(man[man.split != "train"].chip) for c in leaked)
    with pytest.raises(bi.IntegrityError):
        bi.assert_pool_train_only(leaked, man)


def test_pool_with_unknown_chip_is_caught(toy):
    _, man = toy
    with pytest.raises(bi.IntegrityError):
        bi.assert_pool_train_only(["not_a_chip"], man)


def test_target_based_pool_is_empty_so_label_based_is_required(toy):
    _, man = toy
    assert man.target.isin(bi.RARE_CLASSES).sum() == 0


def test_toy_pool_serves_requested_classes(toy):
    root, man = toy
    pool = bi.build_rare_pool(root, man, classes=(8, 9, 10))
    assert {k for v in pool.values() for k in v} == {8, 9, 10}


def test_pool_min_px_threshold_respected(toy):
    root, man = toy
    assert bi.build_rare_pool(root, man, min_px=10 ** 9) == {}


def test_class_support_counts_labels_per_split(toy):
    root, man = toy
    sup = bi.class_support(root, man)
    assert sup[(sup.split == "train") & (sup.cls == 8)].chips.iloc[0] >= 1
    assert sup[(sup.split == "test") & (sup.cls == 8)].chips.iloc[0] == 0
    assert "wetland" in bi.unmeasurable_classes(sup, "test", min_chips=1)
