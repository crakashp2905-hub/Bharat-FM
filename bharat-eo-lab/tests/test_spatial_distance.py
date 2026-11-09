"""Spatial-distance tests: train and evaluation chips are provably far apart."""
import bench_integrity as bi


def test_haversine_known_values():
    d = bi.haversine_km([0.0], [0.0], [0.0], [1.0])[0, 0]
    assert abs(d - 111.19) < 0.1                                   # 1 degree of latitude
    d = bi.haversine_km([77.2090], [28.6139], [72.8777], [19.0760])[0, 0]
    assert 1140 < d < 1165                                         # Delhi -> Mumbai
    assert bi.haversine_km([10.0], [10.0], [10.0], [10.0])[0, 0] < 1e-6


def test_parse_buffer_km():
    assert bi.parse_buffer_km("spatial_block_1.0deg_buf25.0km") == 25.0


def test_real_min_train_to_eval_distance_meets_declared_buffer(real_man):
    buf = bi.parse_buffer_km(real_man.split_type.iloc[0])
    assert buf >= 25.0
    for ev in ("val", "test"):
        assert bi.min_distance_km(real_man, "train", ev) >= buf


def test_real_spatial_check_passes(real_man):
    assert bi.check_spatial(real_man, bi.parse_buffer_km(real_man.split_type.iloc[0])) == []


def test_real_no_chip_closer_than_one_chip_width_across_splits(real_man):
    # chips are 224 px x 10 m = 2.24 km wide; anything closer would overlap
    for ev in ("val", "test"):
        assert bi.min_distance_km(real_man, "train", ev) > 2.24


def test_detects_train_chip_next_to_test_chip(toy):
    _, man = toy
    bad = man.copy()
    t = bad[bad.split == "test"].iloc[0]
    bad.loc[0, ["lon", "lat"]] = [t.lon + 0.05, t.lat]             # ~5 km from a test chip
    assert any("min train->test" in x for x in bi.check_spatial(bad, 25.0))


def test_toy_passes_spatial(toy):
    _, man = toy
    assert bi.check_spatial(man, 25.0) == []


def test_buffer_distance_boundaries(toy):
    """Legal beyond the buffer, illegal inside it, regardless of block membership."""
    _, man = toy
    bad = man.copy()
    t = bad[bad.split == "test"].iloc[0]
    bad.loc[0, ["lon", "lat"]] = [t.lon - 1.01, t.lat]             # ~105 km: legal
    assert bi.check_spatial(bad, 25.0) == []
    bad.loc[0, ["lon", "lat"]] = [t.lon - 0.1, t.lat]              # ~10 km: illegal
    assert bi.check_spatial(bad, 25.0) != []
