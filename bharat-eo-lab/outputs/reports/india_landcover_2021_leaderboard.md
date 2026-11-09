# Bharat-EO-Bench v0 -- India Land Cover 2021

- **Region:** India   **Year:** 2021   **Resolution:** 10 m
- **Samples:** 39,899   **Classes present:** 11 (tree_cover, shrubland, grassland, cropland, built_up, bare_sparse, snow_ice, water, herbaceous_wetland, mangroves, moss_lichen)
- **Split:** spatial block hold-out (1.0 deg, 20% test)
- **macro_f1_supported** = macro-F1 over classes with >= 10 test samples

| Rank | Model | Overall acc | Macro-F1 | Macro-F1 (supported) | #features |
|---|---|---|---|---|---|
| 1 | B2s AlphaEarth + XGBoost (SMOTE) | 0.852 | 0.694 | 0.739 | 64 |
| 2 | B2w AlphaEarth + XGBoost (class-weighted) | 0.853 | 0.681 | 0.748 | 64 |
| 3 | B2  AlphaEarth + XGBoost | 0.854 | 0.663 | 0.744 | 64 |
| 4 | B3  Embedding + geography | 0.855 | 0.657 | 0.736 | 66 |
| 5 | B1  Geography only (lat/lon) | 0.543 | 0.167 | 0.204 | 2 |
| 6 | B0  Majority class | 0.460 | 0.057 | 0.070 | 1 |

**Reading the result:** the embedding rows (B2*) should dominate B1 (geography) and B0 (majority). Class-balancing (B2w weighted / B2s SMOTE) targets rarer classes -- compare their macro-F1 against plain B2. Ultra-rare, spatially-clustered classes (e.g. mangroves) may remain hard because a spatial-block split can place them entirely in train or test.