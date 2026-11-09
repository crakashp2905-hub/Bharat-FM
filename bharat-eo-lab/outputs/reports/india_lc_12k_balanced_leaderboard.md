# Bharat-EO-Bench v0 -- India Land Cover 2021

- **Region:** India   **Year:** 2021   **Resolution:** 10 m
- **Samples:** 11,979   **Classes present:** 11 (tree_cover, shrubland, grassland, cropland, built_up, bare_sparse, snow_ice, water, herbaceous_wetland, mangroves, moss_lichen)
- **Split:** spatial block hold-out (1.0 deg, 20% test)
- **macro_f1_supported** = macro-F1 over classes with >= 10 test samples

| Rank | Model | Overall acc | Macro-F1 | Macro-F1 (supported) | #features |
|---|---|---|---|---|---|
| 1 | B2s AlphaEarth + XGBoost (SMOTE) | 0.861 | 0.675 | 0.789 | 64 |
| 2 | B2  AlphaEarth + XGBoost | 0.860 | 0.668 | 0.780 | 64 |
| 3 | B3  Embedding + geography | 0.860 | 0.643 | 0.781 | 66 |
| 4 | B2w AlphaEarth + XGBoost (class-weighted) | 0.858 | 0.625 | 0.783 | 64 |
| 5 | B1  Geography only (lat/lon) | 0.544 | 0.151 | 0.237 | 2 |
| 6 | B0  Majority class | 0.496 | 0.060 | 0.095 | 1 |

**Reading the result:** the embedding rows (B2*) should dominate B1 (geography) and B0 (majority). Class-balancing (B2w weighted / B2s SMOTE) targets rarer classes -- compare their macro-F1 against plain B2. Ultra-rare, spatially-clustered classes (e.g. mangroves) may remain hard because a spatial-block split can place them entirely in train or test.