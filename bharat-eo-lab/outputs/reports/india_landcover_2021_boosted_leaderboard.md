# Bharat-EO-Bench v0 -- India Land Cover 2021

- **Region:** India   **Year:** 2021   **Resolution:** 10 m
- **Samples:** 42,045   **Classes present:** 11 (tree_cover, shrubland, grassland, cropland, built_up, bare_sparse, snow_ice, water, herbaceous_wetland, mangroves, moss_lichen)
- **Split:** spatial block hold-out (1.0 deg, 20% test)
- **macro_f1_supported** = macro-F1 over classes with >= 10 test samples

| Rank | Model | Overall acc | Macro-F1 | Macro-F1 (supported) | #features |
|---|---|---|---|---|---|
| 1 | B2  AlphaEarth + XGBoost | 0.858 | 0.801 | 0.782 | 64 |
| 2 | B2w AlphaEarth + XGBoost (class-weighted) | 0.858 | 0.798 | 0.778 | 64 |
| 3 | B2s AlphaEarth + XGBoost (SMOTE) | 0.856 | 0.791 | 0.770 | 64 |
| 4 | B4  AlphaEarth + MLP (neural net) | 0.862 | 0.782 | 0.760 | 64 |
| 5 | B3  Embedding + geography | 0.857 | 0.779 | 0.764 | 66 |
| 6 | B1  Geography only (lat/lon) | 0.504 | 0.228 | 0.236 | 2 |
| 7 | B0  Majority class | 0.443 | 0.056 | 0.061 | 1 |

**Reading the result:** the embedding rows (B2*) should dominate B1 (geography) and B0 (majority). Class-balancing (B2w weighted / B2s SMOTE) targets rarer classes -- compare their macro-F1 against plain B2. Ultra-rare, spatially-clustered classes (e.g. mangroves) may remain hard because a spatial-block split can place them entirely in train or test.