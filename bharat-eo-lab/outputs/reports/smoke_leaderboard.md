# Bharat-EO-Bench v0 -- India Land Cover 2024

- **Region:** India   **Year:** 2024   **Resolution:** 10 m
- **Samples:** 4,500   **Classes present:** 9 (water, trees, grass, flooded_vegetation, crops, shrub_and_scrub, built, bare, snow_and_ice)
- **Split:** spatial block hold-out (1.0 deg, 20% test)

| Rank | Model | Overall accuracy | Macro-F1 | #features |
|---|---|---|---|---|
| 1 | B3  Embedding + geography | 0.999 | 0.999 | 66 |
| 2 | B2  AlphaEarth embedding + XGBoost | 0.999 | 0.999 | 64 |
| 3 | B1  Geography only (lat/lon) | 0.114 | 0.114 | 2 |
| 4 | B0  Majority class | 0.098 | 0.020 | 1 |

**Reading the result:** if B2 (embedding) clearly beats B1 (geography only) and B0 (majority), the AlphaEarth representation carries real land-cover signal -- the foundation for every downstream Bharat-EarthFM task.