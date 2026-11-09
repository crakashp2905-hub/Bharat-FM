# Assam Flood 2022 -- Bharat-EO-Bench (flood susceptibility)

- Region: Assam   Event: Jun-Jul 2022 (Brahmaputra)   Points: 12,000
- Labels from Sentinel-1 SAR (cloud-proof) + JRC permanent water
- Predictors: STATIC only (no during-flood SAR) -> non-circular susceptibility
- Class mix: dry 9621, permanent_water 371, flood_water 2008

| Rank | Model | Overall acc | Macro-F1 | Flood F1 | Flood recall |
|---|---|---|---|---|---|
| 1 | F4  Embedding + terrain + pre-SAR (all static) | 0.884 | 0.671 | 0.561 | 0.461 |
| 2 | F3  AlphaEarth embedding | 0.874 | 0.655 | 0.506 | 0.410 |
| 3 | F2  Pre-flood SAR only | 0.842 | 0.560 | 0.391 | 0.323 |
| 4 | F1  Terrain only (elev+slope) | 0.817 | 0.305 | 0.014 | 0.007 |
| 5 | F0  Majority (dry) | 0.818 | 0.300 | 0.000 | 0.000 |