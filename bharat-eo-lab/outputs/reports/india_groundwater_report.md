# Groundwater depletion (GRACE) -- NW India / Punjab

- GRACE TWS linear trend 2003-2016 (cm water/yr); 1,321 ~50 km cells
- Punjab mean trend: -0.86 cm/yr
- COARSE & correlational (GRACE ~110 km); not a field-scale product

| Model | R2 (spatial CV) | MAE (cm/yr) | #features |
|---|---|---|---|
| G1  Rainfall only | -0.142 | 0.526 | 1 |
| G2  Climate + agriculture | 0.335 | 0.361 | 4 |
| G3  + AlphaEarth embedding | 0.889 | 0.139 | 68 |

**Top drivers (G2):** elevation 0.33, cropland_frac 0.25, rain 0.23, ndvi 0.19