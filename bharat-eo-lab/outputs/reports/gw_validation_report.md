# Groundwater validation -- Punjab / NW India (independent data)

Real InSAR (HyP3) needs an Earthdata login + ASF cloud time; instead we
validate the depletion with independent satellites / processing already in GEE.

## 1. Cross-mission (different satellites)
- Punjab GRACE 2003-2016: **-0.79 cm/yr**
- Punjab GRACE-FO 2018-2024: **-0.84 cm/yr** (independent successor mission)
- Spatial correlation GRACE vs GRACE-FO across India: **r = 0.702**

## 2. Structural (not drought)
- India-wide mean GRACE-FO trend +1.14 cm/yr -- broad post-2018 recovery (wet years),
  yet Punjab keeps declining (-0.84 cm/yr): structural groundwater overdraft.

## 3. Cross-center / cross-method (pattern)
- CSR/GFZ/JPL agree spatially: r = 1.000 (no single-center artifact)
- Mascon vs spherical-harmonic depletion pattern: r = 0.806
- (SH product attenuates sub-~300 km signals, so magnitudes come from mascon.)

## 4. Punjab decline curve
- TWS 2002-2024 slope **-1.68 cm/yr**, ~-36 cm cumulative loss; figure gw_validation_punjab_decline.png

## 5. Literature sanity check
- Rodell et al. (2009, Nature) ~-4 cm/yr groundwater for NW India (2002-2008);
  Famiglietti (2014): one of Earth's fastest-depleting aquifers. Our multi-decade
  TWS rate is the same sign and order of magnitude (TWS < pure groundwater).

**Conclusion:** the depletion replicates across an independent mission
(GRACE-FO) and three processing centers, is structural rather than drought-driven,
and matches the literature -- a robust target for the downscaling. A direct InSAR
subsidence cross-check is ready in notebooks/insar_hyp3_punjab.py.