# Bharat-EarthFM — Drafted Patent Claims (crown jewels)

Working claim drafts for the highest-value filings. **Not legal advice** — for
attorney review before filing. Strategy (from blueprint Part IX): provisional in
India first → PCT within 12 months → national phase (US/EU/CN/JP/AU). Under
**Indian Patents Act §3(k)** ("computer programme per se" exclusion), each is
framed as a **system/apparatus producing a tangible technical effect** (improved
sensor fusion under data loss; reduced bandwidth; a physical control/alert
output) — not as a bare algorithm. Weights/recipe/data pipeline are kept as
**trade secrets**, not disclosed here.

Full portfolio P01–P33: see blueprint Part IX (one-line claim cores). Below are
the five crown jewels with full claim language.

---

## P01 — Universal Indian-sensor dynamic tokenizer

**Field:** machine processing of multi-sensor Earth-observation imagery.

**Independent system claim 1.** An Earth-observation processing system comprising:
a sensor interface receiving image data from a plurality of heterogeneous
satellite sensors differing in spectral wavelength, polarization, ground sample
distance (GSD), and acquisition cadence, including at least one L-band and one
S-band synthetic-aperture-radar sensor and one geostationary imaging sensor; a
**hypernetwork** configured to generate, for each input channel, patch-embedding
convolution weights from a conditioning vector comprising the channel's centre
wavelength, bandwidth, polarization, GSD, sensor identity and acquisition
geometry; a tokenizer applying said generated weights to produce, for any said
sensor without sensor-specific retraining, embedding tokens in a common
representation space; and a transformer encoder operating on said tokens to
output a unified spatiotemporal representation used to generate a geophysical map.

**Independent method claim 9.** A method comprising: receiving channels from
heterogeneous satellite sensors; for each channel, computing patch-embedding
weights with a hypernetwork conditioned on {wavelength, bandwidth, polarization,
GSD, sensor-id, geometry}; tokenizing each channel with its computed weights into
a shared space; and encoding the tokens with a transformer to produce a surface
map, whereby a new sensor type is admitted at inference by supplying its
conditioning vector, without retraining.

**Dependent claims.** 2/10: wherein the geostationary sensor cadence is encoded
in the conditioning vector to fuse high-temporal data with polar-orbiting data.
3: wherein the polarization is one-hot over {none,VV,VH,HH,HV}. 4: wherein tokens
are produced at multiple scales forming a Laplacian pyramid for sub-5 m field
boundaries. 5: wherein the sensors comprise NISAR (L/S), EOS-04/RISAT (C), INSAT
(geostationary), and Resourcesat (AWiFS/LISS).

**Distinction over prior art:** DOFA/Clay condition patch embedding on wavelength
only; none admit polarization **and** geostationary cadence into one token space,
nor target NISAR L/S + INSAT.

---

## P02 — Monsoon-phase-aware temporal encoding (MPATE)

**Independent system claim 1.** A system for time-series Earth observation
comprising: an input for a temporally-sampled sequence of satellite observations
over a region; a phase estimator deriving, from meteorological rainfall data for
the region, a monsoon phase label selected from {pre-onset, onset, active, break,
withdrawal}; a temporal-encoding module producing positional encodings for the
sequence as a function of (i) said monsoon phase, (ii) an agro-calendar day index
indicating a cropping season, and (iii) absolute date; and a transformer applying
attention over the sequence using said encodings to output a crop, water, or
hazard map.

**Independent method claim 8.** A method comprising deriving a monsoon phase from
rainfall data; forming a temporal positional encoding from {monsoon-phase,
agro-calendar-day, date}; and applying time attention conditioned on said encoding
to irregularly-sampled satellite observations to produce a surface-state output.

**Dependent claims.** 2/9: wherein the agro-calendar index distinguishes kharif,
rabi and zaid seasons. 3: wherein the phase estimator uses a standardized
rainfall anomaly and dry-spell rule to separate active and break phases.
4: wherein the attention is continuous-time and gap-tolerant for cloud-interrupted
sequences. 5: wherein the rainfall data is a gridded daily product (e.g. IMD 0.25°).

**Distinction:** prior EO models condition time encodings on calendar date or
lat/lon (Prithvi-EO-2.0 "TL", Clay); none condition on a hydro-meteorological
phase variable.

---

## P03 — "Think-in-Modalities-then-Time" generative forecaster

**Independent system claim 1.** An Earth-observation forecasting system
comprising: an encoder producing a latent representation from multimodal
observations in which one or more modalities may be missing; a **modality-
completion decoder** generating, in latent space, a representation of a missing
modality from available modalities; and a **temporal-forecast decoder** rolling
the completed latent representation forward to one or more future times to output
a predicted future surface state, the system thereby first completing modalities
and then predicting future state.

**Independent method claim 9.** A method comprising: encoding multimodal EO data;
synthesizing a missing modality in latent space from present modalities; and
autoregressively or by latent diffusion advancing the representation to a future
time to output a forecast of flood extent, drought index, reservoir level, crop
biomass, or groundwater anomaly.

**Dependent claims.** 2/10: wherein the missing modality is cloud-obscured optical
synthesized from SAR and geostationary inputs. 3: wherein the temporal decoder is
a latent diffusion model conditioned on a numerical-weather forecast field
(Patent P21). 4: wherein forecasts are emitted at multiple horizons {7,15,30,60,90
days} with calibrated uncertainty. 5: wherein a physics-consistency penalty
constrains the forecast (Patent P07).

**Distinction:** TerraMind completes missing modalities but does not forecast
future timesteps; no EO foundation model predicts future surface states.

---

## P04 / P23 — Multimodal groundwater storage-change estimator (system)

**Independent system claim 1.** A groundwater-monitoring system comprising:
inputs for (a) interferometric-SAR-derived ground deformation, (b) satellite
gravimetry terrestrial-water-storage anomaly, and (c) in-situ groundwater-level
measurements from observation wells; a fusion model estimating a groundwater
storage-change field at a spatial resolution finer than the gravimetry input;
a constraint module penalizing departures of the estimate from a water-balance
relation (precipitation − evapotranspiration − runoff − storage-change ≈ 0); and
an output stage producing a depletion-risk map and a signal actuating a
groundwater-extraction-control or alert decision.

**Independent method claim 10.** A method comprising fusing InSAR deformation,
gravimetric water-storage anomaly, and well measurements under a learned
water-balance constraint to estimate a downscaled groundwater storage-change
field, and emitting an extraction-control or prioritization output therefrom.

**Dependent claims.** 2/11: wherein the gravimetry comprises GRACE/GRACE-FO
mascon solutions. 3: wherein the deformation is from Sentinel-1 or NISAR. 4:
wherein the estimate is aggregated to administrative districts and ranked for
intervention prioritization. 5: wherein well sparsity is propagated as an
uncertainty band via kriging variance. 6: wherein a foundation-model surface
embedding is an additional predictor in the fusion model.

**Distinction:** no prior system fuses InSAR + gravimetry + piezometry under a
water-balance constraint to output a control/prioritization decision; existing
GRACE downscaling uses surface drivers without InSAR or a physical constraint.

---

## P07 — Physics-informed EO adapter (water/energy balance)

**Independent system claim 1.** An Earth-observation inference system comprising
a pretrained encoder and a task head, and a **differentiable constraint layer**
that, during training and inference, computes a residual of a physical
conservation relation — a surface water balance and/or surface energy balance —
from the head's outputs and penalizes or projects the outputs to reduce said
residual, the system thereby producing physically-admissible geophysical maps.

**Independent method claim 8.** A method comprising producing a geophysical
estimate from EO data with a neural head, computing a water- or energy-balance
residual from the estimate, and adjusting the estimate to reduce the residual.

**Dependent claims.** 2/9: wherein the relation is P−ET−Q−ΔS. 3: wherein an
energy-balance residual constrains evapotranspiration. 4: wherein a reservoir
mass-balance residual constrains a reservoir-level estimate. 5: wherein the
penalty is applied as a soft loss during fine-tuning and a projection at inference.

**Distinction:** no EO foundation model embeds water/energy-balance conservation
as a differentiable constraint on outputs.

---

## Filing waves (recap)
- **Wave 1 (M0–6):** P01, P02, P03, P07, P12, P04/P23 — file provisionals now.
- **Wave 2 (M6–12):** P24, P25, P06, P13, P30, P33, P27.
- **Wave 3 (M12–24):** P05, P08, P21, P26, P31, P28.
- **Defensive publication:** P11, P16, P18, P22, P29, P32.
- **Trade secret (no filing):** trained weights, exact training recipe, full data
  pipeline.

## FTO / hygiene notes
- Run a freedom-to-operate search before each PCT national phase against IBM/NASA
  (Prithvi), ESA (TerraMind), Clay, Google (AlphaEarth).
- Pre-assign IP cleanly in ISRO/IN-SPACe/IndiaAI co-development agreements
  (government co-funding can create ownership ambiguity).
- Open-core balance: open -S weights + Bharat-EO-Bench; patent the **systems**;
  keep -L/-XL weights + recipe proprietary.
```
