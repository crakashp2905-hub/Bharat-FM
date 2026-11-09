const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat, TabStopType, TabStopPosition,
  TableOfContents, HeadingLevel, BorderStyle, WidthType, ShadingType,
  PageNumber, PageBreak, VerticalAlign
} = require("docx");

const CW = 10080; // content width: US Letter, 0.75" margins
const border = { style: BorderStyle.SINGLE, size: 1, color: "BFBFBF" };
const borders = { top: border, bottom: border, left: border, right: border };
const HEAD = "1F3864", ALT = "EAF0F8";

// ---------- helpers ----------
function tr(text, o = {}) {
  return new TextRun({ text: String(text), bold: !!o.bold, italics: !!o.italics,
    color: o.color, size: o.size, font: o.font });
}
function h1(t){ return new Paragraph({ heading: HeadingLevel.HEADING_1, children:[tr(t)] }); }
function h2(t){ return new Paragraph({ heading: HeadingLevel.HEADING_2, children:[tr(t)] }); }
function h3(t){ return new Paragraph({ heading: HeadingLevel.HEADING_3, children:[tr(t)] }); }
function p(...runs){
  const ch = runs.map(r => typeof r === "string" ? tr(r) : tr(r.t, r));
  return new Paragraph({ spacing:{ after:120 }, children: ch });
}
function lead(label, rest){ return new Paragraph({ spacing:{after:120}, children:[ tr(label,{bold:true}), tr(rest) ] }); }
function bullet(...runs){
  const ch = runs.map(r => typeof r === "string" ? tr(r) : tr(r.t, r));
  return new Paragraph({ numbering:{reference:"bullets",level:0}, spacing:{after:40}, children: ch });
}
function bl(label, rest){ return new Paragraph({ numbering:{reference:"bullets",level:0}, spacing:{after:40}, children:[ tr(label,{bold:true}), tr(rest) ] }); }
function numN(ref, ...runs){
  const ch = runs.map(r => typeof r === "string" ? tr(r) : tr(r.t, r));
  return new Paragraph({ numbering:{reference:ref,level:0}, spacing:{after:40}, children: ch });
}
function nl(ref, label, rest){ return new Paragraph({ numbering:{reference:ref,level:0}, spacing:{after:40}, children:[ tr(label,{bold:true}), tr(rest) ] }); }
function mono(t){ return new Paragraph({ spacing:{after:0}, children:[ new TextRun({ text:t, font:"Consolas", size:13 }) ] }); }
function spacer(){ return new Paragraph({ spacing:{after:60}, children:[tr("")] }); }

function cell(text, w, o = {}){
  return new TableCell({
    borders, width:{ size:w, type:WidthType.DXA },
    verticalAlign: VerticalAlign.CENTER,
    shading: o.fill ? { fill:o.fill, type:ShadingType.CLEAR } : undefined,
    margins:{ top:50, bottom:50, left:90, right:90 },
    children:[ new Paragraph({ alignment:o.align, children:[ tr(text,{ bold:o.bold, color:o.white?"FFFFFF":undefined, size:o.size||16 }) ] }) ]
  });
}
function table(colWidths, headers, rows, size=16){
  const hdr = new TableRow({ tableHeader:true, children: headers.map((h,i)=>cell(h, colWidths[i], {bold:true, fill:HEAD, white:true, size})) });
  const body = rows.map((r,ri)=> new TableRow({ children: r.map((c,i)=> cell(c, colWidths[i], { size, fill: ri%2?ALT:undefined })) }));
  return new Table({ width:{ size: colWidths.reduce((a,b)=>a+b,0), type:WidthType.DXA }, columnWidths: colWidths, rows:[hdr, ...body] });
}

// ---------- numbering config ----------
const numConfig = [
  { reference:"bullets", levels:[{ level:0, format:LevelFormat.BULLET, text:"•", alignment:AlignmentType.LEFT, style:{ paragraph:{ indent:{ left:460, hanging:240 } } } }] },
];
for (let i=1;i<=14;i++) numConfig.push({ reference:"n"+i, levels:[{ level:0, format:LevelFormat.DECIMAL, text:"%1.", alignment:AlignmentType.LEFT, style:{ paragraph:{ indent:{ left:460, hanging:260 } } } }] });

// ============================================================
// CONTENT
// ============================================================
const body = [];

// ---- COVER ----
body.push(new Paragraph({ spacing:{ before:2600 }, children:[] }));
body.push(new Paragraph({ alignment:AlignmentType.CENTER, spacing:{after:120},
  children:[ new TextRun({ text:"BHARAT-EarthFM", bold:true, size:72, color:"1F3864" }) ] }));
body.push(new Paragraph({ alignment:AlignmentType.CENTER, spacing:{after:80},
  border:{ bottom:{ style:BorderStyle.SINGLE, size:12, color:"C55A11", space:8 } },
  children:[ new TextRun({ text:"Sovereign India-Scale Earth Observation Foundation Model", size:30, color:"404040" }) ] }));
body.push(new Paragraph({ alignment:AlignmentType.CENTER, spacing:{before:240, after:60},
  children:[ new TextRun({ text:"Technical Blueprint & Strategy Dossier", bold:true, size:28 }) ] }));
body.push(new Paragraph({ alignment:AlignmentType.CENTER, spacing:{after:600},
  children:[ new TextRun({ text:"Landscape · Research Gaps · Architecture · Data · Training · Infrastructure · Go-to-Market · Patents", italics:true, size:20, color:"595959" }) ] }));
body.push(new Paragraph({ alignment:AlignmentType.CENTER, spacing:{after:40}, children:[ new TextRun({ text:"Version 1.0  —  June 2026", size:22 }) ] }));
body.push(new Paragraph({ alignment:AlignmentType.CENTER, spacing:{after:40}, children:[ new TextRun({ text:"Prepared by: Bharat-EarthFM Founding Team", size:22 }) ] }));
body.push(new Paragraph({ alignment:AlignmentType.CENTER, children:[ new TextRun({ text:"CONFIDENTIAL — For internal, partner & investor use", size:18, color:"C00000", bold:true }) ] }));
body.push(new Paragraph({ children:[ new PageBreak() ] }));

// ---- TOC ----
body.push(h1("Table of Contents"));
body.push(new TableOfContents("Table of Contents", { hyperlink:true, headingStyleRange:"1-2" }));
body.push(new Paragraph({ children:[ new PageBreak() ] }));

// ---- EXECUTIVE SUMMARY ----
body.push(h1("Executive Summary"));
body.push(p("Bharat-EarthFM is a sovereign, India-scale Earth Observation (EO) foundation model designed to outperform existing global EO models on Indian geospatial tasks across agriculture, monsoons, floods, droughts, reservoirs, groundwater, infrastructure, climate, land use, urban growth, coastal change, and disaster management."));
body.push(p("Three structural facts make India a distinct EO modeling regime rather than a transfer-learning afterthought:"));
body.push(numN("n1","The monsoon destroys the optical signal exactly when agriculture happens — the kharif crop grows June–September under 60–80% cloud cover, while every optical foundation model assumes near-clear scenes."));
body.push(numN("n1","Indian fields are sub-pixel for the sensors the incumbents were trained on — mean holding ≈1.08 ha, often <0.4 ha, multi-cropped; 10–30 m models produce mixed pixels at the unit of Indian agricultural decision-making."));
body.push(numN("n1","The highest-value Indian problems are sub-surface or hydrological (groundwater, reservoirs, flood/drought forecasting) — invisible to optical reflectance and untouched by any EO foundation model."));
body.push(p("Bharat-EarthFM is therefore defined by four pillars the incumbents lack: monsoon-phase-aware temporal modeling, cloud-resilient multi-sensor generative fusion, sub-surface / hydrological reasoning, and field-scale (≤5 m) resolution — wrapped in a sovereign data, compute, and governance stack."));
body.push(new Paragraph({ children:[ new PageBreak() ] }));

// ============================================================
// PART I
// ============================================================
body.push(h1("Part I — Landscape Analysis of Existing EO Foundation Models"));
body.push(p("For each model: architecture, parameters, training strategy, datasets, weaknesses, and the specific opportunity it opens for Bharat-EarthFM."));

const models = [
  ["1.1 Prithvi-1.0 (IBM + NASA, 2023)", [
    ["Architecture:", " ViT encoder, Masked Autoencoder (MAE); first geospatial FM with 3D patch embedding for short time series."],
    ["Parameters:", " ~100M."],
    ["Training:", " MAE reconstruction, ~0.75 masking, single sensor."],
    ["Datasets:", " Harmonized Landsat-Sentinel (HLS) over CONUS, 6 bands, 30 m."],
    ["Weaknesses:", " US-only; optical-only (no SAR); short temporal context; 30 m too coarse for Indian fields; no geolocation; descriptive only."],
    ["Opportunity:", " Inherit the 3D-patch idea; replace CONUS HLS with an Indian multi-sensor corpus and calendar time with monsoon-phase temporal embeddings."]]],
  ["1.2 Prithvi-EO-2.0 (IBM + NASA + Jülich, Dec 2024)", [
    ["Architecture:", " ViT-L (300M) and ViT-H (600M), MAE with 3D patch + 3D positional embeddings; TL variants inject geolocation + acquisition date."],
    ["Parameters:", " 300M and 600M."],
    ["Training:", " MAE; 600M trained on 240 GPUs, ~58,000 GPU-hours, 400 epochs, global batch 3,840."],
    ["Datasets:", " 4.2M global HLS time-series samples, 6 bands, 30 m (HLS V2)."],
    ["Weaknesses:", " Optical-only HLS; 30 m; no SAR/weather fusion; no generative gap-fill; no native forecasting."],
    ["Opportunity:", " Use as warm-start teacher for the optical branch and as the public baseline Bharat-EarthFM must beat on Indian tasks."]]],
  ["1.3 Clay v1 / v1.5 (Clay Foundation, 2024–2025)", [
    ["Architecture:", " ViT MAE + DINOv2 teacher; dynamic embedding block conditioned on each band's wavelength (any-sensor); GSD-scaled positional encoding + lat/lon + time."],
    ["Parameters:", " ~632M; 768-d embeddings."],
    ["Training:", " Masked reconstruction + DINOv2 self-distillation, multi-sensor."],
    ["Datasets:", " Sentinel-2, Landsat, Sentinel-1 SAR, NAIP, LINZ, MODIS (global)."],
    ["Weaknesses:", " Embedding-centric (weak task-native forecasting); no physics; no sub-surface; no monsoon/India tuning; not generative across modalities."],
    ["Opportunity:", " Adopt the wavelength-conditioned dynamic patch embedding to ingest Indian sensors (AWiFS, LISS, EOS-04, NISAR, INSAT) without per-sensor retraining."]]],
  ["1.4 SatMAE / SatMAE++ (Stanford, 2022 / 2023)", [
    ["Architecture:", " ViT-L MAE with spectral-group encoding and temporal embedding; SatMAE++ adds multi-scale reconstruction."],
    ["Parameters:", " ViT-L (~300M class)."],
    ["Training / Datasets:", " MAE on fMoW-RGB and fMoW-Sentinel."],
    ["Weaknesses:", " Scene-classification bias; weak dense prediction; no SAR; no India context."],
    ["Opportunity:", " Reuse spectral-group tokenization for India's varied band sets; weight dense + temporal tasks over classification."]]],
  ["1.5 Scale-MAE (Berkeley, 2023)", [
    ["Architecture:", " ViT-L MAE with GSD-aware positional encoding and a Laplacian-pyramid decoder (scale-aware)."],
    ["Parameters / Datasets:", " ViT-L; fMoW RGB."],
    ["Weaknesses:", " RGB-only; no temporal/SAR; classification-leaning."],
    ["Opportunity:", " GSD-conditioned encoding + pyramid decoder is the mechanism to fuse 56 m AWiFS to 5.8 m LISS-IV into one multi-resolution pyramid with super-resolution to field scale."]]],
  ["1.6 DOFA — Dynamic One-For-All (2024)", [
    ["Architecture:", " ViT with a hypernetwork that generates patch-embedding weights from input wavelengths, unifying RGB/MS/HS/SAR."],
    ["Parameters:", " DOFA-Large ≈ 330M."],
    ["Weaknesses:", " No temporal forecasting; no weather/in-situ; no physics; generic geography."],
    ["Opportunity:", " The hypernetwork-by-wavelength is the most elegant any-sensor mechanism; combine with Clay's dynamic embedding as the sensor-agnostic front-end."]]],
  ["1.7 AnySat (2024–2025)", [
    ["Architecture:", " I-JEPA-style multimodal, multi-resolution; merges sub-patch tokens at multiple resolutions."],
    ["Training / Datasets:", " JEPA latent prediction on GeoPlex."],
    ["Weaknesses:", " Research-grade; limited productization; no physics/forecasting; no India tuning."],
    ["Opportunity:", " JEPA latent objective is sample-efficient and avoids reconstructing cloud/noise — strong secondary pretraining objective for cloud-heavy Indian optical."]]],
  ["1.8 Galileo (NASA Harvest / AI2 consortium, 2025)", [
    ["Architecture:", " Multimodal ViT; dual global + local objective; tokenizes into spatial patches × timesteps × logical channel groups."],
    ["Modalities:", " Multispectral, SAR, elevation, weather/climate, pseudo-labels — broadest ingestion to date."],
    ["Training:", " ~127k globally clustered samples (9 data types), 500 epochs."],
    ["Weaknesses:", " Small corpus; representation-focused (not generative/forecasting); no in-situ hydrology; global."],
    ["Opportunity:", " Adopt the dual global/local objective and channel-group tokenization; extend with in-situ (CGWB/CWC/IMD), generative fusion, and forecasting."]]],
  ["1.9 TerraMind (IBM + ESA + Jülich, ICCV 2025)", [
    ["Architecture:", " First any-to-any generative EO FM; dual-scale encoder-decoder (pixel + token); Thinking-in-Modalities (TiM) generates missing modalities."],
    ["Training:", " 500B tokens from 9M aligned multimodal samples (TerraMesh)."],
    ["Performance:", " Beat 12 EO FMs on PANGAEA by ≥8% on land cover & change detection."],
    ["Weaknesses:", " EU/global emphasis; no in-situ hydrology/gravimetry; no physics; generates missing modalities, not future timesteps."],
    ["Opportunity:", " Import TiM for monsoon cloud gap-fill (SAR + INSAT → optical-equivalent); then extend 'thinking in modalities' → 'thinking in time' (forecasting)."]]],
  ["1.10 AlphaEarth Foundations (Google DeepMind, 2025)", [
    ["Architecture:", " Space-Time Precision (STP) encoder; multimodal autoencoder with implicit decoders; teacher-student + contrastive losses."],
    ["Output:", " 64-d embeddings at 10 m, annual, served in Google Earth Engine; >1.4 trillion footprints/year."],
    ["Weaknesses:", " Not open weights (a product, not a controllable model); annual cadence; 10 m; sovereignty non-starter (data leaves the country)."],
    ["Opportunity:", " Proves the embeddings-as-a-service model at planetary scale — and why India needs its own. Ship a comparable Bharat-EO Embedding Layer at finer cadence/resolution, fully sovereign."]]],
];
models.forEach(([title, rows]) => {
  body.push(h3(title));
  rows.forEach(([l,r]) => body.push(bl(l, r)));
});
body.push(h3("1.11 Supporting / adjacent models"));
body.push(table([2000,4400,3680],
  ["Model","What it adds","Lesson for Bharat-EarthFM"],
  [
    ["Prithvi-WxC (~2.3B)","Weather/climate FM on MERRA-2; forecasting + downscaling","Blueprint for a monsoon companion model"],
    ["SatlasPretrain (AI2)","Swin backbones; 302M labels; supervised + SSL","Value of massive labeled supervision alongside SSL"],
    ["CROMA","Contrastive + reconstruction on Sentinel-1+2","Proven SAR-optical cross-modal recipe"],
    ["Presto (~0.4M)","Lightweight pixel time-series transformer","Edge / low-compute pixel inference for villages, mobile"],
    ["SpectralGPT","3D spectral MAE for multispectral","Spectral-cube modeling for future Indian hyperspectral"],
    ["DeCUR / msGFM","Decoupled multisensor representations","Avoiding modality collapse in fusion"],
  ]));
body.push(h3("1.12 TerraTorch — the ecosystem, not a model"));
body.push(bl("What it is:", " IBM's open fine-tuning/benchmarking framework (PyTorch Lightning + TorchGeo); official toolkit for Prithvi and TerraMind; provides backbones, task heads, datamodules, and GEO-Bench/PANGAEA evaluation."));
body.push(bl("Weakness:", " Built around IBM model families; no India task cards, no hydrology/groundwater heads, no monsoon-aware datamodules."));
body.push(bl("Opportunity:", " Do not rebuild the harness — extend TerraTorch. Ship Bharat-EarthFM as TerraTorch-compatible backbones plus a 'bharat-eo' plugin with Indian datamodules, hydrology heads, and the Bharat-EO-Bench suite."));

body.push(spacer());
body.push(h2("1.13 Master Comparison Table"));
body.push(table([1080,1080,1280,1300,1260,860,860,2360],
  ["Model","Org / Year","Backbone & Params","Modalities","Objective","Res.","Forecast?","Key weakness for India"],
  [
    ["Prithvi-1.0","IBM/NASA '23","ViT, 100M","Optical (HLS)","MAE (3D patch)","30 m","No","US-only, optical-only, coarse"],
    ["Prithvi-EO-2.0","IBM/NASA '24","ViT-L/H, 300/600M","Optical (HLS)","MAE + geo/date","30 m","No","Optical-only, 30 m, no SAR/forecast"],
    ["Clay v1.5","Clay '25","ViT, 632M","S2/L8/S1/MODIS","MAE + DINOv2","mixed","No","Embeddings only, no physics/India"],
    ["SatMAE/++","Stanford '22/'23","ViT-L","Optical (fMoW)","MAE spectral/multiscale","mixed","No","Classification bias, no SAR"],
    ["Scale-MAE","Berkeley '23","ViT-L","RGB","MAE + GSD pyramid","multi","No","RGB-only, no time/SAR"],
    ["DOFA","'24","ViT, ~330M","RGB/MS/HS/SAR","Wavelength-hypernet MAE","mixed","No","No time/physics/India"],
    ["AnySat","'24/'25","ViT (JEPA)","Multimodal multi-res","I-JEPA latent","multi","No","Research-grade, no forecast"],
    ["Galileo","Harvest/AI2 '25","ViT (multi-size)","Optical+SAR+DEM+wx","Global+local contrastive","multi","No","Small corpus, no in-situ/forecast"],
    ["TerraMind","IBM/ESA '25","Dual-scale enc-dec","Any-to-any multimodal","Generative + TiM","mixed","No (modalities only)","EU-centric, no hydrology/physics"],
    ["AlphaEarth","Google DM '25","STP autoencoder","Optical+radar+LiDAR+DEM","Multimodal AE","10 m","No","Closed, annual, not sovereign"],
    ["Bharat-EarthFM","Proposed","MoE 0.3→3B + WxC","Optical+SAR(C/L/S)+INSAT+InSAR+GRACE+wx+in-situ","Masked+JEPA+contrastive+TiM+forecast+physics","≤5 m pyramid","Yes (modalities + time)","— (sovereign)"],
  ], 14));
body.push(new Paragraph({ children:[ new PageBreak() ] }));

// ============================================================
// PART II
// ============================================================
body.push(h1("Part II — Research Gaps"));
body.push(h2("2.1 Global gaps (true for all incumbent EO FMs)"));
[
  ["Forecasting void.", " Every EO FM is descriptive/representational or imputes missing modalities; none forecast future surface states (flood extent, yield, reservoir trajectory, drought onset)."],
  ["Sub-surface blindness.", " No FM ingests InSAR deformation, gravimetry (GRACE-FO), or in-situ piezometry — so groundwater is untouched."],
  ["No physics.", " Pure data-driven models violate water/energy balance, hallucinate impossible states, and generalize poorly out-of-distribution."],
  ["Resolution ceiling.", " 10–30 m dominates; fragmented Global-South landscapes are sub-pixel. FM-scale super-resolution to ≤5 m is unsolved."],
  ["Shallow temporal modeling.", " 'Multi-temporal' means a handful of stamps; long, irregular, gap-riddled series and explicit seasonality are poorly handled."],
  ["Cloud regime as afterthought.", " Models assume clear scenes; generative gap-filling is not tuned to tropical monsoon atmospheres and ignores geostationary cadence."],
  ["Benchmark bias.", " GEO-Bench, PANGAEA, EuroSAT, So2Sat are Western/temperate-weighted; no credible South-Asian/smallholder/hydrology benchmark exists."],
  ["Weak uncertainty & calibration.", " Decision-grade EO needs calibrated predictive uncertainty; FMs ship point features."],
  ["No causal/counterfactual reasoning.", " Dam impact, policy intervention, 'what if monsoon fails' are out of reach of representation-only models."],
  ["Sovereignty & edge.", " Best models are closed products or compute-prohibitive; none center data residency, on-prem inference, or edge deployment."],
].forEach(([l,r])=> body.push(nl("n2", l, r)));

body.push(spacer());
body.push(h2("2.2 India-specific gaps"));
body.push(table([520,2400,3560,3600],
  ["#","Gap","Why incumbents fail","What is missing"],
  [
    ["I1","Monsoon-phase-aware representation","Calendar-date conditioning ignores active/break spells, onset/withdrawal, kharif/rabi/zaid","Temporal embedding conditioned on IMD rainfall phase + agro-calendar"],
    ["I2","Cloud-resilient kharif monitoring","Optical FMs lose June–Sept, the most important season","SAR-first + geostationary INSAT + monsoon-tuned generative gap-fill"],
    ["I3","Smallholder field-scale embeddings","10–30 m mixed pixels on ~1 ha fields; intercropping","≤5 m pyramid + boundary-aware tokens"],
    ["I4","Groundwater storage-change","No FM fuses InSAR + GRACE + CGWB wells","Sub-surface module (Punjab/Haryana depletion)"],
    ["I5","Reservoir & irrigation forecasting","No FM ingests CWC storage / canal command areas","Hydrological head linking reservoir + command-area greening"],
    ["I6","Indian cropping-system crop type/yield","Trained on US/EU crops; misses rice/wheat/pulses/oilseeds/cotton, multi-cropping","India crop ontology + FASAL/yield labels"],
    ["I7","Flood + cyclone (deltaic/orographic)","Generic flood benchmarks; no Indian deltaic regimes","Region-specific flood/inundation forecasting head"],
    ["I8","Himalayan cryosphere & GLOF","No glacier-lake-outburst, snowmelt, steep-terrain priors","Cryosphere module + DEM/InSAR"],
    ["I9","Urban informality & heat","Western urban morphology; misses informal settlements, UHI","Indian urban morphology + LST heat head"],
    ["I10","Administrative & multilingual grounding","Embeddings not linked to district/tehsil/village or Indic-language queryable","Admin-graph alignment + Indic GeoLLM connector"],
    ["I11","No Indian EO benchmark","Cannot prove 'outperforms' on India","Bharat-EO-Bench"],
  ]));

body.push(spacer());
body.push(h2("2.3 Opportunity ranking (1–5; composite = sum)"));
body.push(table([2400,820,820,940,1040,940,3120],
  ["Opportunity","Novelty","Sci. impact","Commercial","Patent.","Comp.","Notes"],
  [
    ["Groundwater storage-change (InSAR+GRACE+CGWB)","5","5","5","5","20","Untouched globally; strong method patents"],
    ["Monsoon-phase-aware temporal encoding","5","4","4","5","18","New, broadly reusable, patentable"],
    ["Cloud-resilient monsoon gap-fill (SAR+INSAT)","4","5","5","4","18","Unlocks kharif; insurance/agri demand"],
    ["Surface-state forecasting head","5","5","5","3","18","Descriptive→anticipatory; huge market"],
    ["Reservoir + irrigation forecasting","4","4","5","4","17","B2G + agri-fintech; CWC integration moat"],
    ["Field-scale (≤5 m) pyramid + super-resolution","4","4","5","4","17","Smallholder unlock; credit/insurance"],
    ["Physics-informed adapters (water/energy)","5","5","3","4","17","Scientific credibility; OOD robustness"],
    ["Crop type/yield (Indian systems)","3","4","5","3","15","Largest near-term revenue; data moat"],
    ["Indic geospatial NL interface","4","3","5","3","15","Mass-market delivery to officials/farmers"],
    ["Coastal/cyclone (Bay of Bengal)","3","4","4","3","14","Disaster B2G; reinsurance"],
    ["Himalayan cryosphere / GLOF","4","4","3","3","14","High scientific value; narrower market"],
    ["Bharat-EO-Bench (sovereign benchmark)","3","5","3","2","13","Standards-setting; ecosystem control"],
  ]));
body.push(p({t:"Strategic read: ",bold:true},"the top quartile is hydrology-and-time (groundwater, monsoon encoding, gap-fill, forecasting, reservoirs) — precisely where no incumbent operates and where India has unique data assets (CGWB, CWC, IMD, NISAR, INSAT). That is the defensible core; crop/yield is the revenue engine layered on top."));
body.push(new Paragraph({ children:[ new PageBreak() ] }));

// ============================================================
// PART III
// ============================================================
body.push(h1("Part III — Bharat-EarthFM Architecture"));
body.push(h2("3.1 System overview (five planes)"));
[
 "+-----------------------------------------------------------------------+",
 "| DELIVERY    Indic GeoLLM connector . APIs . Embedding-as-a-Service     |",
 "+-----------------------------------------------------------------------+",
 "| REASONING   Forecasting heads . Physics adapters . Task heads          |",
 "|             (crop/flood/drought/GW/reservoir/urban/coastal)            |",
 "+-----------------------------------------------------------------------+",
 "| FOUNDATION  Core encoder (MoE ViT) + Generative decoder (TiM) +        |",
 "|             Monsoon-WxC companion                                      |",
 "+-----------------------------------------------------------------------+",
 "| SENSOR      Wavelength/GSD-conditioned dynamic tokenizer (any-sensor)  |",
 "+-----------------------------------------------------------------------+",
 "| DATA        Bharat-EO-Corpus (analysis-ready datacube, sovereign)      |",
 "+-----------------------------------------------------------------------+",
].forEach(l=>body.push(mono(l)));
body.push(spacer());

body.push(h2("3.2 Sensor Plane — universal Indian-sensor tokenizer"));
body.push(p("A single dynamic patch-embedding hypernetwork (fusing DOFA's wavelength-hypernetwork, Clay's dynamic embedding, and Scale-MAE's GSD encoding) maps any sensor to a common token space. Per-channel conditioning = {central wavelength, bandwidth, polarization, GSD, sensor-id, acquisition geometry}."));
body.push(bl("Optical/MS:"," Sentinel-2, Landsat/HLS, Resourcesat AWiFS (56 m)/LISS-III (24 m)/LISS-IV (5.8 m), EOS-06/Oceansat-3."));
body.push(bl("SAR:"," Sentinel-1 (C), EOS-04/RISAT-1A (C, all-weather), NISAR (L+S, 12-day, InSAR) — the cloud-penetrating backbone."));
body.push(bl("Geostationary:"," INSAT-3D/3DR/3DS — high-cadence imaging the incumbents lack."));
body.push(bl("Geophysical:"," InSAR deformation (subsidence), GRACE-FO terrestrial water storage anomaly, CartoDEM/SRTM elevation."));
body.push(bl("Atmospheric/in-situ:"," IMD gridded rainfall & temperature, IMD AWS; CWC reservoir storage & discharge; CGWB groundwater levels; soil moisture."));
body.push(p({t:"Patentable (P1): ",bold:true},"the Indian-sensor conditioning schema admitting NISAR L/S-band and INSAT geostationary into one token space."));

body.push(h2("3.3 Foundation Plane — core encoder"));
body.push(bl("Backbone:"," ViT with Mixture-of-Experts FFN (modality- and region-routed). Sizes: Bharat-EarthFM-S (300M) → -L (1B) → -XL (3B)."));
body.push(bl("Tokenization:"," spatial patches × timesteps × logical channel groups over a multi-resolution pyramid; a super-resolution branch targets ≤5 m for field boundaries."));
body.push(bl("Temporal core (MPATE):"," temporal embedding conditioned on (a) monsoon phase (pre-onset/onset/active/break/withdrawal), (b) agro-calendar (kharif/rabi/zaid day-index), (c) absolute date; long, irregular, gap-tolerant sequences via continuous-time attention. Patentable (P2)."));

body.push(h2("3.4 Generative cross-modal decoder — 'Think in Modalities, then in Time'"));
body.push(bl("TiM (modalities):"," import TerraMind's any-to-any decoder, re-tuned for India: synthesize cloud-free optical-equivalent from EOS-04/NISAR SAR + INSAT, conditioned on monsoon phase. Solves the kharif blackout."));
body.push(bl("ThiT (forecasting):"," a latent diffusion/autoregressive forecaster rolls embeddings forward to predict future surface states (flood extent, SPI/SPEI, reservoir level, biomass/yield, groundwater anomaly) — the descriptive→anticipatory leap no EO FM has made. Patentable (P3)."));

body.push(h2("3.5 Reasoning Plane — physics-informed heads"));
body.push(bl("Groundwater module:"," fuses InSAR (NISAR/Sentinel-1) + GRACE-FO TWS + CGWB piezometry under a water-balance loss; outputs storage-change & depletion-risk. Patentable (P4)."));
body.push(bl("Reservoir/irrigation head:"," CWC storage + water-surface area + command-area greening under reservoir mass-balance."));
body.push(bl("Hydrology heads:"," energy-balance-constrained ET; rainfall-runoff coupling with the Monsoon-WxC companion (Prithvi-WxC-style)."));
body.push(bl("Physics-informed adapters:"," PINN-style soft constraints embed water/energy balance into fine-tuning for physically admissible, OOD-robust outputs."));
body.push(bl("Task heads:"," crop type/yield, LULC, urban + UHI/LST, coastal, disaster — all TerraTorch-compatible, each emitting calibrated uncertainty."));

body.push(h2("3.6 Delivery Plane"));
body.push(bl("Bharat-EO Embedding Layer:"," an AlphaEarth-style downloadable embedding product at seasonal/monsoon-phase cadence, ≤10 m to field scale, served from sovereign infrastructure."));
body.push(bl("APIs:"," tile/AOI query, task inference, forecasting endpoints."));
body.push(bl("Indic GeoLLM connector:"," natural-language geospatial Q&A grounded on the administrative graph (state→district→tehsil→village) in multiple Indian languages."));

body.push(h2("3.7 Evaluation — Bharat-EO-Bench"));
body.push(p("A sovereign benchmark spanning all twelve domains: smallholder crop-type under cloud, district yield forecasting, Indian flood mapping (Brahmaputra/Ganga/Bay-of-Bengal deltas), SPI/SPEI drought, CGWB groundwater-anomaly forecasting, CWC reservoir-level forecasting, urban-growth change, coastal-erosion, GLOF risk. Baselines: Prithvi-EO-2.0, Clay, TerraMind, Galileo, AlphaEarth embeddings."));

body.push(h2("3.8 Training strategy (staged curriculum)"));
body.push(table([520,3000,3760,2800],
  ["Stage","Objective","Data","Output"],
  [
    ["0","Build Bharat-EO-Corpus (ARD datacube)","All sensor sources","Analysis-ready cubes"],
    ["1","SSL: masked + JEPA + cross-modal contrastive (warm-start from Prithvi-EO-2.0)","Full corpus","Core encoder"],
    ["2","Generative: TiM modality completion (monsoon gap-fill)","Aligned multimodal","Cross-modal decoder"],
    ["3","Physics + supervised adapters","Labeled Indian sets (FASAL, CGWB, CWC, IMD, Bhuvan)","Task heads"],
    ["4","ThiT temporal forecasting fine-tune","Time-series labels","Forecasting heads"],
    ["5","Alignment: Indic GeoLLM instruction tuning","NL + geo pairs","Delivery layer"],
  ]));

body.push(h2("3.9 Sovereign compute & MLOps"));
body.push(bl("Compute:"," C-DAC PARAM / AIRAWAT (NSM) and the IndiaAI Mission GPU pool."));
body.push(bl("Stack:"," PyTorch + FSDP/DeepSpeed + MoE; TerraTorch-compatible checkpoints; DVC/lakeFS; Kubernetes + Ray; ONNX/TensorRT + quantized Presto-style edge variants."));
body.push(bl("Governance:"," full data residency in India; aligned with National Geospatial Policy 2022 and IndiaAI Mission; ISRO/NRSC + MeitY governance; open weights for Indian academia/government; tiered commercial licensing."));

body.push(h2("3.10 Patent portfolio (priority filings — expanded in Part IX)"));
["P1 Indian multi-sensor conditioning tokenizer","P2 Monsoon-Phase-Aware Temporal Encoding","P3 Think-in-Modalities-then-Time generative forecaster","P4 Multimodal groundwater storage-change estimator","P5 Reservoir/irrigation forecasting under mass balance","P6 Field-scale super-resolution boundary-aware tokenization"].forEach(x=>body.push(bullet(x)));

body.push(h2("3.11 Commercialization & 3.12 Roadmap"));
body.push(bl("B2G anchor:"," ISRO/NRSC, MoA (PMFBY), CWC, CGWB, NDMA, state RS centres."));
body.push(bl("B2B:"," agri-fintech, crop-insurance, banks, infrastructure/ESG, reinsurance, commodity trading."));
body.push(bl("Embeddings-as-a-Service:"," Bharat-EO Embedding Layer subscription (sovereign)."));
body.push(table([1500,8580],
  ["Phase","Milestone"],
  [
    ["0–6 mo","Bharat-EO-Corpus v0 + Bharat-EO-Bench v0; warm-start 300M from Prithvi-EO-2.0; beat baselines on 3 Indian tasks"],
    ["6–12 mo","Add SAR/INSAT, TiM gap-fill, MPATE; release -S (300M) open weights + TerraTorch plugin"],
    ["12–24 mo","Groundwater + reservoir + forecasting heads; scale to -L (1B); first B2G pilots"],
    ["24–36 mo","-XL (3B) + Monsoon-WxC; Embedding Layer product; Indic GeoLLM; national deployment; patents filed"],
  ]));
body.push(new Paragraph({ children:[ new PageBreak() ] }));

// ============================================================
// PART IV
// ============================================================
body.push(h1("Part IV — Complete Dataset Inventory"));
body.push(p("Per dataset: Source/Access, Download method, native Resolution, Update frequency, Licensing, and India curated volume (the analysis-ready datacube footprint over India + a coastal/transboundary buffer of ~4.0 M km², not the raw global archive)."));
const dsCols = [1500,1560,1480,1180,1080,1480,1800];
const dsHdr = ["Dataset","Source / Access","Download","Resolution","Update","Licensing","India vol."];

body.push(h3("IV.1 Optical — multispectral & high-resolution"));
body.push(table(dsCols, dsHdr, [
  ["Sentinel-2 L1C/L2A","ESA Copernicus","Copernicus DS / AWS / MPC (STAC+COG)","10/20/60 m, 13 bands","5-day","Copernicus open","~100 TB"],
  ["Landsat 8/9 + HLS","USGS / NASA","USGS M2M, AWS, LP DAAC","30 m (HLS)","8-day","US public domain","~40 TB"],
  ["Resourcesat AWiFS/LISS-III/IV","ISRO NRSC","Bhuvan / NOEDA","56 / 24 / 5.8 m","5–24 day","Free (NOEDA); NGP-2022","~50 TB"],
  ["Cartosat-1/2/3","ISRO NRSC","NOEDA / commercial","2.5 m → 0.25 m","tasking","Restricted (high-res)","~10 TB"],
  ["EOS-06 / Oceansat-3","ISRO NRSC","Bhuvan / NOEDA","360 m–1 km","2-day","Free (NOEDA)","~5 TB"],
  ["MODIS Terra/Aqua","NASA LP DAAC","Earthdata, GEE","250 m–1 km","daily","US public domain","~10 TB"],
  ["Planet NICFI basemaps","Planet / NICFI","NICFI program API","~4.7 m","monthly","NICFI license","~8 TB"],
  ["Optical subtotal","","","","","","~223 TB"],
]));

body.push(h3("IV.2 SAR — cloud-penetrating backbone"));
body.push(table(dsCols, dsHdr, [
  ["Sentinel-1 GRD","ESA Copernicus","Copernicus DS / ASF / MPC","10–20 m, C","6–12 day","Copernicus open","~25 TB"],
  ["Sentinel-1 SLC (InSAR)","ESA Copernicus","ASF DAAC (HyP3)","5×20 m SLC","6–12 day","Copernicus open","~120 TB"],
  ["NISAR L-/S-band","NASA-ISRO","ASF DAAC + ISRO","3–10 m, 12-day","12-day","Free & open","~200 TB"],
  ["EOS-04 / RISAT","ISRO NRSC","NOEDA / Bhuvan","C, ~1–50 m","all-weather","Free (NOEDA)","~20 TB"],
  ["ALOS-2 PALSAR","JAXA","G-Portal","25 m, L","annual mosaics","Free (JAXA)","~5 TB"],
  ["SAR subtotal","","","","","","~370 TB"],
]));

body.push(h3("IV.3 Geostationary & precipitation"));
body.push(table(dsCols, dsHdr, [
  ["INSAT-3D/3DR/3DS","ISRO / MOSDAC","MOSDAC order/FTP","1–4 km, 6+ ch","15–30 min","Free (MOSDAC)","~80 TB"],
  ["GPM IMERG","NASA","GES DISC, GEE","0.1°","30 min","US public domain","~15 TB"],
  ["Geostationary subtotal","","","","","","~95 TB"],
]));

body.push(h3("IV.4 Hydrology / geophysical / terrain"));
body.push(table(dsCols, dsHdr, [
  ["GRACE / GRACE-FO mascon","NASA JPL PO.DAAC","PO.DAAC netCDF","0.5° (~55 km)","monthly","US public domain","~0.005 TB"],
  ["InSAR deformation","derived (S1/NISAR)","ASF HyP3 / ISCE2","~30–90 m","12-day","derived (open)","~10 TB"],
  ["GLO-30 DEM / CartoDEM / SRTM","ESA/ISRO/NASA","AWS / Bhuvan / EE","30 m (10 m Carto)","static","ESA/NOEDA/public","~0.15 TB"],
  ["Snow & glacier","NASA / GLIMS","NSIDC, GLIMS","250–500 m","daily/static","open","~5 TB"],
  ["Terrain subtotal","","","","","","~20 TB"],
]));

body.push(h3("IV.5 Weather / climate reanalysis & in-situ meteorology"));
body.push(table(dsCols, dsHdr, [
  ["ERA5 reanalysis","ECMWF / C3S","CDS API, ARCO-ERA5 Zarr","0.25°, hourly","hourly","C3S licence","~15 TB"],
  ["MERRA-2","NASA GMAO","GES DISC","0.5×0.625°, hourly","hourly","US open","~5 TB"],
  ["IMD gridded rainfall + temp","IMD Pune","imdpune.gov.in","0.25°/0.1°, daily","daily","IMD terms","~0.05 TB"],
  ["Climate subtotal","","","","","","~20 TB"],
]));

body.push(h3("IV.6 In-situ networks (small, tabular/vector — the supervision moat)"));
body.push(table(dsCols, dsHdr, [
  ["CGWB groundwater (~26k wells)","CGWB / India-WRIS","India-WRIS + API","point","monthly–quarterly","NDSAP open","~0.05 TB"],
  ["CWC reservoir + discharge","CWC / India-WRIS","India-WRIS bulletins","point","weekly","NDSAP open","~0.02 TB"],
  ["IMD AWS/ARG stations","IMD","IMD / MOSDAC","point","sub-daily","IMD terms","~0.05 TB"],
  ["In-situ subtotal","","","","","","~0.15 TB"],
]));

body.push(h3("IV.7 Labels & ancillary"));
body.push(table(dsCols, dsHdr, [
  ["WorldCover / Dynamic World / ESRI LC","ESA/Google/Esri","GEE, AWS, Zenodo","10 m","annual / NRT","CC-BY-4.0","~3 TB"],
  ["Bhuvan LULC","ISRO NRSC","Bhuvan","1:50k/250k","multi-year","NOEDA","~1 TB"],
  ["JRC Global Surface Water","EC JRC","GEE, JRC portal","30 m","annual","open (EC)","~1 TB"],
  ["GEDI L2/L4 biomass","NASA","LP DAAC, ORNL","25 m footprints","mission","US public domain","~2 TB"],
  ["WorldCereal / crop labels / FASAL","ESA / MoA&FW","Zenodo / institutional","10 m","seasonal","CC-BY / restricted","~1 TB"],
  ["OSM / SHRUG / admin & census","OSM / DevDataLab","Geofabrik, SHRUG","vector","rolling","ODbL / open","~0.5 TB"],
  ["SoilGrids / NBSS&LUP","ISRIC / ICAR","ISRIC WCS","250 m","static","CC-BY-4.0 / restricted","~2 TB"],
  ["Labels subtotal","","","","","","~10.5 TB"],
]));

body.push(spacer());
body.push(h2("IV.8 Total Storage Requirement"));
body.push(h3("Step 1 — irreducible curated ARD corpus"));
body.push(table([6080,4000],
  ["Tier","Volume"],
  [["Optical","~223 TB"],["SAR","~370 TB"],["Geostationary / precip","~95 TB"],["Terrain / geophysical","~20 TB"],
   ["Climate / reanalysis","~20 TB"],["In-situ","~0.15 TB"],["Labels / ancillary","~10.5 TB"],["Curated ARD total","≈ 0.75 PB"]]));
body.push(h3("Step 2 — full datacenter provisioning"));
body.push(table([3300,4180,2600],
  ["Component","Sizing logic","Volume"],
  [
    ["Curated ARD (1 copy)","from above","0.75 PB"],
    ["Raw / staging mirror","S1 SLC + NISAR raw + S2 raw (partial)","~1.5 PB"],
    ["Re-sharded training shards","≈ 1 copy re-chunked (WebDataset)","~0.75 PB"],
    ["Hot working set ×3 replication","active subset ~400 TB × 3","~1.2 PB"],
    ["Checkpoints + optimizer + experiments","MoE total ~4B × 12 B/param × ~100 ckpts","~30–50 TB"],
    ["Embedding product output","India @10 m, 64-d int8, 4 seasons/yr","~10–70 TB/yr"],
    ["Logs / features / eval","—","~20 TB"],
    ["Recommended provisioning","hot ~1.5 PB + warm ~2 PB + cold ~5 PB","≈ 8–9 PB"],
  ]));
body.push(p({t:"Sovereign-infrastructure advantage: ",bold:true},"ISRO ground-station direct downlink for Resourcesat/Cartosat/EOS/INSAT/NISAR-India makes the raw-mirror tier a local feed, not an internet pull. Only Copernicus/NASA/ECMWF data needs sustained external ingest."));
body.push(new Paragraph({ children:[ new PageBreak() ] }));

// ============================================================
// PART V
// ============================================================
body.push(h1("Part V — Foundation Model from First Principles"));
body.push(h2("5.1 Derive the architecture from the data's properties"));
body.push(table([5040,5040],
  ["Property of Indian EO data","Forced design consequence"],
  [
    ["Many sensors, different bands/polarizations/GSD","Wavelength-conditioned dynamic tokenizer (any sensor)"],
    ["Pervasive missing data (monsoon cloud, revisit gaps)","Masked + generative training; never assume complete input"],
    ["Long, irregular, seasonal time series","Continuous-time temporal attention + monsoon-phase conditioning (MPATE)"],
    ["Multi-resolution (56 m ↔ 0.25 m)","Pyramid tokenization + GSD-scaled positions + super-resolution"],
    ["Quadratic attention over space×time×modality unaffordable","Factorized spatio-temporal attention + windowed/global mix"],
    ["Broad capacity but cheap national-scale inference","Mixture-of-Experts FFN: high total capacity, low active FLOPs"],
    ["Decision-grade outputs must obey hydrology","Physics-informed adapters (water/energy balance)"],
    ["Corpus (~0.75 PB) is small vs LLMs","Compute-optimal size ≈ 1–1.5 B active params"],
  ]));

body.push(h2("5.2 Recommended final architecture"));
body.push(p("A multimodal, multi-temporal, multi-resolution Transformer encoder with MoE FFNs, a wavelength-conditioned dynamic tokenizer, three decoders (reconstruction / cross-modal-generative / temporal-forecast), and physics-informed task adapters. The -L variant is the recommended flagship."));
body.push(table([1700,1000,900,900,1380,1300,2900],
  ["Variant","d_model","Layers","Heads","Active params","Total params","Use"],
  [
    ["Bharat-EarthFM-S","1024","24","16","300 M","300 M","Edge/village; warm-start from Prithvi-EO-2.0 ViT-L"],
    ["Bharat-EarthFM-L (flagship)","1408","40","16","~1.3 B","~3.8 B","Flagship — compute-optimal for the corpus"],
    ["Bharat-EarthFM-XL","1792","48","16","~3 B","~12 B","Reserve — when corpus + compute scale up"],
  ]));

body.push(h2("5.3 Layer-by-layer design (Bharat-EarthFM-L flagship)"));
[
 "INPUT: variable set of {sensor patches xi, per-channel meta ci=(lambda, bandwidth,",
 "       polarization, GSD, sensor-id), geo=(lat,lon), time=(date, monsoon-phase, agro-day)}",
 "",
 "[0] Dynamic Patch Tokenizer (hypernetwork)",
 "    - Per channel: conv-stem weights W = H(ci)  (hypernet from wavelength/polarization/GSD)",
 "    - Patch 16x16 at reference GSD; multi-scale crops -> Laplacian-pyramid tokens",
 "    - Output tokens in R^(N x 1408), N = (H/16)(W/16) x T x G  (G = channel groups)",
 "",
 "[1] Positional / conditioning encoding (added to tokens)",
 "    - Space: 2D RoPE scaled by GSD",
 "    - Time: MPATE = f(date) (+) f(monsoon-phase) (+) f(agro-day)   <- novel (P2)",
 "    - Spectral/sensor: learned embedding from ci ; Geo: Fourier(lat,lon)",
 "",
 "[2] Modality-fusion prelude (layers 1-4): full attention within (space x group),",
 "    cross-modal token mixing so SAR/optical/weather tokens see each other early",
 "",
 "[3] Factorized spatio-temporal MoE backbone (layers 5-36, 32 blocks). Each block:",
 "    LN -> Spatial windowed attn -> +res",
 "    LN -> Temporal attn (continuous-time, gap-aware) -> +res",
 "    LN -> FFN: even = dense GeGLU x4; odd = MoE (8 experts, top-2, modality+region gating) -> +res",
 "    Every 8th block: GLOBAL attention (basin-scale hydrology, monsoon fronts)",
 "    QK-norm, RMSNorm, FlashAttention-3, rotary positions",
 "",
 "[4] Encoder output: Z in R^(N x 1408) + pooled global/local CLS tokens",
 "",
 "DECODERS (pretraining/heads):",
 "[D1] Reconstruction decoder (8 layers, d=512) -> MAE pixel/feature reconstruction",
 "[D2] Cross-modal generative decoder (TiM) -> synthesize missing modality   <- (P3)",
 "[D3] Temporal forecast decoder (ThiT) -> latent diffusion/AR roll-forward    <- (P3)",
 "",
 "TASK / REASONING ADAPTERS (fine-tune):",
 "[H] Physics-informed heads: groundwater (InSAR+GRACE+CGWB, water-balance loss),",
 "    reservoir (mass-balance), flood/drought (energy-balance ET), crop, LULC, urban, coastal",
 "    + calibrated uncertainty (deep ensemble / evidential) on every output",
].forEach(l=>body.push(mono(l)));

body.push(spacer());
body.push(h2("5.4 Why this size is optimal"));
body.push(bl("Data-constrained regime:"," the corpus yields ~few-×100 B to ~1.5 T trainable tokens; Chinchilla-style optimality (~20 tokens/param) puts the sweet spot at ~1–1.5 B active params — exactly -L. Going to 10B+ would over-fit a data-limited regime."));
body.push(bl("MoE decouples capacity from cost:"," 3.8 B total / 1.3 B active gives breadth across sensors/regions/seasons while keeping inference cheap enough for national-scale sovereign on-prem runs."));
body.push(bl("Warm-start economics:"," -S (d1024/L24) is intentionally isomorphic to Prithvi-EO-2.0 ViT-L so its optical branch initializes from open weights, cutting stage-1 compute."));
body.push(new Paragraph({ children:[ new PageBreak() ] }));

// ============================================================
// PART VI
// ============================================================
body.push(h1("Part VI — Pretraining Objectives"));
body.push(p("Six objectives, staged. Each: mechanism, why it is chosen, and the India-specific reason."));
body.push(table([520,1700,2600,2560,2700],
  ["#","Objective","Mechanism","Why chosen","India-specific reason"],
  [
    ["1","Masked Image / Modality Modeling","High-ratio random + structured (whole-modality/timestep) masking; reconstruct pixels & deep features","Forces learning of redundancy across space/bands/sensors","Structured modality-masking trains the model to operate when optical is absent (kharif cloud)"],
    ["2","Temporal prediction","Mask future/intermediate timesteps; predict masked-time tokens (gap-aware)","Teaches seasonality & dynamics, not static appearance","Encodes monsoon active/break rhythm + crop phenology via MPATE"],
    ["3","Cross-modal alignment","(a) generative TiM modality synthesis; (b) contrastive alignment of co-located SAR/optical/weather","Makes one modality substitutable; enables fusion & missing-data robustness","Fills monsoon cloud gaps (SAR + INSAT → optical-equivalent)"],
    ["4","Contrastive (global + local)","InfoNCE on global tokens + DINO-style local self-distillation","Linearly-separable, transferable embeddings for few-shot + product","Powers the Bharat-EO Embedding Layer + few-shot on data-scarce Indian tasks"],
    ["5","Forecasting","ThiT multi-horizon latent diffusion / AR roll-forward vs future EO + in-situ","Turns a descriptive FM into an anticipatory one","Directly produces flood/drought/yield/reservoir/groundwater forecasts"],
    ["+","Physics-consistency (auxiliary)","Soft penalties: water balance, energy balance, reservoir mass-balance","Keeps generated/forecast states physically admissible; improves OOD","Essential for decision-grade hydrology (insurance, disaster, irrigation)"],
  ], 15));
body.push(spacer());
body.push(h2("Loss schedule (staged curriculum)"));
[
 "Stage 1 (SSL):        L = 1.0*MIM + 0.5*Temporal + 0.3*Contrastive(global+local)",
 "Stage 2 (generative): L += 1.0*TiM(cross-modal) + 0.3*Contrastive-align",
 "Stage 3 (physics+SL): L += 0.7*TaskHeads + 0.3*PhysicsConsistency",
 "Stage 4 (forecast):   L += 1.0*ThiT(multi-horizon) + 0.3*PhysicsConsistency",
 "Stage 5 (align):      instruction-tune Indic GeoLLM connector (frozen encoder)",
 "+ MoE load-balancing aux loss (0.01) throughout",
].forEach(l=>body.push(mono(l)));
body.push(spacer());
body.push(p({t:"Why staged, not joint-from-scratch: ",bold:true},"generative and forecasting objectives are unstable on a cold encoder; the representation must stabilize first (Stages 1–2) before physics/forecasting heads (Stages 3–4). This is standard for multi-objective FMs and de-risks divergence."));
body.push(new Paragraph({ children:[ new PageBreak() ] }));

// ============================================================
// PART VII
// ============================================================
body.push(h1("Part VII — Infrastructure Estimates"));
body.push(p("The pre-flight specification — everything sized and justified before a single GPU is allocated."));
body.push(h2("7.1 Compute model (assumptions stated)"));
body.push(p("FLOPs ≈ 6 · N_active · N_tokens. Calibration anchor: Prithvi-EO-2.0-600M = ~58,000 GPU-hours for ~3.6×10^21 FLOPs, implying ~15–25 effective TFLOPS/GPU (low MFU is normal for EO). We assume H100/H200-class at ~25% MFU (~250 effective TFLOPS), with a 3–5× overhead for multi-objective training, MoE routing, restarts, and eval."));
body.push(table([1500,1300,1300,1500,1500,2980],
  ["Variant","Active params","Token budget","Core FLOPs","GPU-hours","Cluster × wall-clock"],
  [
    ["-S (300M)","0.3 B","~0.6 T","1.1×10^21","~15–25k","64 H100 × ~1.5 weeks"],
    ["-L (1.3B) flagship","1.3 B","~3 T","2.3×10^22","~90–150k","256–384 H100 × ~3–5 weeks"],
    ["-XL (3B)","3 B","~5 T","9×10^22","~450–700k","512–1024 H100 × ~5–8 weeks"],
  ]));
body.push(p({t:"Per-stage split for -L (~120k GPU-hrs): ",bold:true},"Stage 1 SSL ~45% · Stage 2 generative ~25% · Stage 3 physics/SL ~10% · Stage 4 forecasting ~15% · Stage 5 align ~5%."));

body.push(h2("7.2 GPU recommendation"));
body.push(bl("Flagship -L:"," 256–384 GPUs (H100/H200 80 GB or indigenous equivalent), 8/node, on AIRAWAT / C-DAC PARAM / IndiaAI Mission pool."));
body.push(bl("Parallelism:"," FSDP/ZeRO-3 + expert parallel (MoE) + activation checkpointing; tensor parallel only at -XL. 1.3 B active fits in 80 GB; long multimodal sequences + experts drive the memory plan."));
body.push(bl("Edge/inference:"," -S quantized (INT8) via ONNX/TensorRT for village/mobile and the embedding-serving fleet."));

body.push(h2("7.3 Storage"));
body.push(p("Curated ARD ~0.75 PB; hot parallel-FS (NVMe) ~1.5 PB; warm object ~2 PB; cold/archive ~5 PB → ~8–9 PB provisioned. Format: COG/Zarr datacubes re-sharded to WebDataset tar shards for sequential GPU streaming; NVMe burst buffer per node."));

body.push(h2("7.4 Networking"));
body.push(table([2200,3000,4880],
  ["Fabric","Requirement","Why"],
  [
    ["Intra-node","NVLink/NVSwitch ~900 GB/s","MoE all-to-all (expert dispatch) is bandwidth-bound"],
    ["Inter-node compute","InfiniBand NDR 400 Gb/s/GPU, rail-optimized non-blocking fat-tree","FSDP collectives + expert-parallel all-to-all at 256–384 GPUs"],
    ["Storage fabric","Parallel FS (Lustre/GPFS/Weka/DAOS) ~200–500 GB/s aggregate read","Sustained ingest ~30–50 GB/s (~8×10^6 tokens/s × ~2–5 KB/token)"],
    ["WAN ingest","Multi-Gbps peering to Copernicus/NASA/ECMWF","Only for non-Indian data; ISRO direct downlink removes Indian-sensor WAN load"],
  ]));

body.push(h2("7.5 Training duration (flagship -L)"));
body.push(bl("Core SSL (Stages 1–2):"," ~2–3 weeks on 256–384 H100."));
body.push(bl("Full pipeline (Stages 1–5, incl. forecasting + physics + align + restarts):"," ~6–10 weeks wall-clock."));
body.push(bl("Other sizes:"," -S ~1.5 weeks (warm-started); -XL ~5–8 weeks at 512–1024 GPUs."));

body.push(h2("7.6 Pre-flight readiness check"));
[
  ["Data:"," NOEDA/MOSDAC/India-WRIS accounts; ISRO direct-downlink feed; Copernicus/NASA/ECMWF keys + cloud-adjacent staging. Ready when ~0.75 PB ARD lands."],
  ["Compute:"," 256–384 H100-class allocation on AIRAWAT/PARAM/IndiaAI confirmed; NDR-400 fabric verified."],
  ["Storage:"," ~1.5 PB hot parallel-FS at ≥200 GB/s benchmarked; WebDataset sharding validated."],
  ["Code:"," TerraTorch fork (bharat-eo plugin) with dynamic tokenizer, MPATE, MoE, TiM/ThiT decoders, physics losses; warm-start weights loaded into -S."],
  ["Eval:"," Bharat-EO-Bench v0 task cards locked so 'outperforms' is measurable from day one."],
].forEach(([l,r])=> body.push(nl("n3", l, r)));
body.push(new Paragraph({ children:[ new PageBreak() ] }));

// ============================================================
// PART VIII
// ============================================================
body.push(h1("Part VIII — Startup Strategy (India)"));
body.push(h2("8.0 Company thesis & positioning"));
body.push(p({t:"One line: ",bold:true},"The sovereign Earth-intelligence layer for India — owned, hosted, and governed in India."));
body.push(bl("Policy unlock:"," National Geospatial Policy 2022 (self-certification, private high-res mapping); IN-SPACe opened the EO sector; IndiaAI Mission funds sovereign AI + GPUs; AgriStack / Digital Agriculture Mission creates farmer-data rails."));
body.push(bl("Market:"," India's geospatial economy ~₹63,000 cr (~$7.5 B) by 2025; geospatial-analytics ~$1.58 B (2025) → $3.55 B (2031); agritech $4 B → $34 B by 2030; PMFBY ₹12,242 cr/yr, Atal Bhujal ₹6,000 cr."));
body.push(bl("Gap:"," AlphaEarth proves the embeddings business model but is closed and offshore — a sovereignty non-starter for Indian government data. That is the structural wedge."));
body.push(bl("Moat (4 layers):"," (1) proprietary in-situ fusion (CGWB/CWC/IMD); (2) the only field-scale, monsoon-aware, sub-surface model; (3) the benchmark standard (Bharat-EO-Bench); (4) the patent stack + trade-secret weights."));

body.push(h2("8.1 Product roadmap"));
body.push(table([900,1100,1300,4180,2600],
  ["Phase","Timeline","Model","Products shipped","Commercial milestone"],
  [
    ["P0 Foundation","M0–9","-S (warm-start)","Bharat-EO-Bench v0; ARD datacube; 3 reference tasks","1–2 paid PoCs (state agri/water)"],
    ["P1 Embedding Layer","M9–18","-S open + early -L","Bharat-EO Embedding Layer + STAC/API; TerraTorch plugin","Embeddings-as-a-Service; design partners"],
    ["P2 Vertical apps","M18–30","-L (1.3B)","CropIntel, AquaIntel, FloodIntel","First ₹-cr ARR B2G; insurance/credit deals"],
    ["P3 Forecasting + platform","M30–42","-L + Monsoon-WxC","Forecasting API; Indic GeoLLM console","Multi-state + central ministry; reinsurers"],
    ["P4 Scale & sovereign stack","M42–60","-XL (3B)","On-prem Sovereign Earth Stack","National deployment; Global South export"],
  ]));

body.push(h2("8.2 Commercial applications"));
body.push(table([1700,3200,2880,2300],
  ["Domain","Product feature","Buyer & budget pool","Quantified value"],
  [
    ["Agriculture","Field-scale crop type/area/yield under cloud","MoA&FW, state agri, agri-fintech, FPOs","Replaces costly crop-cutting; faster yield"],
    ["Crop insurance","Parametric anomaly trigger → auto-payout","PMFBY (₹12,242 cr/yr), insurers","Cuts loss-assessment cost & fraud"],
    ["Agri-credit","Field-level productivity/risk score","Banks/NBFCs (>₹31.5 lakh cr by FY26)","Underwrites thin-file smallholders"],
    ["Groundwater","Storage-change & depletion-risk + forecast","CGWB / Atal Bhujal (₹6,000 cr)","Targets 80-district overexploitation"],
    ["Reservoirs/irrigation","Storage + command-area forecast","CWC, irrigation depts, hydropower","Optimizes releases; drought planning"],
    ["Floods","Inundation mapping + 7-day forecast","NDMA, SDMAs, reinsurers","Pre-positioning; parametric flood cover"],
    ["Droughts","SPI/SPEI + agronomic onset","NITI Aayog, relief depts","Triggers relief faster"],
    ["Coastal","Erosion/cyclone inundation","NCCR/MoES, ports, reinsurers","Coastal zone mgmt; port risk"],
    ["Climate/cryosphere","Heatwave/UHI, GLOF, snowmelt","NDMA, hydropower, hill states","Early warning; hydropower planning"],
    ["Infrastructure","Construction/encroachment change detection","NHAI, railways, utilities, ESG","Asset monitoring at scale"],
    ["Land use / urban","LULC change, urban growth, informal settlements","ULBs, MoHUA, town planning","Planning; property tax; slum mapping"],
    ["Disaster mgmt","Multi-hazard awareness + NL console","NDMA, forces, state EOCs","Unified ops picture in local language"],
  ]));

body.push(h2("8.3 Customers"));
body.push(bl("B2G — Central (anchor):"," ISRO/NRSC & IN-SPACe, MoA&FW (PMFBY, AgriStack), Ministry of Jal Shakti (CGWB, CWC, Atal Bhujal, Jal Jeevan), NDMA, MoES, NITI Aayog. Procurement via GeM, tenders, ISRO/IN-SPACe partnerships."));
body.push(bl("B2G — State (replicable):"," State Remote Sensing Application Centres, state agri/irrigation/disaster/GW boards (~28 states)."));
body.push(bl("B2B verticals:"," general & crop insurers, reinsurers (Swiss Re, Munich Re), banks/NBFCs (NABARD, Samunnati, Origo), agri-fintechs, commodity traders, hydropower, ports, ESG/infra funds."));
body.push(bl("B2D / platform:"," GIS developers, research labs, startups building on the Embedding Layer + API (AWS-for-Earth-intelligence)."));
body.push(bl("Global South export (later):"," SAARC/Africa governments needing a non-US/non-Google sovereign EO stack."));

body.push(h2("8.4 Pricing"));
body.push(table([2300,3380,4400],
  ["Product line","Model","Indicative price"],
  [
    ["Bharat-EO Embedding Layer","Freemium; commercial by area-coverage tier","Starter ₹2–10 L/yr; Pro ₹25 L–₹1 cr/yr; Enterprise ₹1–3 cr/yr"],
    ["Inference / Forecasting API","Usage-based + subscription floor","₹0.5–5 per km²-query; tiers ₹50k–₹10 L/mo"],
    ["CropIntel (insurance/credit)","Per-assessment / per-policy / per-loan","₹3–25 per field; or 1–3% of premium / bps on loan book"],
    ["AquaIntel / FloodIntel (B2G)","Annual enterprise license + setup","₹2–20 cr/yr per ministry/state"],
    ["Sovereign on-prem Earth Stack","License + support + GPU","₹5–50 cr license + 18–22% AMC"],
    ["Insights / data subscriptions","SaaS seats / reports","Reinsurers/traders $2k–$25k/mo"],
  ]));
body.push(p({t:"Revenue mix target (Yr 3): ",bold:true},"~55% B2G recurring + project, ~30% B2B verticals (insurance/credit volume), ~15% platform/API."));

body.push(h2("8.5 Go-to-market strategy"));
[
  ["Wedge — dual undeniable-ROI beachhead:"," crop insurance (PMFBY) + groundwater (Atal Bhujal/CGWB). Both have large committed budgets, painful manual status quo, and a measurable before/after. Win one state in each."],
  ["Credibility via ISRO/IN-SPACe + IndiaAI:"," co-develop on Bhuvan/NOEDA data and sovereign GPUs — simultaneously a data moat, a compute subsidy, and a sales credential for risk-averse buyers."],
  ["Standard-setting:"," publish Bharat-EO-Bench and open -S weights to become the default benchmark and developer on-ramp (PLG funnel feeding paid -L/API)."],
  ["Land-and-expand:"," one state dept → adjacent depts → other states (templated) → central ministry (aggregated national contract)."],
  ["B2B via embedded finance:"," plug field-scores into insurers/banks/agri-fintech at point-of-need, monetizing per-policy/per-loan at volume."],
  ["Regulatory leverage:"," position as the sovereign, compliant alternative under NGP-2022 — disqualifies offshore competitors for government data."],
].forEach(([l,r])=> body.push(nl("n4", l, r)));
body.push(lead("Funding strategy: ","non-dilutive first — IndiaAI Mission, IN-SPACe, BIRAC, MeitY, ISRO/ANRF grants + GPU credits — to fund capital-heavy pretraining; blend with seed/Series-A VC and strategic checks from reinsurers/banks (who also become customers)."));
body.push(lead("TAM / SAM / SOM (India): ","TAM = geospatial economy ~$7.5 B (2025); SAM = EO-analytics + agri/water/disaster intelligence ~$1.5–2 B; SOM (Yr 3–4) = $30–60 M ARR."));
body.push(new Paragraph({ children:[ new PageBreak() ] }));

// ============================================================
// PART IX
// ============================================================
body.push(h1("Part IX — Patent Strategy"));
body.push(h2("9.1 Filing philosophy (built for Indian + global law)"));
body.push(p({t:"The Section 3(k) problem: ",bold:true},"the Indian Patents Act excludes 'computer programme per se', algorithms, mathematical methods, and business methods. A bare 'we trained a transformer' claim is not grantable in India. Strategy:"));
[
  ["Claim systems & technical effects, not algorithms."," Frame each invention as a system/apparatus integrating sensors + compute producing a tangible technical improvement (reduced bandwidth/storage, improved fusion under data loss, a physical control output)."],
  ["Dual-track jurisdictions."," File provisional in India first (IPO), then PCT within 12 months → national phase in US, EU, CN, JP, AU where pure ML method claims are more viable. India gets system/application claims; US/EU get algorithmic-method claims."],
  ["Trade-secret what you should not patent."," Keep model weights, exact training recipe, hyperparameters, and the data-cleaning pipeline as trade secrets (patents disclose; weights are better protected as secret + access control)."],
  ["Protect data via copyright/database rights + contracts,"," not patents (datasets are not patentable subject matter)."],
  ["Defensive publication"," for low-value-but-blockable ideas to prevent others patenting them."],
].forEach(([l,r])=> body.push(nl("n5", l, r)));

const patCols = [700,2050,6130,1200];
body.push(spacer());
body.push(h2("9.2 Patent concept portfolio (33 concepts)"));
body.push(h3("A. Novel architectures"));
body.push(table(patCols, ["ID","Title","Core inventive claim","Strength"], [
  ["P01","Multi-sensor dynamic tokenizer for L/S-band + geostationary EO","Hypernetwork generating patch-embed weights from {wavelength, polarization, geostationary cadence, GSD, sensor-id}, admitting NISAR L/S + INSAT into one token space","★★★★★"],
  ["P02","Monsoon-Phase-Aware Temporal Encoding (MPATE)","Temporal attention conditioned on hydro-meteorological phase + agro-calendar, derived from rainfall, not calendar date","★★★★★"],
  ["P03","Think-in-Modalities-then-Time generative forecaster","System that completes missing modalities then rolls embeddings forward in time to forecast future surface state","★★★★★"],
  ["P04","Modality- and region-routed MoE for EO","MoE FFN with expert gating biased by sensor-modality and geographic region/agro-zone","★★★★"],
  ["P05","Gap-aware continuous-time attention for cloud-interrupted series","Attention over irregularly-sampled timesteps with learned gap kernels for monsoon data dropout","★★★★"],
  ["P06","Field-boundary-aware pyramid super-resolution tokenizer","Laplacian-pyramid tokens + boundary-aware SR targeting fragmented smallholder parcels (≤5 m)","★★★★"],
  ["P07","Physics-informed EO adapter enforcing water/energy balance","Differentiable constraint layer penalizing violations of P−ET−Q−ΔS and energy balance in model outputs","★★★★★"],
  ["P08","Administrative-graph alignment module","Module binding pixel embeddings to state→district→tehsil→village hierarchy for queryable outputs","★★★"],
], 15));

body.push(h3("B. Dataset / data-engineering innovations"));
body.push(table(patCols, ["ID","Title","Core inventive claim","Strength"], [
  ["P09","Monsoon-phase auto-labeler","Method deriving onset/active/break/withdrawal labels by fusing IMD grids + satellite to phase-tag the corpus","★★★★"],
  ["P10","Quality-weighted multi-sensor harmonization pipeline","Co-registration + radiometric harmonization producing an ARD datacube with per-token confidence weights","★★★★"],
  ["P11","Tropical-monsoon cloud/shadow probabilistic masking + token dropout","Confidence-weighted masking tuned to Indian atmospheres, driving structured token dropout","★★★"],
  ["P12","Synthetic groundwater-label generation","Fusing InSAR subsidence + GRACE downscaling + well kriging to manufacture dense groundwater labels from sparse wells","★★★★★"],
  ["P13","SAR-coherence field-boundary corpus generation","Multi-temporal SAR-coherence + segmentation to auto-delineate smallholder parcels as labels","★★★★"],
  ["P14","Agro-stratified active-learning sampler","Sampling chips by monsoon-phase × agro-zone × crop-rarity to debias the Indian corpus","★★★"],
  ["P15","Privacy-preserving EO + farmer-data fusion (AgriStack)","Federated/DP fusion of field-level farmer data with EO without exposing PII","★★★★"],
], 15));

body.push(h3("C. Training innovations"));
body.push(table(patCols, ["ID","Title","Core inventive claim","Strength"], [
  ["P16","Stability-gated multi-objective curriculum scheduler","Auto-staged MIM→generative→physics→forecast loss weighting gated on training-stability metrics","★★★"],
  ["P17","Monsoon-conditioned SAR/optical co-training","Cross-modal contrastive+generative training with monsoon-phase-conditioned negative sampling","★★★★"],
  ["P18","Cross-architecture warm-start transplantation","Transferring optical-only FM weights (e.g., Prithvi) into a multimodal model via tokenizer-space alignment","★★★"],
  ["P19","Physics-regularized conformal forecasting head","Calibrated predictive intervals where conformal sets are constrained by physics-consistency","★★★★"],
  ["P20","Modality-dropout self-distillation","Teacher-student training under random modality dropout → embeddings robust to any missing sensor","★★★"],
  ["P21","Weather-FM-forced latent diffusion roll-forward","Temporal forecasting where Monsoon-WxC precipitation conditions the EO latent diffusion","★★★★"],
  ["P22","Edge-distillation to quantized pixel-timeseries student","Distilling the FM into a tiny INT8 pixel-timeseries model for village/mobile inference","★★★"],
], 15));

body.push(h3("D. India-specific application / system IP (strongest under Indian law)"));
body.push(table(patCols, ["ID","Title","Core inventive claim","Strength"], [
  ["P23","Groundwater storage-change estimation system","Apparatus fusing InSAR+GRACE+CGWB under water-balance constraint, outputting depletion-risk maps that trigger extraction-control decisions","★★★★★"],
  ["P24","Reservoir/irrigation forecasting + release-control coupling","Water-surface-area + command-area-greening model producing reservoir-release/gate recommendations under mass balance","★★★★★"],
  ["P25","Parametric crop-insurance trigger engine","System converting model-derived yield/biomass anomaly into an automated index that actuates insurance payouts (PMFBY)","★★★★"],
  ["P26","Multilingual administrative-grounded geospatial query system","Indic-language NL → admin-grounded EO answer/action (GeoLLM connector) as an integrated system","★★★★"],
  ["P27","Deltaic cyclone/flood inundation early-warning system","SAR+geostationary fusion producing timed inundation forecasts that trigger alerts/evacuation","★★★★"],
  ["P28","GLOF risk-monitoring method (Himalaya)","InSAR + optical lake-area + DEM pipeline producing glacial-lake-outburst risk alerts","★★★★"],
  ["P29","Urban-heat / informal-settlement detection system","Indian urban-morphology priors + LST → UHI + informal-settlement maps for planning/relief","★★★"],
  ["P30","Crop-type-under-cloud classifier (SAR-first)","Classifying Indian cropping systems from SAR temporal signatures keyed to the kharif/rabi/zaid calendar","★★★★"],
  ["P31","Sovereign EO embedding-product generation & serving system","Method generating seasonal, field-scale, quantized 64-d embeddings + on-prem serving with data-residency guarantees","★★★★"],
  ["P32","Multi-hazard disaster situational-awareness console","Fusing all heads into a single jurisdiction-aware operational picture with NL interface","★★★"],
  ["P33","Intercropping/multi-cropping disaggregation method","Separating mixed/intercropped signatures within a single smallholder pixel using spectral-temporal unmixing","★★★★"],
], 15));
body.push(p({t:"Total: 33 concepts. ",bold:true},"P01–P07, P12, P23, P24 are the crown-jewel filings (highest novelty × defensibility × commercial pull)."));

body.push(h2("9.3 Prioritized filing roadmap"));
body.push(table([1900,1300,3700,3180],
  ["Wave","Timing","File (provisional → PCT)","Rationale"],
  [
    ["Wave 1 (crown jewels)","M0–6","P01, P02, P03, P07, P12, P23","Highest novelty + revenue-adjacent; establish priority dates"],
    ["Wave 2 (apps that sell)","M6–12","P24, P25, P06, P13, P30, P33, P27","Tie IP to the first commercial wedges"],
    ["Wave 3 (platform/scale)","M12–24","P04, P05, P08, P21, P26, P31, P28","Protect platform + forecasting + sovereign serving"],
    ["Defensive publication","ongoing","P11, P16, P18, P22, P29, P32","Block competitors cheaply on lower-margin mechanisms"],
    ["Trade secret (no filing)","—","weights, exact recipe, full data pipeline","Disclosure-averse, hard to reverse-engineer"],
  ]));

body.push(h2("9.4 Freedom-to-operate & defensive notes"));
body.push(bl("Clear against incumbents:"," our claims pivot on polarization+geostationary tokenization, monsoon-phase conditioning, sub-surface fusion, physics constraints, and India application systems — orthogonal to IBM/ESA/Clay/Google published method claims. Run an FTO search before each PCT national-phase."));
body.push(bl("Inventorship/assignment hygiene:"," ensure ISRO/IN-SPACe/IndiaAI co-development agreements pre-assign IP cleanly; government co-funding can create ownership ambiguity — settle in the grant contract."));
body.push(bl("Open-core balance:"," open the -S weights + benchmark (goodwill, standard-setting) while patenting the systems and keeping -L/-XL weights + recipe proprietary."));
body.push(new Paragraph({ children:[ new PageBreak() ] }));

// ============================================================
// PART X — EMPIRICAL VALIDATION
// ============================================================
body.push(new Paragraph({ children:[ new PageBreak() ] }));
body.push(h1("Part X — Empirical Validation: A Laptop-Scale Pilot"));
body.push(p("Before committing to full sovereign pretraining, the central thesis of this blueprint — that a strong Earth-observation foundation-model representation, with only lightweight task heads, solves diverse Indian problems — was validated on a 16 GB CPU-only laptop using free Google Earth Engine. As a stand-in for the future Bharat-EarthFM embedding, we used Google's global AlphaEarth Foundations embedding (the closed, global incumbent this project aims to beat). Five Indian tasks across agriculture, disaster, and hydrology were run end-to-end, each with a leakage-free spatial-block evaluation."));

body.push(h2("10.1 Method"));
body.push(bl("Compute split:", " Earth Engine samples and processes server-side; the laptop trains lightweight heads (XGBoost / scikit-learn MLP) on the returned tables in minutes. No GPU."));
body.push(bl("Representation:", " frozen AlphaEarth 64-d embeddings as features — a proxy for the future Bharat-EarthFM embedding."));
body.push(bl("Evaluation:", " leakage-free spatial-block hold-out (whole 1-2 degree blocks assigned entirely to train or test) so scores reflect generalization to unseen regions, not spatial autocorrelation."));
body.push(bl("Baselines:", " every task pits the embedding against naive baselines (majority, geography, terrain, rainfall) to isolate its contribution."));

body.push(h2("10.2 Results across five domains"));
body.push(table([2100,1450,2350,4180],
  ["Task","Region","Best result","Key finding"],
  [
    ["Land cover + rare-class boost","All India","0.86 OA / 0.80 macro-F1","embedding >> geography (0.80 vs 0.23 macro-F1)"],
    ["Deep model vs XGBoost","All India","XGBoost 0.80 vs MLP 0.78","head barely matters; representation dominates"],
    ["Flood susceptibility","Assam 2022","flood-class F1 0.56","embedding alone 0.51 >> terrain 0.01"],
    ["Groundwater depletion (GRACE)","Punjab / NW","R2 0.34 -> 0.89","embedding is the dominant driver"],
    ["GRACE downscaling 110km->1.5km","Punjab+Haryana","downscaling R2 0.924","district hotspots (Rohtak, Sonepat...)"],
    ["Independent validation","Punjab","GRACE vs GRACE-FO agree, r 0.70","structural overdraft; -36 cm since 2002"],
  ]));

body.push(h2("10.3 Per-domain detail"));
body.push(bl("Land cover & rare classes:", " 42,045 India points, ESA WorldCover labels. Embedding+XGBoost reached 0.86 overall accuracy / 0.80 macro-F1 vs geography-only 0.23 and majority 0.06. Targeted rare-class sampling rescued mangroves (F1 0.00 to 0.98), moss (0.27 to 0.98) and snow (to 1.00); real targeted data outperformed synthetic SMOTE."));
body.push(bl("Deep model vs gradient boosting:", " an MLP and XGBoost on the same frozen embeddings landed within ~1% (both ~0.86 OA) — the head barely matters; the representation does the work."));
body.push(bl("Flood susceptibility (Assam 2022):", " labels from Sentinel-1 SAR (cloud-proof) minus JRC permanent water; prediction from static pre-event features only (non-circular). Flood-class F1 0.56; the embedding alone (0.51) far exceeded terrain elevation/slope (0.01), having learned the Brahmaputra's flood geomorphology."));
body.push(bl("Groundwater depletion (Punjab/NW India):", " GRACE terrestrial-water-storage trend 2003-2016; Punjab -0.86 cm/yr. Predicting the trend from surface drivers: rainfall-only R2 -0.14, climate+agriculture 0.34 (cropland a top driver), and +AlphaEarth embedding 0.89."));

body.push(h2("10.4 Groundwater deep-dive: downscaling and independent validation"));
body.push(bl("Downscaling:", " a model trained on coarse GRACE with fine-resolution predictors was applied at ~1.5 km over Punjab+Haryana (spatial-CV R2 0.924), yielding a district-level depletion ranking led by Rohtak (-1.85 cm/yr), Sonepat, Jhajjar, Faridabad and the southern-Haryana belt — directly usable for Atal Bhujal prioritization."));
body.push(p("The depletion target was then validated against independent satellites and processing (no InSAR or wells required):"));
body.push(table([2700,5300,2080],
  ["Check","Result","Verdict"],
  [
    ["Cross-mission","Punjab GRACE 2003-16 = -0.79 vs GRACE-FO 2018-24 = -0.84 cm/yr; spatial r 0.70","independent successor mission reproduces it"],
    ["Structural vs drought","India-wide TWS recovered post-2018 (+1.14 cm/yr) but Punjab kept declining (-0.84)","pumping, not rainfall"],
    ["Cross-center / method","CSR=GFZ=JPL spatial r 1.00; mascon vs spherical-harmonic pattern r 0.81","no single-center / method artifact"],
    ["Decline curve","-1.68 cm/yr, ~36 cm cumulative loss 2002-2024; matches Rodell/Famiglietti","correct sign and magnitude vs literature"],
  ]));

body.push(h2("10.5 The through-line, what it de-risks, and caveats"));
body.push(lead("The through-line: ", "across agriculture, disaster, and hydrology the foundation-model embedding was the dominant signal — beating geography (land cover), terrain (flood), and rainfall+agriculture (groundwater). This is direct empirical support for the program: build the sovereign representation, and lightweight heads solve India's problems."));
body.push(lead("What it de-risks: ", "the data pipeline, the benchmark methodology (Bharat-EO-Bench), the spatial-evaluation rigor, and the embeddings-plus-head productization were all proven on commodity hardware before any GPU spend. A model trained on Indian sensors (NISAR, EOS-04, INSAT, Resourcesat) plus in-situ (CGWB, CWC, IMD) is expected to beat this global, closed embedding on every one of these tasks."));
body.push(bl("Caveat:", " these used frozen global AlphaEarth embeddings, not a fine-tuned foundation model."));
body.push(bl("Caveat:", " groundwater is coarse (~110 km GRACE) and correlational; the downscaling is unvalidated against CGWB wells (scale-invariance assumed)."));
body.push(bl("Caveat:", " rare-class flood/land-cover scores reflect performance within sampled regions."));
body.push(bl("Staged next:", " fine-tuning Prithvi/Clay on image chips (Kaggle GPU) and real InSAR subsidence (ASF HyP3) are prepared as runnable notebooks."));

// ---- SOURCES ----
body.push(h1("Sources"));
body.push(p("Specifications and market figures were verified against the following primary sources:"));
[
  "Prithvi-EO-2.0 — arXiv 2412.02732 and Hugging Face (ibm-nasa-geospatial).",
  "Clay v1.5 — Clay Foundation model specification (release notes).",
  "Galileo — Tseng et al., PMLR v267 (2025).",
  "TerraMind — IBM Research blog and arXiv 2504.11171 (ICCV 2025).",
  "AlphaEarth Foundations — Google DeepMind blog and arXiv 2507.22291.",
  "NISAR data volume — NASA Earthdata (Getting Ready for NISAR); eoPortal.",
  "Sentinel-2 L2A / Sentinel-1 GRD-SLC sizes — Copernicus SentiWiki; Sentinel Hub docs.",
  "GRACE-FO Mascon RL06 — NASA JPL PO.DAAC.",
  "IMD 0.25° gridded rainfall — IMD Pune (Climate Prediction Group).",
  "ISRO Bhuvan / NOEDA archive; NISAR — NASA Earthdata.",
  "India geospatial economy — Geospatial World; Mordor Intelligence.",
  "PMFBY scale & budget — IBEF. Atal Bhujal Yojana — IBEF; India Water Portal.",
  "India agritech / agri-fintech market — Inc42; Tracxn; IBEF.",
].forEach(s=>body.push(bullet(s)));

// ============================================================
// DOCUMENT
// ============================================================
const doc = new Document({
  creator: "Bharat-EarthFM Founding Team",
  title: "Bharat-EarthFM Technical Blueprint & Strategy Dossier",
  description: "Sovereign India-scale Earth Observation Foundation Model",
  styles: {
    default: { document: { run: { font: "Calibri", size: 21 } } },
    paragraphStyles: [
      { id:"Title", name:"Title", basedOn:"Normal", next:"Normal", quickFormat:true,
        run:{ size:56, bold:true, font:"Calibri", color:"1F3864" } },
      { id:"Heading1", name:"Heading 1", basedOn:"Normal", next:"Normal", quickFormat:true,
        run:{ size:30, bold:true, font:"Calibri", color:"1F3864" },
        paragraph:{ spacing:{ before:280, after:160 }, outlineLevel:0,
          border:{ bottom:{ style:BorderStyle.SINGLE, size:8, color:"C55A11", space:4 } } } },
      { id:"Heading2", name:"Heading 2", basedOn:"Normal", next:"Normal", quickFormat:true,
        run:{ size:25, bold:true, font:"Calibri", color:"2E5496" },
        paragraph:{ spacing:{ before:200, after:120 }, outlineLevel:1 } },
      { id:"Heading3", name:"Heading 3", basedOn:"Normal", next:"Normal", quickFormat:true,
        run:{ size:22, bold:true, font:"Calibri", color:"404040" },
        paragraph:{ spacing:{ before:140, after:80 }, outlineLevel:2 } },
    ]
  },
  numbering: { config: numConfig },
  sections: [{
    properties: { page: { size:{ width:12240, height:15840 }, margin:{ top:1080, right:1080, bottom:1080, left:1080 } } },
    headers: { default: new Header({ children:[ new Paragraph({ alignment:AlignmentType.RIGHT,
      border:{ bottom:{ style:BorderStyle.SINGLE, size:4, color:"BFBFBF", space:2 } },
      children:[ new TextRun({ text:"Bharat-EarthFM — Technical Blueprint & Strategy Dossier", size:14, color:"808080" }) ] }) ] }) },
    footers: { default: new Footer({ children:[ new Paragraph({
      tabStops:[{ type:TabStopType.RIGHT, position: 10080 }],
      children:[ new TextRun({ text:"CONFIDENTIAL", size:14, color:"808080" }),
        new TextRun({ text:"\tPage ", size:14, color:"808080" }),
        new TextRun({ children:[ PageNumber.CURRENT ], size:14, color:"808080" }),
        new TextRun({ text:" of ", size:14, color:"808080" }),
        new TextRun({ children:[ PageNumber.TOTAL_PAGES ], size:14, color:"808080" }) ] }) ] }) },
    children: body
  }]
});

Packer.toBuffer(doc).then(buf => { fs.writeFileSync("Bharat-EarthFM_Blueprint.docx", buf); console.log("WROTE Bharat-EarthFM_Blueprint.docx", buf.length, "bytes"); });
