# XPrice: Explainable AI for Transparent Dynamic Pricing at Careem
## MIT 622 — Data Analytics for Managers | Final Project Plan
### Group 1: Mohammadsadegh Solouki · Artin Fateh Basharzad · Fatema Alblooshi
### Submission: 2 May 2026 | Presentation: 3 May 2026

---

## Project Overview

**Research Title:**  
*XPrice: An Explainable AI Framework for Transparent Dynamic Pricing in Urban Ride-Hailing — A Dubai Case Study*

**Core Idea:**  
Careem's surge pricing is a black-box ML system. Riders don't know why their fare is AED 42 vs AED 18. Operations managers can't interrogate why prices spiked. This project proposes and demonstrates an XAI (Explainable AI) layer using SHAP that makes every pricing decision transparent — for both internal operators and end-users. We construct a high-fidelity synthetic "mirror dataset" of Dubai ride-hailing operations, train an ML pricing model, apply SHAP for local and global explanations, and deliver a live interactive application that simulates a "Request a Ride" experience with full price breakdowns.

**Research Methodology:** Design Science Research (DSR) — Peffers et al. (2007)  
Six phases: Problem Identification → Objectives → Design & Development → Demonstration → Evaluation → Communication (the paper).

**Data Strategy:**  
We construct a synthetic mirror dataset modeled on Dubai's real ride-hailing operational patterns using public knowledge (Careem Engineering Blog, e& reports, WTW 2024 MENA report, OpenWeatherMap historical data, Dubai events calendar). The paper explicitly states this is a researcher-constructed mirror dataset used to demonstrate the proposed XAI methodology — not proprietary Careem data.

---

## Project Phases & Step-by-Step Plan

---

### PHASE 1 — Dataset Construction
**Goal:** Build a rich, realistic Dubai ride-hailing dataset (~150,000–200,000 records

#### 1.1 Define Feature Set

**Spatial features (Dubai geography):**
- `pickup_zone` / `dropoff_zone` — 12 Dubai zones: Downtown, Marina, JBR, DIFC, Deira, Bur Dubai, Jumeirah, Al Quoz, Business Bay, Dubai Hills, Airport (DXB), Sharjah Border
- `pickup_lat` / `pickup_lon` / `dropoff_lat` / `dropoff_lon` — realistic coordinates per zone
- `route_distance_km` — realistic road distances between zone pairs (via lookup table)
- `is_airport_ride` — binary flag (DXB pickup or dropoff)
- `zone_demand_tier` — Low / Medium / High based on zone commercial density

**Temporal features:**
- `timestamp` — full datetime across Jan–Dec 2025
- `hour` — 0–23
- `day_of_week` — 0–6 (Fri/Sat = UAE weekend)
- `is_weekend` — binary (Fri/Sat in UAE)
- `month` — 1–12
- `is_ramadan` — binary (Ramadan 2025: 1–29 March)
- `is_uae_public_holiday` — New Year, Eid Al Fitr, Eid Al Adha, UAE National Day (2–3 Dec), Islamic New Year, Prophet's Birthday
- `is_peak_hour` — 07:00–09:00 and 17:00–20:00
- `is_late_night` — 00:00–04:00

**Event features (Dubai 2025):**
- `active_event` — None / DSF (Dubai Shopping Festival, Jan–Feb) / GITEX (Oct) / Dubai Airshow (Nov) / NYE Burj / Eid / F1 Weekend / Concert / Sports
- `event_proximity_km` — distance of pickup to nearest active event venue
- `event_demand_multiplier` — estimated demand boost (1.0–2.5x)

**Weather features (Dubai-specific):**
- `temperature_c` — realistic Dubai temps (18°C Jan – 42°C Jul)
- `humidity_pct` — Dubai humidity profile
- `is_rain` — rare but causes large surge (Dubai averages 8 rain days/year)
- `is_sandstorm` — binary (reduces visibility, boosts ride demand)
- `weather_demand_factor` — composite weather impact on demand

**Supply features:**
- `available_captains_zone` — captains available in pickup zone at time of request
- `captain_acceptance_rate` — zone-hour level acceptance rate
- `product_type` — Go / Go+ / Business / Hala Taxi / eBike
- `payment_method` — Card / Cash / Careem Pay / Careem Plus

**Pricing features (target construction):**
- `base_fare_aed` — product base fare (Go: AED 6, Business: AED 12)
- `per_km_rate` — product rate per km (Go: AED 2.1, Business: AED 3.8)
- `surge_multiplier` — calculated from demand/supply ratio (1.0–2.5x)
- `final_price_aed` — TARGET VARIABLE (base + distance + surge + time factors)
- `price_per_km` — derived

**Outcome features:**
- `booking_status` — Completed / Captain Cancelled / Customer Cancelled / No Captain
- `wait_time_minutes` — VTAT
- `captain_rating` / `customer_rating`

#### 1.2 Pricing Formula
```
final_price = (base_fare + per_km_rate × distance) × surge_multiplier × weather_factor × event_factor
surge_multiplier = 1.0 + demand_supply_gap × 0.4  (capped at 2.5)
demand_supply_gap = (ride_requests_zone_hour / avg_requests_zone_hour) - (captains_available / avg_captains)
```

#### 1.3 Deliverable
- File: `data/processed/dubai_rides_2025.csv` (~150k–200k rows, 30+ features)
- Script: `data/generate_dataset.py`
- Documentation: `data/DATA_DICTIONARY.md`

---

### PHASE 2 — Exploratory Data Analysis (EDA)
**Goal:** Understand the dataset, validate it mirrors real Dubai patterns, generate descriptive insights for the paper.

Key analyses:
- Price distribution by zone, product, hour, weather condition
- Surge multiplier heatmap by hour × day
- Event impact on price (before/during/after)
- Ramadan vs non-Ramadan comparison
- Rain days: price spike analysis
- Feature correlation matrix

Deliverable: `notebooks/02_eda.ipynb` + key figures saved to `docs/figures/`

---

### PHASE 3 — ML Price Prediction Model
**Goal:** Train a model that predicts `final_price_aed` with high accuracy, compatible with SHAP.

**Models to train:**
1. XGBoost (primary — best SHAP compatibility, high accuracy)
2. LightGBM (comparison)
3. Random Forest (baseline)

**Evaluation metrics:** RMSE, MAE, R², feature importance

**Why XGBoost:** Boosted tree models are natively supported by TreeExplainer in SHAP, giving exact (not approximate) Shapley values — much faster and more reliable than KernelExplainer for neural networks.

**Feature engineering:**
- One-hot encode: zone, product, payment, event type
- Cyclical encoding: hour (sin/cos), day of week (sin/cos), month (sin/cos)
- Interaction features: is_rain × is_peak_hour, is_ramadan × hour, event × zone

**Train/test split:** 80/20 stratified by month (to test generalization across seasons)

Deliverables:
- `models/train_model.py`
- `models/saved/xgboost_price_model.pkl`
- `notebooks/03_model_training.ipynb`

---

### PHASE 4 — XAI Layer (SHAP Analysis)
**Goal:** Apply SHAP to the trained XGBoost model for both global and local explanations.

#### 4.1 Global Explanations (Operations Manager view)
- **SHAP summary beeswarm plot** — which features globally drive price variation
- **SHAP bar chart** — mean absolute SHAP values (ranked feature importance)
- **SHAP dependence plots** — how `surge_multiplier`, `hour`, `distance`, `is_rain`, `event_demand_multiplier` interact with price
- **SHAP heatmap** — feature impact across time of day × zone

#### 4.2 Local Explanations (Individual ride — End-User view)
- **SHAP waterfall plot** — for a single ride: base price → contribution of each feature → final price
  - e.g., "Base: AED 12.00 | +AED 8.20 (distance 14km) | +AED 4.10 (GITEX nearby) | +AED 2.80 (peak hour) | +AED 1.50 (rain) | = AED 28.60"
- **SHAP force plot** — interactive version of waterfall
- **Natural language explanation generator** — converts SHAP values into plain English sentences for the user-facing app

#### 4.3 LIME Comparison
Apply LIME to the same predictions as a methodological comparison — discuss differences in consistency, stability, computational cost.

Deliverables:
- `models/xai_analysis.py`
- `notebooks/04_xai_analysis.ipynb`
- Figures saved to `docs/figures/shap/`

---

### PHASE 5 — Interactive Application (XPrice App)
**Goal:** Build a dual-audience Streamlit application demonstrating the XAI framework.

#### Architecture
```
app/
├── app.py                      # Entry point + navigation
├── pages/
│   ├── 1_ride_simulator.py     # End-user: Request a Ride + Price Explanation
│   ├── 2_operations_xai.py     # Ops Manager: Global SHAP Dashboard
│   └── 3_feature_explorer.py   # Deep-dive: Dependence plots
└── utils/
    ├── model_loader.py         # Load trained XGBoost model
    ├── shap_engine.py          # Generate SHAP values + plots
    ├── weather_api.py          # OpenWeatherMap API integration
    ├── geo_utils.py            # Dubai zone lookup, distance calc
    └── nlp_explainer.py        # SHAP → natural language
```

#### Page 1 — Ride Simulator (End-User View)
- **Map interface** (Folium embedded in Streamlit) — user clicks to set pickup and dropoff in Dubai
- Zone auto-detected from coordinates
- Route distance estimated from zone-pair lookup table
- **Real-time weather** pulled from OpenWeatherMap API (Dubai, current conditions)
- **Active events** detected from a hardcoded 2025 events calendar
- Time auto-filled from system clock (or user-selectable)
- **Price estimate** generated by ML model
- **SHAP waterfall** showing exactly what drives the price
- **Natural language breakdown:** "Your fare is AED 31.50. The biggest factors are: distance (14km adds AED 8.20), current event (GITEX adds AED 4.10), and the time of day (peak hour adds AED 2.80). Weather is clear so no weather surcharge applies."
- Product selector: Go / Go+ / Business / Hala

#### Page 2 — Operations XAI Dashboard (Internal View)
- Global SHAP beeswarm and bar chart
- Zone-level average SHAP contribution heatmap
- Hour-of-day price driver decomposition
- Ramadan vs baseline SHAP comparison
- Rain event SHAP spike analysis
- Filter by: zone, product, month, event

#### Page 3 — Feature Explorer
- SHAP dependence plots (interactive)
- Partial dependence plot for any selected feature
- "What-if" simulator: change one input, see how SHAP values shift

#### External APIs Used
- **OpenWeatherMap API** (free tier) — current weather for Dubai
- **Folium** (open source) — interactive map rendering
- **OSRM** (optional, open source) — more realistic route distance (fallback: zone-pair lookup table)

Deliverable: Fully functional Streamlit app, runnable with `streamlit run app/app.py`

---

### PHASE 6 — Paper Writing
**Goal:** Write the 1,800–2,200 word academic paper following the assignment structure.

See `PAPER_OUTLINE.md` for full section-by-section breakdown with arguments and citations.

Key differentiators from the Group Project:
1. Methodologically distinct: this is an XAI framework paper (DSR), not a descriptive dashboard
2. Dual-audience framing (internal ops + external user) — original contribution
3. Mirror dataset explicitly disclosed and justified
4. Live app as DSR artifact/instantiation
5. Minimum 10 academic references from high-quality journals

Deliverable: `paper/XPrice_Final_Paper.docx`

---

### PHASE 7 — Presentation Slides
**Goal:** Build the 10–15 minute group presentation.

Structure:
1. Problem (2 min) — Why black-box pricing hurts trust and operations
2. Literature & Theory (2 min) — XAI, SHAP, DSR methodology
3. Critical Evaluation of Group Project (2 min) — What the dashboard lacked
4. Our Solution: XPrice Framework (3 min) — Mirror dataset + model + SHAP
5. Live App Demo (3 min) — Ride simulator + ops dashboard
6. Business Impact & Recommendations (2 min)
7. Q&A

Deliverable: `paper/XPrice_Presentation.pptx`

---

## Timeline

| Phase | Task | Target |
|-------|------|--------|
| 1 | Dataset generation script | Day 1 |
| 2 | EDA + validation | Day 1–2 |
| 3 | Model training + evaluation | Day 2 |
| 4 | SHAP analysis + figures | Day 2–3 |
| 5 | Streamlit app (core pages) | Day 3–4 |
| 5 | Weather API + map integration | Day 4 |
| 6 | Paper draft | Day 3–5 |
| 7 | Presentation slides | Day 5 |
| — | Final review + submission | 2 May 2026 |

---

## Technology Stack

| Component | Tool |
|-----------|------|
| Dataset generation | Python (pandas, numpy, faker) |
| ML Model | XGBoost, LightGBM, scikit-learn |
| XAI | SHAP (shap library), LIME |
| Application | Streamlit |
| Maps | Folium |
| Weather API | OpenWeatherMap (free tier) |
| Visualization | matplotlib, seaborn, plotly |
| Paper | MS Word (.docx) |
| Slides | PowerPoint (.pptx) |
| Version notes | This PLAN.md |

---

## Workspace Structure

```
Data final proj/
├── PLAN.md                         ← This file
├── REFERENCES.md                   ← All academic references
├── PAPER_OUTLINE.md                ← Section-by-section paper plan
│
├── data/
│   ├── generate_dataset.py         ← Dubai ride dataset generator
│   ├── DATA_DICTIONARY.md          ← Every feature explained
│   ├── dubai_zones.json            ← Zone centroids + demand tiers
│   ├── events_2025.json            ← Dubai events calendar
│   ├── raw/                        ← Any raw external inputs
│   ├── processed/
│   │   └── dubai_rides_2025.csv    ← Final generated dataset
│   └── external/
│       └── weather_history.json    ← Dubai 2025 weather patterns
│
├── models/
│   ├── train_model.py              ← Training pipeline
│   ├── xai_analysis.py             ← SHAP + LIME analysis
│   └── saved/
│       ├── xgboost_price_model.pkl
│       └── feature_columns.pkl
│
├── app/
│   ├── app.py                      ← Streamlit entry point
│   ├── pages/
│   │   ├── 1_ride_simulator.py     ← End-user: price + SHAP explanation
│   │   ├── 2_operations_xai.py     ← Ops: global SHAP dashboard
│   │   └── 3_feature_explorer.py   ← Dependence + what-if
│   └── utils/
│       ├── model_loader.py
│       ├── shap_engine.py
│       ├── weather_api.py          ← OpenWeatherMap integration
│       ├── geo_utils.py            ← Zone detection + distance
│       └── nlp_explainer.py        ← SHAP → plain English
│
├── notebooks/
│   ├── 01_data_generation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_xai_analysis.ipynb
│
├── docs/
│   ├── architecture.md             ← System architecture notes
│   └── figures/
│       └── shap/                   ← Saved SHAP plots for paper
│
└── paper/
    ├── XPrice_Final_Paper.docx     ← Submission paper
    └── XPrice_Presentation.pptx   ← Presentation slides
```

---

## Key Design Decisions & Rationale

**Why XGBoost over neural networks?**  
SHAP's TreeExplainer gives exact Shapley values for tree-based models in O(TLD²) time. Neural network explanations via KernelExplainer are approximate and 100x slower. For a demonstration that needs to run in real-time in a web app, XGBoost is the right choice.

**Why a mirror dataset rather than real data?**  
Careem's operational data is proprietary. Our synthetic dataset is constructed using publicly documented operational parameters (Careem Engineering Blog, e& FY2025 report, WTW 2024 MENA ride-hailing report) to produce a statistically plausible representation. This is explicitly disclosed in the paper's methodology section. Using a researcher-constructed dataset to demonstrate a proposed framework is a recognized approach in Design Science Research (Peffers et al., 2007).

**Why dual-audience (internal + external)?**  
Existing XAI literature focuses almost exclusively on model developers and regulators. The contribution of explaining pricing to end-users in plain language at the point of booking is novel, especially in the MENA context where algorithmic transparency is an emerging regulatory concern.

**Why Dubai only?**  
Dubai provides the richest operational context: diverse zone types (airport, tourist, commercial, residential), the full event calendar (DSF, GITEX, NYE at Burj), and UAE-specific factors (Ramadan, Friday/Saturday weekend). Limiting to one city allows deeper, more realistic feature engineering than spreading thinly across five cities.
