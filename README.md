# XPrice: Explainable Ride Pricing for Dubai

XPrice is a research-oriented Streamlit application and analytics pipeline for explainable ride-hailing pricing in Dubai. The project generates a synthetic 2025 ride dataset, trains a coordinate-first XGBoost fare model, exports tree contribution artefacts for local and global explanations, and serves a multi-page Streamlit app for rider, operations, and analyst workflows.

## Model Performance

| Metric | Value |
|--------|-------|
| Test R² | **0.9880** |
| Test RMSE | **AED 5.56** |
| Test MAE | **AED 3.23** |
| CV R² (5-fold block) | **0.9878 ± 0.0003** |
| 90% prediction interval | **± AED 7.28** (conformal, exact 90% coverage) |
| Training rows | 119,970 |
| Features | 76 |
| SHAP explanation pool | 5,000 rides |

## What Is In The Repo

- A synthetic 165,000-row Dubai ride-hailing dataset with 72 columns, including polygon-resolved pickup and dropoff neighborhood metadata.
- A coordinate-first pricing model that learns from route geometry, density, weather, demand, and traffic signals.
- XAI artefacts built with XGBoost native tree contribution outputs (no shap library dependency).
- Conformal prediction intervals (±AED 7.28 for 90% coverage) displayed on every fare quote.
- A Streamlit app with three pages:
  - **Rider Simulator**: draggable pickup/dropoff pins, official neighborhood boundary overlay, polygon-resolved area labels, fare quote, 90% prediction interval, and waterfall explanation.
  - **Operations Dashboard**: filtered global contribution analysis with zone heatmaps, hourly patterns, event breakdowns, and a residuals tab for model quality audit.
  - **Feature Explorer**: PDP + ICE curves, SHAP dependence plots, what-if lab with manual overrides, and a **LIME vs SHAP** method comparison panel.

## Project Structure

```text
zone_config.py
app/
  app.py
  pages/
  utils/
data/
  dubai_neighborhoods.geojson
  generate_dataset.py
  data/
  processed/
models/
  feature_engineering.py
  train_model.py
  xai_analysis.py
  saved/
docs/
paper/
```

## Requirements

- Python 3.12+
- Packages from `requirements.txt`

Install dependencies with:

```powershell
python -m pip install -r requirements.txt
```

## Run Order

If you want to rebuild everything from scratch, use this order:

1. Generate the synthetic dataset.
2. Train the model.
3. Regenerate the XAI artefacts.
4. Launch Streamlit.

Commands:

```powershell
python data/generate_dataset.py
python models/train_model.py
python models/xai_analysis.py
streamlit run app/app.py
```

## API Keys

The app supports two optional live data integrations:

- `OPENWEATHER_API_KEY` for current weather.
- `TOMTOM_API_KEY` for current traffic flow.

If no keys are configured, the app still works. It falls back to the built-in synthetic weather and traffic models.

### Where To Put The Keys

Recommended local setup:

1. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`.
2. Paste your real keys into `.streamlit/secrets.toml`.
3. Restart Streamlit.

Example:

```toml
OPENWEATHER_API_KEY = "your-real-openweathermap-key"
TOMTOM_API_KEY = "your-real-tomtom-key"
```

`.streamlit/secrets.toml` is already ignored by Git in `.gitignore`.

You can also use environment variables instead of a secrets file:

```powershell
$env:OPENWEATHER_API_KEY = "your-real-openweathermap-key"
$env:TOMTOM_API_KEY = "your-real-tomtom-key"
streamlit run app/app.py
```

### How To Get The Keys

OpenWeatherMap:

1. Create an account at https://openweathermap.org/.
2. Open your account dashboard and create an API key.
3. Use the Current Weather API key in this project.

TomTom:

1. Create an account at https://developer.tomtom.com/.
2. Create a new project or application from the developer dashboard.
3. Generate an API key with access to the Traffic Flow service.

### Important Runtime Note

Live weather and live traffic are only used for rides scheduled for the current day. Historical or future dates intentionally fall back to the synthetic research model so the app remains deterministic for the 2025 study scenario.

## Main Files

- `data/generate_dataset.py`: Generates the synthetic Dubai ride dataset.
- `models/feature_engineering.py`: Shared training and inference feature logic.
- `models/train_model.py`: Trains and saves the XGBoost fare model.
- `models/xai_analysis.py`: Builds the saved contribution artefacts and figures.
- `zone_config.py`: Shared pricing-zone metadata plus neighborhood polygon lookup/overlay data.
- `app/app.py`: Streamlit landing page.
- `app/pages/1_ride_simulator.py`: Rider-facing quote simulator.
- `app/pages/2_operations_xai.py`: Operations contribution dashboard.
- `app/pages/3_feature_explorer.py`: Analyst feature and what-if explorer.

## Validation Notes

- The repo uses XGBoost native `pred_contribs=True` for explanation export.
- The app is designed to work without external API keys.
- Route context uses OSRM when available and a deterministic fallback when not.
- Neighborhood labels and map boundaries come from the shared Dubai GeoJSON in `zone_config.py`, with Shapely-backed point-in-polygon lookup and centroid fallback outside mapped polygons.

## Current Outputs

- Processed dataset: `data/processed/dubai_rides_2025.csv`
- Model artefacts: `models/saved/`
- SHAP-style figures: `docs/figures/shap/`

## License / Academic Use

This repository is structured as an academic final-project artefact. The dataset is synthetic and intended to mirror public knowledge about Dubai ride-hailing operations rather than represent proprietary Careem data.