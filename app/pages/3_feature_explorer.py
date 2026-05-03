from __future__ import annotations

import sys
from calendar import monthrange
from datetime import date, datetime, time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
for path in (APP_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils.domain import DEFAULT_DROPOFF_POINT, DEFAULT_PICKUP_POINT, PAYMENT_METHODS, PRODUCT_NAMES, build_inference_frame, build_trip_record
from utils.model_loader import estimate_trip_interval, get_global_interval_half_width, get_interval_basis_percent, load_feature_columns, load_metrics, load_model, load_model_version, load_shap_bundle
from utils.nlp_explainer import build_explanation
from utils.routing_api import get_route_context
from utils.shap_engine import compute_local_contributions, plot_dependence, plot_waterfall
from utils.ui import apply_theme, card, hero, section_header, sidebar_brand, whatif_result
from utils.weather_api import get_weather


try:
    from lime.lime_tabular import LimeTabularExplainer as _LimeTabularExplainer
    _LIME_AVAILABLE = True
except ImportError:
    _LIME_AVAILABLE = False


apply_theme()

# Load all resources first so they are available for the sidebar and page
model = load_model()
feature_columns = load_feature_columns()
bundle = load_shap_bundle()
metrics = load_metrics()
version = load_model_version()
_interval_basis_percent = get_interval_basis_percent(metrics)
_interval_half_width = get_global_interval_half_width(metrics)
sample_features = bundle["sample_features"].copy()
contrib_values = bundle["values"]


@st.cache_resource(show_spinner=False)
def _get_lime_explainer():
    """Build and cache the LIME tabular explainer fitted on the 5k SHAP sample."""
    if not _LIME_AVAILABLE:
        return None
    return _LimeTabularExplainer(
        training_data=sample_features.values,
        feature_names=list(feature_columns),
        mode="regression",
        random_state=42,
    )


@st.cache_data(show_spinner=False)
def _run_lime_explanation(ride_idx: int, _feature_columns_key: str) -> dict:
    """Return LIME weights + surrogate prediction for a single ride. Cached by ride index."""
    _explainer = _get_lime_explainer()
    if _explainer is None:
        return {"weights": [], "local_pred": None, "model_pred": None}

    def _predict_fn(arr):
        return model.predict(pd.DataFrame(arr, columns=feature_columns))

    _row = sample_features.iloc[ride_idx].values
    _exp = _explainer.explain_instance(
        _row,
        _predict_fn,
        num_features=12,
    )
    _model_pred = float(_predict_fn(_row.reshape(1, -1))[0])
    _local_pred = float(_exp.local_pred[0]) if _exp.local_pred is not None else None
    return {"weights": _exp.as_list(), "local_pred": _local_pred, "model_pred": _model_pred}


# ---- Sidebar ----
with st.sidebar:
    sidebar_brand()
    st.markdown("### Model info")
    st.markdown(
        f"- **Test R\u00b2:** {metrics['test']['r2']:.4f}  \n"
        f"- **RMSE:** AED {metrics['test']['rmse']:.2f}  \n"
        f"- **CV R\u00b2:** {metrics['cv']['r2_mean']:.4f} \u00b1 {metrics['cv']['r2_std']:.4f}  \n"
        f"- **{_interval_basis_percent}% PI:** \u00b1AED {_interval_half_width:.2f}"
    )
    if version.get("training_date"):
        st.caption(f"Trained {version['training_date'][:10]}")

hero(
    "Feature Explorer",
    "Analyst workbench",
    "Explore how any single driver influences fares across the full sample. Inspect its SHAP contribution shape, plot the marginal response curve, then run a what-if scenario with custom coordinates and manual overrides for distance, traffic, and supply.",
)

continuous_features = [
    column
    for column in sample_features.columns
    if pd.api.types.is_numeric_dtype(sample_features[column]) and sample_features[column].nunique() > 12
]

section_header("Feature to explore")
selected_feature = st.selectbox("Select a continuous feature", continuous_features, index=0, label_visibility="collapsed")

left, right = st.columns([1.1, 0.9], gap="large")
with left:
    section_header("Dependence view")
    st.pyplot(plot_dependence(contrib_values, sample_features, selected_feature), width="stretch")

with right:
    section_header("Partial dependence + ICE curves")
    baseline_row = sample_features.mean().to_frame().T.reset_index(drop=True)
    low = float(sample_features[selected_feature].quantile(0.05))
    high = float(sample_features[selected_feature].quantile(0.95))
    sweep_values = np.linspace(low, high, 30)

    # Mean PDP: sweep the selected feature against the mean baseline row
    predictions = []
    for value in sweep_values:
        scenario = baseline_row.copy()
        scenario[selected_feature] = value
        predictions.append(float(model.predict(scenario)[0]))

    # ICE: 15 random rides from the SHAP sample — shows spread of individual responses
    _N_ICE = 15
    _ice_idx = np.random.RandomState(42).choice(
        len(sample_features), min(_N_ICE, len(sample_features)), replace=False
    )
    _ice_sample = sample_features.iloc[_ice_idx].reset_index(drop=True)
    _ice_traces = []
    for _i in range(len(_ice_sample)):
        _ice_row = _ice_sample.iloc[[_i]].copy()
        _ice_preds = []
        for _v in sweep_values:
            _r = _ice_row.copy()
            _r[selected_feature] = _v
            _ice_preds.append(float(model.predict(_r)[0]))
        _ice_traces.append(_ice_preds)

    sweep_fig = go.Figure()
    # ICE traces (thin, translucent gray)
    for _trace in _ice_traces:
        sweep_fig.add_trace(go.Scatter(
            x=sweep_values, y=_trace,
            mode="lines",
            line=dict(color="rgba(100,116,139,0.22)", width=1),
            showlegend=False,
            hoverinfo="skip",
        ))
    # Mean PDP (thick teal)
    sweep_fig.add_trace(go.Scatter(
        x=sweep_values, y=predictions,
        mode="lines+markers",
        name="Mean (PDP)",
        line=dict(color="#0d9488", width=2.5),
        marker=dict(color="#0d9488", size=6),
    ))
    sweep_fig.update_layout(
        title=f"PDP + ICE: {selected_feature}",
        height=360,
        margin={"l": 0, "r": 0, "t": 44, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#0f172a", "family": "Inter, Segoe UI, sans-serif"},
        title_font={"size": 14, "color": "#0f172a"},
        xaxis={"gridcolor": "#f1f5f9", "title": {"text": selected_feature, "font": {"color": "#64748b"}}},
        yaxis={"gridcolor": "#f1f5f9", "title": {"text": "Predicted fare (AED)", "font": {"color": "#64748b"}}},
        legend={"font": {"size": 11, "color": "#64748b"}},
    )
    st.plotly_chart(sweep_fig, width="stretch")
    st.caption("Gray lines are Individual Conditional Expectation (ICE) curves for 15 random rides. Teal line is the population mean (PDP).")

st.divider()
section_header("LIME vs SHAP \u2014 Local explanation comparison")
card(
    "Method comparison",
    "SHAP uses Shapley values from cooperative game theory \u2014 guaranteed to be consistent, locally accurate, and globally coherent. "
    "LIME fits a sparse linear surrogate model on perturbed samples around a single prediction point. "
    "Both explain <em>why</em> the model predicted a specific fare; the panel below lets you compare them ride-by-ride.",
)
_lime_ride_idx = st.slider(
    "Select ride from the 5,000-ride SHAP sample to explain",
    0, len(sample_features) - 1, 0, key="lime_ride_idx",
)
lime_left, lime_right = st.columns(2, gap="large")

with lime_left:
    section_header("SHAP (tree contribution decomposition)")
    _single_frame = sample_features.iloc[[_lime_ride_idx]]
    _l_contrib, _l_base, _l_pred = compute_local_contributions(model, _single_frame)
    st.pyplot(plot_waterfall(_l_contrib, _l_base, _l_pred), width="stretch")
    st.caption(f"Predicted: AED {_l_pred:.2f}  \u00b7  Base: AED {_l_base:.2f}")

with lime_right:
    section_header("LIME (local linear surrogate)")
    if _LIME_AVAILABLE:
        with st.spinner("Computing LIME explanation\u2026"):
            _lime_result = _run_lime_explanation(_lime_ride_idx, str(feature_columns[:5]))
        _lime_weights = _lime_result["weights"]
        _lime_local_pred = _lime_result["local_pred"]
        _lime_model_pred = _lime_result["model_pred"]
        if _lime_weights:
            _lw_frame = pd.DataFrame(_lime_weights, columns=["feature", "weight"]).sort_values("weight")
            _lime_bar = px.bar(
                _lw_frame,
                y="feature",
                x="weight",
                orientation="h",
                color="weight",
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0,
                title="LIME local weights",
                template="plotly_white",
            )
            _lime_bar.update_layout(
                height=420,
                margin={"l": 0, "r": 0, "t": 44, "b": 0},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#0f172a", "family": "Inter, Segoe UI, sans-serif"},
                xaxis={"title": "LIME weight (AED approximation)", "gridcolor": "#f1f5f9"},
                yaxis={"gridcolor": "#f1f5f9"},
                coloraxis_showscale=False,
            )
            st.plotly_chart(_lime_bar, width="stretch")
            _lime_pred_str = f"AED {_lime_local_pred:.2f}" if _lime_local_pred is not None else "N/A"
            _lime_model_str = f"AED {_lime_model_pred:.2f}" if _lime_model_pred is not None else "N/A"
            st.caption(
                f"LIME surrogate prediction: **{_lime_pred_str}** \u00b7 XGBoost model prediction: **{_lime_model_str}**  \n"
                "LIME weights are estimates from a local linear model fitted on perturbed samples around this ride. "
                "Values approximate contribution magnitude but may differ from SHAP due to the linear surrogate assumption."
            )
    else:
        st.info("Install the `lime` package to enable this panel: `pip install lime`")
        _lime_result = {"weights": [], "local_pred": None, "model_pred": None}
        _lime_weights, _lime_local_pred, _lime_model_pred = [], None, None

# ---- Prediction reconciliation & feature agreement table ----
st.markdown("<br>", unsafe_allow_html=True)
section_header("SHAP vs LIME \u2014 Prediction reconciliation & feature agreement")

_pred_col1, _pred_col2, _pred_col3 = st.columns(3, gap="small")
_pred_col1.metric(
    "XGBoost prediction (SHAP basis)",
    f"AED {_l_pred:.2f}",
    help="Exact model output. SHAP values sum to (prediction \u2212 base value).",
)
if _LIME_AVAILABLE and _lime_result.get("model_pred") is not None:
    _pred_col2.metric(
        "XGBoost prediction (LIME basis)",
        f"AED {_lime_result['model_pred']:.2f}",
        help="Same model prediction as seen by the LIME explainer \u2014 should match SHAP\u2019s prediction exactly.",
    )
    if _lime_result.get("local_pred") is not None:
        _pred_col3.metric(
            "LIME surrogate prediction",
            f"AED {_lime_result['local_pred']:.2f}",
            delta=f"{_lime_result['local_pred'] - _l_pred:+.2f} vs model",
            delta_color="off",
            help="What LIME\u2019s local linear model predicts. The gap vs the XGBoost prediction shows how well the surrogate approximates the true model at this point.",
        )

if _LIME_AVAILABLE and _lime_result.get("weights"):
    _shap_series = pd.Series(dict(zip(sample_features.columns, _l_contrib)), name="SHAP contribution (AED)")
    _shap_series = _shap_series.reindex(sample_features.columns).fillna(0.0)
    _lime_series = pd.Series(dict(_lime_result["weights"]), name="LIME weight (AED approx.)")

    # LIME returns condition strings like "route_distance_km > 28.44" or
    # "0.00 < demand_index <= 1.24". Extract the underlying column name by
    # finding which known column name appears in the condition string.
    _known_cols = list(sample_features.columns)

    def _extract_col(condition: str) -> str:
        for col in _known_cols:
            if col in condition:
                return col
        return condition  # fallback: no match found

    _lime_conditions = [f for f, _ in _lime_result["weights"]]
    _lime_col_names = [_extract_col(cond) for cond in _lime_conditions]

    _align_df = pd.DataFrame({
        "LIME condition": _lime_conditions,
        "Feature": _lime_col_names,
        "SHAP contribution (AED)": [round(float(_shap_series.get(col, 0.0)), 4) for col in _lime_col_names],
        "LIME weight (AED approx.)": [round(float(w), 4) for _, w in _lime_result["weights"]],
    })
    _align_df["SHAP direction"] = _align_df["SHAP contribution (AED)"].apply(lambda v: "\u2191 raises fare" if v > 0 else ("\u2193 lowers fare" if v < 0 else "neutral"))
    _align_df["LIME direction"] = _align_df["LIME weight (AED approx.)"].apply(lambda v: "\u2191 raises fare" if v > 0 else ("\u2193 lowers fare" if v < 0 else "neutral"))
    _align_df["Agree?"] = (_align_df["SHAP direction"] == _align_df["LIME direction"]).map({True: "\u2705 Yes", False: "\u274c No"})

    _agree_pct = (_align_df["Agree?"] == "\u2705 Yes").mean() * 100
    st.markdown(
        f"**Direction agreement across {len(_align_df)} features: {_agree_pct:.0f}%** \u2014 "
        "when SHAP and LIME agree on direction (both push the fare up or both push it down), the feature\u2019s influence is considered robust."
    )
    st.dataframe(_align_df, width="stretch", hide_index=True)

    try:
        from scipy.stats import spearmanr as _spearmanr
        _shap_vals = _align_df["SHAP contribution (AED)"].values
        _lime_vals = _align_df["LIME weight (AED approx.)"].values
        _rho, _pval = _spearmanr(_shap_vals, _lime_vals)
        st.caption(
            f"Spearman rank correlation between SHAP and LIME weights: **\u03c1 = {_rho:.3f}** (p = {_pval:.3f}). "
            "A high \u03c1 (> 0.7) means both methods agree on feature importance ordering even if magnitudes differ."
        )
    except Exception:
        pass

st.caption(
    "\u26a0\ufe0f **SHAP vs LIME:** Shapley values satisfy four desirable axioms (efficiency, symmetry, dummy, additivity). "
    "LIME does not guarantee consistency across runs \u2014 the same feature can receive different weights on repeated calls. "
    "For tree ensembles, XGBoost\u2019s native `pred_contribs=True` is both exact and orders of magnitude faster "
    "than LIME\u2019s perturbation approach (Salih et al., 2025)."
)

st.divider()
section_header("What-if lab")

today = datetime.now()
default_day = min(today.day, monthrange(today.year, today.month)[1])
default_date = date(today.year, today.month, default_day)

card(
    "How to use the lab",
    "Set pickup and dropoff coordinates, choose a product and time, then drag the override sliders to explore how distance, traffic, and demand change the predicted fare. The waterfall below shows exactly how each factor contributes.",
)

control_columns = st.columns(4)
pickup_lat = control_columns[0].number_input("Pickup latitude", value=float(DEFAULT_PICKUP_POINT[0]), format="%.6f", step=0.000001, key="whatif_pickup_lat")
pickup_lon = control_columns[1].number_input("Pickup longitude", value=float(DEFAULT_PICKUP_POINT[1]), format="%.6f", step=0.000001, key="whatif_pickup_lon")
dropoff_lat = control_columns[2].number_input("Dropoff latitude", value=float(DEFAULT_DROPOFF_POINT[0]), format="%.6f", step=0.000001, key="whatif_dropoff_lat")
dropoff_lon = control_columns[3].number_input("Dropoff longitude", value=float(DEFAULT_DROPOFF_POINT[1]), format="%.6f", step=0.000001, key="whatif_dropoff_lon")

control_columns_1b = st.columns(2)
product_type = control_columns_1b[0].selectbox("Product", PRODUCT_NAMES, index=1, key="whatif_product")
payment_method = control_columns_1b[1].selectbox("Payment", PAYMENT_METHODS, index=0, key="whatif_payment")

control_columns_2 = st.columns(3)
ride_date = control_columns_2[0].date_input("Ride date", value=default_date, key="whatif_date")
ride_time = control_columns_2[1].time_input("Ride time", value=time(8, 30), key="whatif_time")
use_live_weather = control_columns_2[2].toggle("Use live weather", value=False, key="whatif_weather_toggle")

control_columns_3 = st.columns(2)
use_live_traffic = control_columns_3[0].toggle("Use live traffic", value=False, key="whatif_traffic_toggle")
control_columns_3[1].caption("Live traffic uses TomTom when configured and only for today's date; otherwise the app falls back to the synthetic traffic model.")

_DUBAI_LAT = (24.6, 25.4)
_DUBAI_LON = (54.9, 56.0)
for _name, _lat, _lon in [("Pickup", pickup_lat, pickup_lon), ("Dropoff", dropoff_lat, dropoff_lon)]:
    if not (_DUBAI_LAT[0] <= _lat <= _DUBAI_LAT[1] and _DUBAI_LON[0] <= _lon <= _DUBAI_LON[1]):
        st.error(
            f"{_name} coordinates ({_lat:.4f}, {_lon:.4f}) are outside the Dubai/Sharjah service area. "
            f"Valid range: lat {_DUBAI_LAT[0]}–{_DUBAI_LAT[1]}, lon {_DUBAI_LON[0]}–{_DUBAI_LON[1]}."
        )
        st.stop()

scenario_dt = datetime.combine(ride_date, ride_time)
weather = get_weather(pickup_lat, pickup_lon, scenario_dt, prefer_live=use_live_weather)
route_context = get_route_context(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, scenario_dt, prefer_live_traffic=use_live_traffic)
scenario_record = build_trip_record(
    pickup_lat,
    pickup_lon,
    dropoff_lat,
    dropoff_lon,
    product_type,
    scenario_dt,
    weather,
    payment_method,
    route_context=route_context,
)

st.markdown("<br>", unsafe_allow_html=True)
section_header("Manual overrides")
override_columns = st.columns(3)
distance_override = override_columns[0].slider("Distance override (km)", min_value=1.0, max_value=65.0, value=round(float(scenario_record["route_distance_km"]), 1), step=0.1)
traffic_override = override_columns[1].slider("Traffic index override", min_value=0.65, max_value=2.20, value=round(float(scenario_record["traffic_index"]), 2), step=0.01)
demand_override_col3 = override_columns[2].slider("Demand index override", min_value=0.75, max_value=3.0, value=round(float(scenario_record["demand_index"]), 2), step=0.01)

scenario_record["route_distance_km"] = round(distance_override, 2)
scenario_record["traffic_index"] = round(traffic_override, 3)
scenario_record["route_efficiency_ratio"] = round(distance_override / max(float(scenario_record["route_direct_distance_km"]), 0.5), 3)
scenario_record["avg_speed_kmh"] = round(max(12.0, min(78.0, 52.0 / max(traffic_override, 0.65))), 1)
scenario_record["trip_duration_min"] = round(distance_override / max(float(scenario_record["avg_speed_kmh"]), 1.0) * 60, 1)
scenario_record["demand_index"] = round(demand_override_col3, 3)
scenario_record["captain_availability_score"] = round(max(0.15, min(1.0, 1.0 - 0.32 * (demand_override_col3 - 1.0))), 3)
scenario_record["supply_pressure_index"] = round(1.0 - scenario_record["captain_availability_score"], 3)
scenario_record["wait_time_min"] = round(max(1.0, scenario_record["wait_time_min"] * (1.05 + scenario_record["supply_pressure_index"] * 0.25)), 1)

feature_frame, _ = build_inference_frame(scenario_record, feature_columns)
contribution_series, base_value, predicted_price = compute_local_contributions(model, feature_frame)
explanation = build_explanation(scenario_record, contribution_series, predicted_price, base_value)

st.markdown("<br>", unsafe_allow_html=True)
lab_left, lab_right = st.columns([1.0, 1.0], gap="large")
with lab_left:
    _scenario_interval = estimate_trip_interval(scenario_record, metrics)
    _wi_low  = max(0.0, predicted_price - _scenario_interval["half_width"])
    _wi_high = predicted_price + _scenario_interval["half_width"]
    whatif_result(
        predicted_price,
        float(scenario_record["final_price_aed"]),
        low_aed=_wi_low,
        high_aed=_wi_high,
        interval_label=f"{_scenario_interval['basis_percent']}% range",
    )
    if _scenario_interval["is_adaptive"]:
        st.caption(
            f"Adaptive {_scenario_interval['basis_percent']}% band: {_scenario_interval['label']} scenario ±AED {_scenario_interval['half_width']:.2f} "
            f"(global baseline ±AED {_scenario_interval['global_half_width']:.2f})."
        )
    metric_row = st.columns(3)
    metric_row[0].metric("Route source", scenario_record.get("route_source", "Synthetic"))
    metric_row[1].metric("Traffic source", scenario_record.get("traffic_source", "Synthetic"))
    metric_row[2].metric("Traffic condition", scenario_record.get("traffic_condition", "Moderate"))
    section_header("Contribution waterfall")
    st.pyplot(plot_waterfall(contribution_series, base_value, predicted_price), width="stretch")

with lab_right:
    section_header("Scenario explanation")
    st.caption(
        f"Pickup ({pickup_lat:.4f}, {pickup_lon:.4f}) in **{scenario_record['pickup_zone']}** → Dropoff ({dropoff_lat:.4f}, {dropoff_lon:.4f}) in **{scenario_record['dropoff_zone']}**"
    )
    st.markdown(f"**{explanation['headline']}**")
    for sentence in explanation["sentences"]:
        st.write(sentence)
    section_header("Driver breakdown")
    st.dataframe(explanation["summary_table"].head(8), width="stretch", hide_index=True)