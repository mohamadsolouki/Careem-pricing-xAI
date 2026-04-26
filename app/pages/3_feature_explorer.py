from __future__ import annotations

import sys
from calendar import monthrange
from datetime import date, datetime, time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
for path in (APP_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils.domain import DEFAULT_DROPOFF_POINT, DEFAULT_PICKUP_POINT, PAYMENT_METHODS, PRODUCT_NAMES, build_inference_frame, build_trip_record
from utils.model_loader import load_feature_columns, load_model, load_shap_bundle
from utils.nlp_explainer import build_explanation
from utils.routing_api import get_route_context
from utils.shap_engine import compute_local_contributions, plot_dependence, plot_waterfall
from utils.ui import apply_theme, card, hero, section_header, whatif_result
from utils.weather_api import get_weather


st.set_page_config(page_title="XPrice Feature Explorer", layout="wide")
apply_theme()

hero(
    "Feature Explorer",
    "Analyst workbench",
    "Explore how any single driver influences fares across the full sample. Inspect its SHAP contribution shape, plot the marginal response curve, then run a what-if scenario with custom coordinates and manual overrides for distance, traffic, and supply.",
)

model = load_model()
feature_columns = load_feature_columns()
bundle = load_shap_bundle()
sample_features = bundle["sample_features"].copy()
contrib_values = bundle["values"]

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
    st.pyplot(plot_dependence(contrib_values, sample_features, selected_feature), use_container_width=True)

with right:
    section_header("Partial dependence (marginal response)")
    baseline_row = sample_features.iloc[[0]].copy()
    low = float(sample_features[selected_feature].quantile(0.05))
    high = float(sample_features[selected_feature].quantile(0.95))
    sweep_values = np.linspace(low, high, 30)
    predictions = []
    for value in sweep_values:
        scenario = baseline_row.copy()
        scenario[selected_feature] = value
        predictions.append(float(model.predict(scenario)[0]))
    sweep_frame = pd.DataFrame({"feature_value": sweep_values, "predicted_price_aed": predictions})
    sweep_chart = px.line(
        sweep_frame,
        x="feature_value",
        y="predicted_price_aed",
        markers=True,
        title=f"Predicted fare response to {selected_feature}",
        template="plotly_white",
    )
    sweep_chart.update_traces(
        line={"color": "#0d9488", "width": 2.5},
        marker={"color": "#0d9488", "size": 6},
    )
    sweep_chart.update_layout(
        height=360,
        margin={"l": 0, "r": 0, "t": 44, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#0f172a", "family": "Inter, Segoe UI, sans-serif"},
        title_font={"size": 14, "color": "#0f172a"},
        xaxis={"gridcolor": "#f1f5f9", "title": {"text": selected_feature, "font": {"color": "#64748b"}}},
        yaxis={"gridcolor": "#f1f5f9", "title": {"text": "Predicted fare (AED)", "font": {"color": "#64748b"}}},
    )
    st.plotly_chart(sweep_chart, use_container_width=True)

st.divider()
section_header("What-if lab")

today = datetime.now()
default_day = min(today.day, monthrange(2025, today.month)[1])
default_date = date(2025, today.month, default_day)

card(
    "How to use the lab",
    "Set pickup and dropoff coordinates, choose a product and time, then drag the override sliders to explore how distance, traffic, and captain availability change the predicted fare. The waterfall below shows exactly how each factor contributes.",
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
availability_override = override_columns[2].slider("Availability override", min_value=0.15, max_value=1.0, value=round(float(scenario_record["captain_availability_score"]), 2), step=0.01)

demand_override = st.slider("Demand index override", min_value=0.75, max_value=3.0, value=round(float(scenario_record["demand_index"]), 2), step=0.01)

scenario_record["route_distance_km"] = round(distance_override, 2)
scenario_record["traffic_index"] = round(traffic_override, 3)
scenario_record["route_efficiency_ratio"] = round(distance_override / max(float(scenario_record["route_direct_distance_km"]), 0.5), 3)
scenario_record["avg_speed_kmh"] = round(max(12.0, min(78.0, 52.0 / max(traffic_override, 0.65))), 1)
scenario_record["trip_duration_min"] = round(distance_override / max(float(scenario_record["avg_speed_kmh"]), 1.0) * 60, 1)
scenario_record["demand_index"] = round(demand_override, 3)
scenario_record["captain_availability_score"] = round(availability_override, 3)
scenario_record["supply_pressure_index"] = round(1.0 - availability_override, 3)
scenario_record["wait_time_min"] = round(max(1.0, scenario_record["wait_time_min"] * (1.05 + scenario_record["supply_pressure_index"] * 0.25)), 1)

feature_frame, _ = build_inference_frame(scenario_record, feature_columns)
contribution_series, base_value, predicted_price = compute_local_contributions(model, feature_frame)
explanation = build_explanation(scenario_record, contribution_series, predicted_price, base_value)

st.markdown("<br>", unsafe_allow_html=True)
lab_left, lab_right = st.columns([1.0, 1.0], gap="large")
with lab_left:
    whatif_result(predicted_price, float(scenario_record["final_price_aed"]))
    metric_row = st.columns(3)
    metric_row[0].metric("Route source", scenario_record.get("route_source", "Synthetic"))
    metric_row[1].metric("Traffic source", scenario_record.get("traffic_source", "Synthetic"))
    metric_row[2].metric("Traffic condition", scenario_record.get("traffic_condition", "Moderate"))
    section_header("Contribution waterfall")
    st.pyplot(plot_waterfall(contribution_series, base_value, predicted_price), use_container_width=True)

with lab_right:
    section_header("Scenario explanation")
    st.caption(
        f"Pickup ({pickup_lat:.4f}, {pickup_lon:.4f}) in **{scenario_record['pickup_zone']}** → Dropoff ({dropoff_lat:.4f}, {dropoff_lon:.4f}) in **{scenario_record['dropoff_zone']}**"
    )
    st.markdown(f"**{explanation['headline']}**")
    for sentence in explanation["sentences"]:
        st.write(sentence)
    section_header("Driver breakdown")
    st.dataframe(explanation["summary_table"].head(8), use_container_width=True, hide_index=True)