from __future__ import annotations

import sys
from calendar import monthrange
from datetime import date, datetime, time
from pathlib import Path

import streamlit as st
from streamlit_folium import st_folium


APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
for path in (APP_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils.domain import DEFAULT_DROPOFF_POINT, DEFAULT_PICKUP_POINT, PAYMENT_METHODS, PRODUCT_NAMES, build_inference_frame, build_trip_record
from utils.geo_utils import build_picker_map, get_nearest_zone
from utils.model_loader import load_feature_columns, load_model
from utils.nlp_explainer import build_explanation
from utils.routing_api import get_route_context
from utils.shap_engine import compute_local_contributions, plot_waterfall
from utils.ui import apply_theme, card, hero
from utils.weather_api import get_weather


st.set_page_config(page_title="XPrice Rider Simulator", layout="wide")
apply_theme()

today = datetime.now()
default_day = min(today.day, monthrange(2025, today.month)[1])
default_date = date(2025, today.month, default_day)

hero(
    "Ride Simulator",
    "Rider-facing quote",
    "Click directly on the Dubai map to set pickup and dropoff points. The app then pulls a route preview, optional live traffic, weather context, and an explainable fare quote for that exact coordinate pair.",
)

model = load_model()
feature_columns = load_feature_columns()

if "pickup_point" not in st.session_state:
    st.session_state["pickup_point"] = DEFAULT_PICKUP_POINT
if "dropoff_point" not in st.session_state:
    st.session_state["dropoff_point"] = DEFAULT_DROPOFF_POINT
if "map_click_signature" not in st.session_state:
    st.session_state["map_click_signature"] = None

with st.sidebar:
    st.markdown("### Build a ride")
    click_mode = st.radio("Map click mode", ["Set pickup", "Set dropoff"], horizontal=True)
    if st.button("Reset route points", use_container_width=True):
        st.session_state["pickup_point"] = DEFAULT_PICKUP_POINT
        st.session_state["dropoff_point"] = DEFAULT_DROPOFF_POINT
        st.session_state["map_click_signature"] = None
        st.rerun()

    product_type = st.selectbox("Product", PRODUCT_NAMES, index=0)
    payment_method = st.selectbox("Payment method", PAYMENT_METHODS, index=0)
    ride_date = st.date_input("Ride date", value=default_date)
    ride_time = st.time_input("Ride time", value=time(18, 30))
    use_live_weather = st.toggle("Use live weather when available", value=True)
    use_live_traffic = st.toggle("Use live traffic when available", value=True)

    st.markdown("### Fine tune coordinates")
    pickup_lat = st.number_input("Pickup latitude", value=float(st.session_state["pickup_point"][0]), format="%.6f", step=0.000001)
    pickup_lon = st.number_input("Pickup longitude", value=float(st.session_state["pickup_point"][1]), format="%.6f", step=0.000001)
    dropoff_lat = st.number_input("Dropoff latitude", value=float(st.session_state["dropoff_point"][0]), format="%.6f", step=0.000001)
    dropoff_lon = st.number_input("Dropoff longitude", value=float(st.session_state["dropoff_point"][1]), format="%.6f", step=0.000001)

st.session_state["pickup_point"] = (pickup_lat, pickup_lon)
st.session_state["dropoff_point"] = (dropoff_lat, dropoff_lon)

ride_dt = datetime.combine(ride_date, ride_time)
pickup_point = st.session_state["pickup_point"]
dropoff_point = st.session_state["dropoff_point"]
route_context = get_route_context(
    pickup_point[0],
    pickup_point[1],
    dropoff_point[0],
    dropoff_point[1],
    ride_dt,
    prefer_live_traffic=use_live_traffic,
)
weather = get_weather(pickup_point[0], pickup_point[1], ride_dt, prefer_live=use_live_weather)

map_payload = st_folium(
    build_picker_map(pickup_point, dropoff_point, route_context.get("route_geometry")),
    height=460,
    returned_objects=["last_clicked"],
)
clicked = map_payload.get("last_clicked") if map_payload else None
if clicked:
    click_signature = (round(clicked["lat"], 6), round(clicked["lng"], 6), click_mode)
    if click_signature != st.session_state.get("map_click_signature"):
        if click_mode == "Set pickup":
            st.session_state["pickup_point"] = (clicked["lat"], clicked["lng"])
        else:
            st.session_state["dropoff_point"] = (clicked["lat"], clicked["lng"])
        st.session_state["map_click_signature"] = click_signature
        st.rerun()

record = build_trip_record(
    pickup_lat=pickup_point[0],
    pickup_lon=pickup_point[1],
    dropoff_lat=dropoff_point[0],
    dropoff_lon=dropoff_point[1],
    product_type=product_type,
    ride_dt=ride_dt,
    weather=weather,
    payment_method=payment_method,
    route_context=route_context,
)
feature_frame, _ = build_inference_frame(record, feature_columns)
contribution_series, base_value, predicted_price = compute_local_contributions(model, feature_frame)
explanation = build_explanation(record, contribution_series, predicted_price, base_value)
anchor_delta = predicted_price - float(record["final_price_aed"])

top_left, top_right = st.columns([1.05, 0.95], gap="large")

with top_left:
    st.subheader("Route lens")
    details = st.columns(4)
    details[0].metric("Route distance", f"{record['route_distance_km']:.1f} km")
    details[1].metric("Direct distance", f"{record['route_direct_distance_km']:.1f} km")
    details[2].metric("Traffic index", f"{record['traffic_index']:.2f}")
    details[3].metric("Demand index", f"{record['demand_index']:.2f}")
    card(
        "Derived geography",
        f"Pickup zone: {record['pickup_zone']} ({pickup_point[0]:.4f}, {pickup_point[1]:.4f}). Dropoff zone: {record['dropoff_zone']} ({dropoff_point[0]:.4f}, {dropoff_point[1]:.4f}). Route source: {record['route_source']}.",
    )

with top_right:
    st.subheader("Fare quote")
    price_columns = st.columns(3)
    price_columns[0].metric("Predicted fare", f"AED {predicted_price:,.2f}")
    price_columns[1].metric("Engine anchor", f"AED {record['final_price_aed']:,.2f}")
    price_columns[2].metric("Model vs engine", f"AED {anchor_delta:+.2f}")

    card(
        "Context snapshot",
        f"Weather: {record['weather_label']} ({record['weather_source']}). Traffic: {record['traffic_condition']} ({record['traffic_source']}). Event: {record['active_event']}. Availability score: {record['captain_availability_score']:.2f}. Selected product: {record['product_type']}.",
    )
    st.markdown(f"**{explanation['headline']}**")
    for sentence in explanation["sentences"]:
        st.write(sentence)
    st.caption(
        f"Map-derived zones: pickup {get_nearest_zone(*pickup_point)}, dropoff {get_nearest_zone(*dropoff_point)}. Traffic source falls back to a synthetic model when no live provider is configured for the selected date."
    )

bottom_left, bottom_right = st.columns([1.15, 0.85], gap="large")
with bottom_left:
    st.subheader("Local contribution waterfall")
    st.pyplot(plot_waterfall(contribution_series, base_value, predicted_price), width="stretch")

with bottom_right:
    st.subheader("Top local drivers")
    st.dataframe(explanation["summary_table"].head(8), width="stretch", hide_index=True)
    st.caption("The pricing engine and the model are intentionally close because the synthetic mirror dataset is generated from explicit pricing logic and then learned back by the model. The difference is that the quote is now anchored to map-selected coordinates and route context.")