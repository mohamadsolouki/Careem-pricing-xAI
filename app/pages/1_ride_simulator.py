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
from utils.model_loader import load_feature_columns, load_metrics, load_model, load_model_version
from utils.nlp_explainer import build_explanation
from utils.routing_api import get_route_context
from utils.shap_engine import compute_local_contributions, plot_waterfall
from utils.ui import apply_theme, card, fare_result, hero, section_header
from utils.weather_api import get_weather


st.set_page_config(page_title="XPrice Rider Simulator", layout="wide")
apply_theme()


def _apply_page_styles() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255,255,255,0.96) !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 20px !important;
            box-shadow: 0 10px 30px rgba(15,23,42,0.05) !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] > div {
            padding: 0.3rem 0.45rem !important;
        }
        .ride-dash-caption {
            margin-top: -0.2rem;
            margin-bottom: 0.9rem;
            color: #64748b;
            font-size: 0.93rem;
            line-height: 1.65;
        }
        .ride-inline-note {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-bottom: 0.9rem;
        }
        .ride-inline-note span {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 999px;
            padding: 0.34rem 0.8rem;
            font-size: 0.76rem;
            font-weight: 700;
            color: #0f766e;
            letter-spacing: 0.02em;
        }
        .ride-bullet-list {
            margin: 0.45rem 0 0 0;
            padding-left: 1.1rem;
            color: #475569;
        }
        .ride-bullet-list li {
            margin-bottom: 0.45rem;
            line-height: 1.55;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _sync_coordinate_inputs() -> None:
    st.session_state["pickup_lat_input"] = float(st.session_state["pickup_point"][0])
    st.session_state["pickup_lon_input"] = float(st.session_state["pickup_point"][1])
    st.session_state["dropoff_lat_input"] = float(st.session_state["dropoff_point"][0])
    st.session_state["dropoff_lon_input"] = float(st.session_state["dropoff_point"][1])


def _location_card(title: str, zone: str, point: tuple[float, float]) -> str:
    return f"""
    <div class="xprice-card">
        <h3>{title}</h3>
        <div class="xprice-stat-strip">
            <div class="xprice-stat-item">
                <div class="stat-label">Zone</div>
                <div class="stat-value">{zone}</div>
            </div>
            <div class="xprice-stat-item">
                <div class="stat-label">Latitude</div>
                <div class="stat-value">{point[0]:.4f}</div>
            </div>
            <div class="xprice-stat-item">
                <div class="stat-label">Longitude</div>
                <div class="stat-value">{point[1]:.4f}</div>
            </div>
        </div>
    </div>
    """


def _builder_snapshot(ride_dt: datetime, pickup_point: tuple[float, float], dropoff_point: tuple[float, float], use_live_weather: bool, use_live_traffic: bool) -> str:
    return f"""
    <div class="xprice-stat-strip">
        <div class="xprice-stat-item">
            <div class="stat-label">Pickup zone</div>
            <div class="stat-value">{get_nearest_zone(*pickup_point)}</div>
        </div>
        <div class="xprice-stat-item">
            <div class="stat-label">Dropoff zone</div>
            <div class="stat-value">{get_nearest_zone(*dropoff_point)}</div>
        </div>
        <div class="xprice-stat-item">
            <div class="stat-label">Ride window</div>
            <div class="stat-value">{ride_dt.strftime('%d %b · %H:%M')}</div>
        </div>
        <div class="xprice-stat-item">
            <div class="stat-label">Route editing</div>
            <div class="stat-value">Drag pins</div>
        </div>
        <div class="xprice-stat-item">
            <div class="stat-label">Live signals</div>
            <div class="stat-value">{'Weather' if use_live_weather else 'Static'} / {'Traffic' if use_live_traffic else 'Static'}</div>
        </div>
    </div>
    """


_apply_page_styles()

today = datetime.now()
default_day = min(today.day, monthrange(today.year, today.month)[1])
default_date = date(today.year, today.month, default_day)

model = load_model()
feature_columns = load_feature_columns()
metrics = load_metrics()
_pi90_half_width = metrics.get("prediction_interval_90_half_width", 0.0)
version_sim = load_model_version()

if "pickup_point" not in st.session_state:
    st.session_state["pickup_point"] = DEFAULT_PICKUP_POINT
if "dropoff_point" not in st.session_state:
    st.session_state["dropoff_point"] = DEFAULT_DROPOFF_POINT
if "map_center" not in st.session_state:
    st.session_state["map_center"] = None
if "map_zoom" not in st.session_state:
    st.session_state["map_zoom"] = None
if "route_editor_signature" not in st.session_state:
    st.session_state["route_editor_signature"] = None
if any(
    key not in st.session_state
    for key in ("pickup_lat_input", "pickup_lon_input", "dropoff_lat_input", "dropoff_lon_input")
):
    _sync_coordinate_inputs()

previous_pickup = st.session_state["pickup_point"]
previous_dropoff = st.session_state["dropoff_point"]
proposed_pickup = (float(st.session_state["pickup_lat_input"]), float(st.session_state["pickup_lon_input"]))
proposed_dropoff = (float(st.session_state["dropoff_lat_input"]), float(st.session_state["dropoff_lon_input"]))
manual_route_change = proposed_pickup != previous_pickup or proposed_dropoff != previous_dropoff
st.session_state["pickup_point"] = proposed_pickup
st.session_state["dropoff_point"] = proposed_dropoff
if manual_route_change:
    st.session_state["map_center"] = None
    st.session_state["map_zoom"] = None
    st.session_state["route_editor_signature"] = None

hero(
    "Ride Simulator",
    "Rider-facing quote",
    "Keep the route, timing, and quote in one view. Drag the pickup and dropoff pins directly on the Dubai map, tune trip settings above the fold, and read the fare explanation without the quote falling below the canvas.",
)

with st.container(border=True):
    st.markdown("#### Trip builder")
    st.markdown(
        '<div class="ride-dash-caption">All primary filters now live on the page so you can adjust the route, ride timing, and quote inputs without bouncing between a sidebar and the map.</div>',
        unsafe_allow_html=True,
    )
    builder_col_1, builder_col_2, builder_col_3, builder_col_4, builder_col_5 = st.columns([1.05, 1.05, 0.95, 0.95, 1.2], gap="medium")
    with builder_col_1:
        product_type = st.selectbox("Product", PRODUCT_NAMES, index=0)
    with builder_col_2:
        payment_method = st.selectbox("Payment method", PAYMENT_METHODS, index=0)
    with builder_col_3:
        ride_date = st.date_input("Ride date", value=default_date)
    with builder_col_4:
        ride_time = st.time_input("Ride time", value=time(18, 30))
    with builder_col_5:
        use_live_weather = st.toggle("Use live weather", value=True)
        use_live_traffic = st.toggle("Use live traffic", value=True)
        if st.button("Reset route", use_container_width=True):
            st.session_state["pickup_point"] = DEFAULT_PICKUP_POINT
            st.session_state["dropoff_point"] = DEFAULT_DROPOFF_POINT
            st.session_state["map_center"] = None
            st.session_state["map_zoom"] = None
            st.session_state["route_editor_signature"] = None
            _sync_coordinate_inputs()
            st.rerun()

    ride_dt = datetime.combine(ride_date, ride_time)
    st.markdown(
        _builder_snapshot(
            ride_dt,
            st.session_state["pickup_point"],
            st.session_state["dropoff_point"],
            use_live_weather,
            use_live_traffic,
        ),
        unsafe_allow_html=True,
    )

# ── Coordinate bounds check (Dubai / Sharjah bounding box) ───────────────────
_DUBAI_LAT = (24.6, 25.4)
_DUBAI_LON = (54.9, 56.0)

pickup_point = st.session_state["pickup_point"]
dropoff_point = st.session_state["dropoff_point"]
pickup_lat, pickup_lon = pickup_point
dropoff_lat, dropoff_lon = dropoff_point

def _out_of_bounds(lat: float, lon: float) -> bool:
    return not (_DUBAI_LAT[0] <= lat <= _DUBAI_LAT[1] and _DUBAI_LON[0] <= lon <= _DUBAI_LON[1])

if _out_of_bounds(pickup_lat, pickup_lon):
    st.error(
        f"Pickup coordinates ({pickup_lat:.4f}, {pickup_lon:.4f}) are outside the Dubai/Sharjah "
        f"service area (lat {_DUBAI_LAT[0]}–{_DUBAI_LAT[1]}, lon {_DUBAI_LON[0]}–{_DUBAI_LON[1]}). "
        "Please reposition the pin on the map."
    )
    st.stop()
if _out_of_bounds(dropoff_lat, dropoff_lon):
    st.error(
        f"Dropoff coordinates ({dropoff_lat:.4f}, {dropoff_lon:.4f}) are outside the Dubai/Sharjah "
        f"service area (lat {_DUBAI_LAT[0]}–{_DUBAI_LAT[1]}, lon {_DUBAI_LON[0]}–{_DUBAI_LON[1]}). "
        "Please reposition the pin on the map."
    )
    st.stop()

route_context = get_route_context(
    pickup_point[0],
    pickup_point[1],
    dropoff_point[0],
    dropoff_point[1],
    ride_dt,
    prefer_live_traffic=use_live_traffic,
)
weather = get_weather(pickup_point[0], pickup_point[1], ride_dt, prefer_live=use_live_weather)

top_left, top_right = st.columns([1.28, 0.92], gap="large")

with top_left:
    with st.container(border=True):
        st.markdown("#### Route editor")
        st.markdown(
            '<div class="ride-dash-caption">Drag the green pickup chip and the red dropoff chip directly on the map. The map still pans normally, but marker drags now update the route instead of forcing you to juggle sidebar modes.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="ride-inline-note"><span>Drag pins to reposition</span><span>Pan and zoom freely</span><span>Manual lat/lon edit is still available below</span></div>',
            unsafe_allow_html=True,
        )
        map_payload = st_folium(
            build_picker_map(
                pickup_point,
                dropoff_point,
                route_context.get("route_geometry"),
                center=st.session_state.get("map_center"),
                zoom=st.session_state.get("map_zoom"),
            ),
            key="ride_simulator_map",
            height=560,
            returned_objects=[],
            use_container_width=True,
        )
        edited_feature = map_payload.get("last_active_drawing") if map_payload else None
        if edited_feature and edited_feature.get("geometry", {}).get("type") == "Point":
            edited_role = edited_feature.get("properties", {}).get("role")
            geometry = edited_feature.get("geometry", {})
            coordinates = geometry.get("coordinates", [])
            if edited_role in {"pickup", "dropoff"} and len(coordinates) == 2:
                edited_point = (float(coordinates[1]), float(coordinates[0]))
                signature = f"{edited_role}:{edited_point[0]:.6f}:{edited_point[1]:.6f}"
            else:
                signature = None
                edited_point = None

            if signature and edited_point and signature != st.session_state.get("route_editor_signature"):
                if edited_role == "pickup":
                    st.session_state["pickup_point"] = edited_point
                else:
                    st.session_state["dropoff_point"] = edited_point
                center_payload = map_payload.get("center") if map_payload else None
                if center_payload:
                    st.session_state["map_center"] = (float(center_payload["lat"]), float(center_payload["lng"]))
                zoom_payload = map_payload.get("zoom") if map_payload else None
                if zoom_payload is not None:
                    st.session_state["map_zoom"] = int(zoom_payload)
                st.session_state["route_editor_signature"] = signature
                _sync_coordinate_inputs()
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

        pickup_summary, dropoff_summary = st.columns(2, gap="medium")
        with pickup_summary:
            st.markdown(_location_card("Pickup pin", record["pickup_zone"], pickup_point), unsafe_allow_html=True)
        with dropoff_summary:
            st.markdown(_location_card("Dropoff pin", record["dropoff_zone"], dropoff_point), unsafe_allow_html=True)

        with st.expander("Fine-tune coordinates", expanded=False):
            coord_left, coord_right = st.columns(2, gap="medium")
            with coord_left:
                st.number_input("Pickup latitude", key="pickup_lat_input", format="%.6f", step=0.000001)
                st.number_input("Pickup longitude", key="pickup_lon_input", format="%.6f", step=0.000001)
            with coord_right:
                st.number_input("Dropoff latitude", key="dropoff_lat_input", format="%.6f", step=0.000001)
                st.number_input("Dropoff longitude", key="dropoff_lon_input", format="%.6f", step=0.000001)

    with st.container(border=True):
        section_header("Route lens")
        details = st.columns(4)
        details[0].metric("Route distance", f"{record['route_distance_km']:.1f} km")
        details[1].metric("Direct distance", f"{record['route_direct_distance_km']:.1f} km")
        details[2].metric("Traffic index", f"{record['traffic_index']:.2f}")
        details[3].metric("Demand index", f"{record['demand_index']:.2f}")
        card(
            "Derived geography",
            f"<strong>Pickup zone:</strong> {record['pickup_zone']} ({pickup_point[0]:.4f}, {pickup_point[1]:.4f}) &nbsp;→&nbsp; <strong>Dropoff zone:</strong> {record['dropoff_zone']} ({dropoff_point[0]:.4f}, {dropoff_point[1]:.4f}). Route source: {record['route_source']}.",
        )

with top_right:
    with st.container(border=True):
        section_header("Fare quote")
        anchor_delta = predicted_price - float(record["final_price_aed"])
        _pi_low = max(0.0, predicted_price - _pi90_half_width)
        _pi_high = predicted_price + _pi90_half_width
        fare_result(
            f"{product_type} · {record['pickup_zone']} → {record['dropoff_zone']}",
            predicted_price,
            sub=f"Engine anchor AED {record['final_price_aed']:,.2f} · Model vs engine: {'+' if anchor_delta >= 0 else ''}{anchor_delta:.2f} AED",
            low_aed=_pi_low,
            high_aed=_pi_high,
        )
        quote_metrics_top = st.columns(2, gap="medium")
        quote_metrics_bottom = st.columns(2, gap="medium")
        quote_metrics_top[0].metric("Weather", record["weather_label"])
        quote_metrics_top[1].metric("Traffic", record["traffic_condition"])
        quote_metrics_bottom[0].metric("Demand", f"{record['demand_index']:.2f}")
        quote_metrics_bottom[1].metric("Availability", f"{record['captain_availability_score']:.2f}")

        card(
            "Context snapshot",
            f"<strong>Weather:</strong> {record['weather_label']} ({record['weather_source']}) &nbsp;·&nbsp; <strong>Traffic:</strong> {record['traffic_condition']} ({record['traffic_source']}) &nbsp;·&nbsp; <strong>Event:</strong> {record['active_event']} &nbsp;·&nbsp; <strong>Demand index:</strong> {record['demand_index']:.2f} &nbsp;·&nbsp; <strong>Availability:</strong> {record['captain_availability_score']:.2f} <em>(informational — not a model input)</em>",
        )
        st.markdown(f"**{explanation['headline']}**")
        st.markdown("<ul class='ride-bullet-list'>" + "".join(f"<li>{sentence}</li>" for sentence in explanation["sentences"]) + "</ul>", unsafe_allow_html=True)
        st.caption(
            f"Map-derived zones: pickup {get_nearest_zone(*pickup_point)}, dropoff {get_nearest_zone(*dropoff_point)}. Traffic falls back to a synthetic model when no live provider is configured for the selected ride window."
        )

    with st.container(border=True):
        section_header("Model status")
        model_metrics = st.columns(3)
        model_metrics[0].metric("Test R²", f"{metrics['test']['r2']:.4f}")
        model_metrics[1].metric("RMSE", f"AED {metrics['test']['rmse']:.2f}")
        model_metrics[2].metric("90% PI", f"±AED {_pi90_half_width:.2f}")
        st.caption(f"CV R² {metrics['cv']['r2_mean']:.4f} ± {metrics['cv']['r2_std']:.4f}")
        if version_sim.get("training_date"):
            st.caption(f"Trained {version_sim['training_date'][:10]}")

bottom_left, bottom_right = st.columns([1.15, 0.85], gap="large")
with bottom_left:
    with st.container(border=True):
        section_header("Local contribution waterfall")
        st.pyplot(plot_waterfall(contribution_series, base_value, predicted_price), use_container_width=True)

with bottom_right:
    with st.container(border=True):
        section_header("Top local drivers")
        st.dataframe(explanation["summary_table"].head(8), use_container_width=True, hide_index=True)
        st.caption("The pricing engine and the model are intentionally close because the synthetic mirror dataset is generated from explicit pricing logic and then learned back by the model. The difference is that the quote is now anchored to map-selected coordinates and route context.")