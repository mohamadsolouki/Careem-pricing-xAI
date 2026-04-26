from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
for path in (APP_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils.model_loader import load_shap_bundle
from utils.shap_engine import build_top_driver_frame, plot_beeswarm, plot_dependence, plot_importance_bar
from utils.ui import apply_theme, hero, section_header, sidebar_brand


st.set_page_config(page_title="XPrice Operations Dashboard", layout="wide")
apply_theme()

hero(
    "Operations Contribution Dashboard",
    "Manager-facing XAI",
    "Filter the SHAP-style contribution sample by zone, product, month, and event context to see which inputs are driving fare movements across the city. Drill down by time, geography, or event to isolate causal patterns.",
)

bundle = load_shap_bundle()
sample_raw = bundle["sample_raw"].copy()
sample_features = bundle["sample_features"].copy()
contrib_values = bundle["values"]
sample_raw["active_event_display"] = sample_raw["active_event"].fillna("No active event").replace({"None": "No active event"}).astype(str)

with st.sidebar:
    sidebar_brand()
    st.markdown("### Filters")
    zone_choice = st.selectbox("Pickup zone", ["All"] + sorted(sample_raw["pickup_zone"].unique().tolist()))
    product_choice = st.selectbox("Product", ["All"] + sorted(sample_raw["product_type"].unique().tolist()))
    month_choice = st.selectbox("Month", ["All"] + sorted(sample_raw["month_name"].unique().tolist(), key=lambda item: pd.Timestamp(f"2025-{item}-01").month))
    event_choice = st.selectbox("Event", ["All"] + sorted(sample_raw["active_event_display"].unique().tolist()))

mask = pd.Series(True, index=sample_raw.index)
if zone_choice != "All":
    mask &= sample_raw["pickup_zone"] == zone_choice
if product_choice != "All":
    mask &= sample_raw["product_type"] == product_choice
if month_choice != "All":
    mask &= sample_raw["month_name"] == month_choice
if event_choice != "All":
    mask &= sample_raw["active_event_display"] == event_choice

if not mask.any():
    st.warning("The selected filter combination returns no sampled rides. Relax one of the filters.")
    st.stop()

filtered_raw = sample_raw.loc[mask].reset_index(drop=True)
filtered_features = sample_features.loc[mask].reset_index(drop=True)
filtered_values = contrib_values[mask.to_numpy()]
top_driver_frame = build_top_driver_frame(filtered_values, list(filtered_features.columns), max_display=12)

st.markdown("<br>", unsafe_allow_html=True)
summary_columns = st.columns(4, gap="small")
summary_columns[0].metric("Filtered rides", f"{len(filtered_raw):,}")
summary_columns[1].metric("Average fare", f"AED {filtered_raw['final_price_aed'].mean():.2f}")
summary_columns[2].metric("Average demand", f"{filtered_raw['demand_index'].mean():.2f}")
summary_columns[3].metric("Avg supply pressure", f"{filtered_raw['supply_pressure_index'].mean():.2f}")

st.markdown("<br>", unsafe_allow_html=True)
tab_global, tab_zone, tab_time, tab_event = st.tabs(["Global view", "Zone lens", "Time lens", "Event lens"])

with tab_global:
    left, right = st.columns(2, gap="large")
    with left:
        section_header("Feature impact distribution")
        st.pyplot(plot_beeswarm(filtered_values, filtered_features, max_display=18), use_container_width=True)
    with right:
        section_header("Mean absolute contribution")
        st.pyplot(plot_importance_bar(filtered_values, list(filtered_features.columns), max_display=18), use_container_width=True)
    section_header("Top drivers summary")
    st.dataframe(top_driver_frame, use_container_width=True, hide_index=True)

with tab_zone:
    zone_summary = (
        filtered_raw.groupby("pickup_zone")
        .agg(rides=("final_price_aed", "size"), avg_fare=("final_price_aed", "mean"), avg_demand=("demand_index", "mean"), avg_supply_pressure=("supply_pressure_index", "mean"))
        .reset_index()
        .sort_values("avg_fare", ascending=False)
    )
    section_header("Average fare by pickup zone")
    st.plotly_chart(
        px.bar(
            zone_summary,
            x="pickup_zone",
            y="avg_fare",
            color="avg_demand",
            text="rides",
            color_continuous_scale="Teal",
            title="Average fare by pickup zone",
            template="plotly_white",
        ).update_layout(
            height=380,
            margin={"l": 0, "r": 0, "t": 44, "b": 0},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#0f172a", "family": "Inter, Segoe UI, sans-serif"},
        ),
        use_container_width=True,
    )

    heat_features = top_driver_frame["feature"].head(6).tolist()
    heat_rows = []
    for zone in zone_summary["pickup_zone"]:
        zone_mask = filtered_raw["pickup_zone"] == zone
        zone_values = filtered_values[zone_mask.to_numpy()]
        zone_frame = pd.DataFrame(zone_values, columns=filtered_features.columns)
        for feature in heat_features:
            heat_rows.append({"pickup_zone": zone, "feature": feature, "mean_contribution": zone_frame[feature].mean()})
    heatmap_frame = pd.DataFrame(heat_rows)
    heatmap_pivot = heatmap_frame.pivot(index="pickup_zone", columns="feature", values="mean_contribution")
    section_header("Contribution heatmap by zone")
    st.plotly_chart(
        px.imshow(
            heatmap_pivot,
            color_continuous_scale="RdBu",
            aspect="auto",
            title="Mean contribution by zone — top filtered drivers",
            template="plotly_white",
        ).update_layout(
            height=420,
            margin={"l": 0, "r": 0, "t": 44, "b": 0},
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#0f172a", "family": "Inter, Segoe UI, sans-serif"},
        ),
        use_container_width=True,
    )

with tab_time:
    hourly_contrib = pd.DataFrame(filtered_values, columns=filtered_features.columns)
    hourly_contrib["hour"] = filtered_raw["hour"].values
    time_features = top_driver_frame["feature"].head(5).tolist()
    hourly_pivot = hourly_contrib.groupby("hour")[time_features].mean().T
    section_header("Hourly contribution heatmap")
    st.plotly_chart(
        px.imshow(
            hourly_pivot,
            color_continuous_scale="RdBu",
            aspect="auto",
            title="Hourly contribution pattern — strongest filtered drivers",
            template="plotly_white",
        ).update_layout(
            height=380,
            margin={"l": 0, "r": 0, "t": 44, "b": 0},
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#0f172a", "family": "Inter, Segoe UI, sans-serif"},
        ),
        use_container_width=True,
    )
    section_header("Dependence plot")
    selected_feature = st.selectbox("Inspect one time-sensitive feature", time_features)
    st.pyplot(plot_dependence(filtered_values, filtered_features, selected_feature), use_container_width=True)

with tab_event:
    event_frame = filtered_raw.copy()
    event_frame["event_group"] = event_frame["active_event_display"]
    top_events = event_frame["event_group"].value_counts().head(8).index.tolist()
    event_subset = event_frame[event_frame["event_group"].isin(top_events)]
    section_header("Fare spread by event context")
    st.plotly_chart(
        px.box(
            event_subset,
            x="event_group",
            y="final_price_aed",
            color="event_group",
            title="Fare spread by event context",
            template="plotly_white",
        ).update_layout(
            height=420,
            margin={"l": 0, "r": 0, "t": 44, "b": 0},
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#0f172a", "family": "Inter, Segoe UI, sans-serif"},
        ),
        use_container_width=True,
    )
    event_stats = (
        event_subset.groupby("event_group")
        .agg(rides=("final_price_aed", "size"), avg_fare=("final_price_aed", "mean"), avg_demand=("demand_index", "mean"), avg_supply_pressure=("supply_pressure_index", "mean"))
        .reset_index()
        .sort_values("avg_fare", ascending=False)
    )
    section_header("Event statistics table")
    st.dataframe(event_stats, use_container_width=True, hide_index=True)