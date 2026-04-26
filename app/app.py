from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
for path in (APP_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils.model_loader import load_dataset_profile, load_metrics, load_shap_bundle
from utils.ui import apply_theme, card, hero


st.set_page_config(page_title="XPrice", layout="wide", initial_sidebar_state="expanded")
apply_theme()

metrics = load_metrics()
profile = load_dataset_profile()
shap_bundle = load_shap_bundle()
shap_summary = pd.DataFrame(shap_bundle["summary"])

hero(
    "XPrice | Explainable Ride Pricing for Dubai",
    "Research artifact",
    "A dual-audience pricing intelligence app built on a refreshed 165k-ride Dubai mirror dataset, a coordinate-first XGBoost fare model, OSRM route previews, and optional live traffic overlays for rider-level and operations-level transparency.",
)

with st.sidebar:
    st.markdown("### XPrice Navigation")
    st.markdown("Use the page picker above to switch between the rider simulator, the operations dashboard, and the feature lab.")
    st.markdown("### Live artefacts")
    st.markdown(f"- Dataset rows: {profile['rows']:,}")
    st.markdown(f"- Completed rate: {profile['completed_rate'] * 100:.1f}%")
    st.markdown(f"- Model features: {metrics['n_features']}")
    st.markdown(f"- Test RMSE: AED {metrics['test']['rmse']:.2f}")

metric_columns = st.columns(5)
metric_columns[0].metric("Dataset", f"{profile['rows']:,} rides")
metric_columns[1].metric("Completed rides", f"{profile['completed_rate'] * 100:.1f}%")
metric_columns[2].metric("Average fare", f"AED {profile['avg_price']:.2f}")
metric_columns[3].metric("Test R²", f"{metrics['test']['r2']:.4f}")
metric_columns[4].metric("Airport share", f"{profile['airport_share'] * 100:.1f}%")

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    card(
        "Why this app exists",
        "The earlier dashboard showed operational outcomes but not why specific fares moved. XPrice adds a trained pricing model, exact tree contribution decomposition, and audience-specific interfaces for riders and operations managers.",
    )
    card(
        "What changed in the data",
        "The dataset was regenerated and expanded to 68 columns. It now includes direct-route distance, route efficiency, bearing, pickup/dropoff density scores, and traffic index features so the model behaves more like a real pricing engine tied to map coordinates rather than neighborhood names.",
    )

    st.subheader("Top global drivers")
    st.dataframe(
        shap_summary.rename(columns={"feature": "Feature", "mean_abs_shap": "Mean absolute contribution (AED)"}),
        width="stretch",
        hide_index=True,
    )

with right:
    st.subheader("Contribution overview")
    figure_path = PROJECT_ROOT / "docs" / "figures" / "shap" / "shap_beeswarm.png"
    if figure_path.exists():
        st.image(str(figure_path), width="stretch")
    else:
        st.info("Run `python models/xai_analysis.py` to regenerate the global contribution figures.")

bottom_left, bottom_right = st.columns(2, gap="large")
with bottom_left:
    st.subheader("Highest-value pickup zones")
    zone_frame = profile["top_pickup_zones"].rename("avg fare").reset_index().rename(columns={"pickup_zone": "zone"})
    zone_chart = px.bar(
        zone_frame,
        x="zone",
        y="avg fare",
        text="avg fare",
        title="Average completed fare by pickup zone",
        color="avg fare",
        color_continuous_scale="Tealgrn",
    )
    zone_chart.update_traces(texttemplate="AED %{y:.1f}", textposition="outside")
    zone_chart.update_layout(height=360, margin={"l": 0, "r": 0, "t": 40, "b": 0}, coloraxis_showscale=False)
    st.plotly_chart(zone_chart, width="stretch")

with bottom_right:
    st.subheader("Premium product mix")
    product_frame = profile["top_products"].rename("avg fare").reset_index().rename(columns={"product_type": "product"})
    product_chart = px.bar(
        product_frame,
        x="product",
        y="avg fare",
        text="avg fare",
        title="Average completed fare by product",
        color="avg fare",
        color_continuous_scale="Sunsetdark",
    )
    product_chart.update_traces(texttemplate="AED %{y:.1f}", textposition="outside")
    product_chart.update_layout(height=360, margin={"l": 0, "r": 0, "t": 40, "b": 0}, coloraxis_showscale=False)
    st.plotly_chart(product_chart, width="stretch")

st.caption("Calendar overlays use the 2025 research season. Live weather is optional and only used when an OpenWeatherMap API key is present and the selected date is today.")