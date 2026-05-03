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

from utils.model_loader import get_global_interval_half_width, get_interval_basis_percent, load_dataset_profile, load_metrics, load_model_version, load_shap_bundle
from utils.ui import apply_theme, card, hero, section_header, sidebar_brand


apply_theme()

metrics = load_metrics()
profile = load_dataset_profile()
shap_bundle = load_shap_bundle()
version = load_model_version()
shap_summary = pd.DataFrame(shap_bundle["summary"])
_interval_basis_percent = get_interval_basis_percent(metrics)
_interval_half_width = get_global_interval_half_width(metrics)

hero(
    "XPrice — Explainable Ride Pricing",
    "Research Artifact · Dubai 2025",
    "A dual-audience pricing intelligence platform built on a 165k-ride Dubai mirror dataset. Combines a coordinate-first XGBoost fare model, polygon-resolved neighborhood labels, tree contribution decomposition, OSRM route previews, and optional live traffic overlays — giving both riders and operations managers transparent, explainable fare insights.",
)

with st.sidebar:
    sidebar_brand()
    st.markdown("### Navigation")
    st.markdown("Use the page picker above to switch between the **Rider Simulator**, the **Operations Dashboard**, and the **Feature Lab**.")
    st.markdown("### Live stats")
    st.markdown(f"- **Dataset rows:** {profile['rows']:,}")
    st.markdown(f"- **Completed rate:** {profile['completed_rate'] * 100:.1f}%")
    st.markdown(f"- **Model features:** {metrics['n_features']}")
    st.markdown(f"- **Test RMSE:** AED {metrics['test']['rmse']:.2f}")
    _cv = metrics.get("cv", {})
    if _cv.get("r2_mean") is not None:
        st.markdown(f"- **CV R² (5-fold):** {_cv['r2_mean']:.4f} ± {_cv['r2_std']:.4f}")
    if _interval_half_width:
        st.markdown(f"- **{_interval_basis_percent}% PI half-width:** ±AED {_interval_half_width:.2f}")
    if version.get("training_date"):
        st.caption(f"Model trained {version['training_date'][:10]}")

st.markdown("<br>", unsafe_allow_html=True)
metric_columns = st.columns(6, gap="small")
metric_columns[0].metric("Dataset", f"{profile['rows']:,} rides")
metric_columns[1].metric("Completed rides", f"{profile['completed_rate'] * 100:.1f}%")
metric_columns[2].metric("Average fare", f"AED {profile['avg_price']:.2f}")
metric_columns[3].metric("Test R²", f"{metrics['test']['r2']:.4f}")
_cv_mean = metrics.get("cv", {}).get("r2_mean")
metric_columns[4].metric("CV R² (5-fold)", f"{_cv_mean:.4f}" if _cv_mean else "N/A")
metric_columns[5].metric("Airport share", f"{profile['airport_share'] * 100:.1f}%")

st.markdown("<br>", unsafe_allow_html=True)
left, right = st.columns([1.05, 0.95], gap="large")

with left:
    card(
        "Why this app exists",
        "The earlier dashboard showed operational outcomes but not <em>why</em> specific fares moved. XPrice adds a trained pricing model, exact tree contribution decomposition, and audience-specific interfaces for riders and operations managers.",
    )
    card(
        "What changed in the data",
        "The dataset was regenerated and expanded to 72 columns. It now carries polygon-resolved pickup/dropoff neighborhoods, location-source flags, direct-route distance, route efficiency, bearing, density scores, and traffic features — so the model stays aligned with the map-selected coordinates shown in the simulator.",
    )

    section_header("Top global drivers")
    st.dataframe(
        shap_summary.rename(columns={"feature": "Feature", "mean_abs_shap": "Mean |contribution| (AED)"}),
        width="stretch",
        hide_index=True,
    )

with right:
    section_header("Contribution overview")
    figure_path = PROJECT_ROOT / "docs" / "figures" / "shap" / "shap_beeswarm.png"
    if figure_path.exists():
        st.image(str(figure_path), width="stretch")
    else:
        st.info("Run `python models/xai_analysis.py` to regenerate the global contribution figures.")

st.markdown("<br>", unsafe_allow_html=True)
bottom_left, bottom_right = st.columns(2, gap="large")
with bottom_left:
    section_header("Highest-value pickup zones")
    zone_frame = profile["top_pickup_zones"].rename("avg fare").reset_index().rename(columns={"pickup_zone": "zone"})
    zone_chart = px.bar(
        zone_frame,
        x="zone",
        y="avg fare",
        text="avg fare",
        title="Average completed fare by pickup zone",
        color="avg fare",
        color_continuous_scale="Teal",
        template="plotly_white",
    )
    zone_chart.update_traces(texttemplate="AED %{y:.1f}", textposition="outside")
    zone_chart.update_layout(
        height=360,
        margin={"l": 0, "r": 0, "t": 44, "b": 0},
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#0f172a", "family": "Inter, Segoe UI, sans-serif"},
        title_font={"size": 14, "color": "#0f172a"},
    )
    st.plotly_chart(zone_chart, width="stretch")

with bottom_right:
    section_header("Premium product mix")
    product_frame = profile["top_products"].rename("avg fare").reset_index().rename(columns={"product_type": "product"})
    product_chart = px.bar(
        product_frame,
        x="product",
        y="avg fare",
        text="avg fare",
        title="Average completed fare by product",
        color="avg fare",
        color_continuous_scale="Oranges",
        template="plotly_white",
    )
    product_chart.update_traces(texttemplate="AED %{y:.1f}", textposition="outside")
    product_chart.update_layout(
        height=360,
        margin={"l": 0, "r": 0, "t": 44, "b": 0},
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#0f172a", "family": "Inter, Segoe UI, sans-serif"},
        title_font={"size": 14, "color": "#0f172a"},
    )
    st.plotly_chart(product_chart, width="stretch")

st.caption("Calendar overlays use the 2025 research season. Live weather is optional and only active when an OpenWeatherMap API key is configured and the selected date is today.")
