from __future__ import annotations

import sys
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

from utils.model_loader import load_feature_columns, load_metrics, load_model, load_model_version, load_shap_bundle
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
model = load_model()
feature_columns_list = load_feature_columns()
metrics = load_metrics()
version = load_model_version()
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
    st.markdown("---")
    st.markdown("**Model**")
    st.caption(
        f"R² {metrics['test']['r2']:.4f} · RMSE AED {metrics['test']['rmse']:.2f}\n\n"
        f"CV R² {metrics['cv']['r2_mean']:.4f} ± {metrics['cv']['r2_std']:.4f}\n\n"
        f"±AED {metrics.get('prediction_interval_90_half_width', 0):.2f} (90% PI)"
    )
    if version.get("training_date"):
        st.caption(f"Trained {version['training_date'][:10]}")

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
summary_columns[3].metric("Avg traffic index", f"{filtered_raw['traffic_index'].mean():.2f}")

if len(filtered_raw) < 50:
    st.warning(
        f"Only {len(filtered_raw)} rides match the current filters. "
        "SHAP charts and partial-dependence plots may be noisy or blank for rare filter combinations. "
        "Try relaxing one filter (e.g. set Month to 'All') to get a more representative sample."
    )

st.markdown("<br>", unsafe_allow_html=True)
tab_global, tab_zone, tab_time, tab_event, tab_residuals = st.tabs(["Global view", "Zone lens", "Time lens", "Event lens", "Residuals"])

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
        .agg(rides=("final_price_aed", "size"), avg_fare=("final_price_aed", "mean"), avg_demand=("demand_index", "mean"), avg_traffic=("traffic_index", "mean"))
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
        .agg(rides=("final_price_aed", "size"), avg_fare=("final_price_aed", "mean"), avg_demand=("demand_index", "mean"), avg_traffic=("traffic_index", "mean"))
        .reset_index()
        .sort_values("avg_fare", ascending=False)
    )
    section_header("Event statistics table")
    st.dataframe(event_stats, use_container_width=True, hide_index=True)

    # SHAP contribution breakdown by event — which drivers change most across events?
    section_header("Feature contributions by event context")
    top_5_features = top_driver_frame["feature"].head(5).tolist()
    _feat_cols = list(filtered_features.columns)
    event_contrib_rows = []
    for evt in top_events:
        evt_mask = filtered_raw["active_event_display"].values == evt
        evt_vals = filtered_values[evt_mask]
        if len(evt_vals) == 0:
            continue
        for feat in top_5_features:
            if feat not in _feat_cols:
                continue
            fidx = _feat_cols.index(feat)
            event_contrib_rows.append({
                "event": evt,
                "feature": feat,
                "mean_contribution_aed": round(float(evt_vals[:, fidx].mean()), 3),
            })
    if event_contrib_rows:
        event_contrib_frame = pd.DataFrame(event_contrib_rows)
        event_contrib_chart = px.bar(
            event_contrib_frame,
            x="event",
            y="mean_contribution_aed",
            color="feature",
            barmode="group",
            title="Mean SHAP contribution by event context — top 5 global drivers",
            template="plotly_white",
        )
        event_contrib_chart.update_layout(
            height=440,
            margin={"l": 0, "r": 0, "t": 44, "b": 80},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#0f172a", "family": "Inter, Segoe UI, sans-serif"},
            xaxis={"gridcolor": "#f1f5f9", "tickangle": -20},
            yaxis={"title": "Mean contribution (AED)", "gridcolor": "#f1f5f9"},
            legend={"title": "Feature", "font": {"size": 11}},
        )
        st.plotly_chart(event_contrib_chart, use_container_width=True)
        st.caption(
            "Each group of bars shows the mean tree contribution of the top 5 global features for rides occurring "
            "during that event. Differences across events reveal which features are event-sensitive vs stable."
        )

with tab_residuals:
    # Compute model predictions on the filtered SHAP sample to derive residuals
    _filtered_pred = model.predict(filtered_features)
    _filtered_actual = filtered_raw["final_price_aed"].values
    _residuals = _filtered_actual - _filtered_pred

    st.markdown("<br>", unsafe_allow_html=True)
    res_cols = st.columns(4, gap="small")
    res_cols[0].metric("Mean residual", f"AED {_residuals.mean():.2f}")
    res_cols[1].metric("RMSE (filtered)", f"AED {np.sqrt((_residuals ** 2).mean()):.2f}")
    res_cols[2].metric("MAE (filtered)", f"AED {np.abs(_residuals).mean():.2f}")
    res_cols[3].metric("Max |error|", f"AED {np.abs(_residuals).max():.2f}")

    st.markdown("<br>", unsafe_allow_html=True)
    _chart_layout = dict(
        margin={"l": 0, "r": 0, "t": 44, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#0f172a", "family": "Inter, Segoe UI, sans-serif"},
    )

    section_header("Residual distribution")
    res_hist = px.histogram(
        x=_residuals,
        nbins=60,
        title="Distribution of (actual \u2212 predicted) residuals — AED",
        template="plotly_white",
        color_discrete_sequence=["#0d9488"],
    )
    res_hist.add_vline(x=0, line_dash="dash", line_color="#dc2626", line_width=1.5)
    res_hist.update_layout(height=300, xaxis_title="Residual (AED)", yaxis_title="Count", **_chart_layout)
    st.plotly_chart(res_hist, use_container_width=True)
    st.caption("Residuals centred near zero with thin tails indicate the model is well-calibrated. Skew or heavy tails would flag systematic bias.")

    res_left, res_right = st.columns(2, gap="large")

    with res_left:
        section_header("Residuals by pickup zone")
        _res_zone_frame = pd.DataFrame({"zone": filtered_raw["pickup_zone"].values, "residual": _residuals})
        zone_order = _res_zone_frame.groupby("zone")["residual"].median().sort_values().index.tolist()
        res_zone_box = px.box(
            _res_zone_frame,
            x="zone",
            y="residual",
            category_orders={"zone": zone_order},
            title="Residual spread by pickup zone",
            template="plotly_white",
            color_discrete_sequence=["#0d9488"],
        )
        res_zone_box.add_hline(y=0, line_dash="dash", line_color="#dc2626", line_width=1.5)
        res_zone_box.update_layout(
            height=420,
            xaxis={"tickangle": -35, "gridcolor": "#f1f5f9"},
            yaxis={"title": "Residual (AED)", "gridcolor": "#f1f5f9"},
            **_chart_layout,
        )
        st.plotly_chart(res_zone_box, use_container_width=True)

    with res_right:
        section_header("Mean residual by hour")
        _res_hour_frame = pd.DataFrame({"hour": filtered_raw["hour"].values, "residual": _residuals})
        hourly_bias = _res_hour_frame.groupby("hour")["residual"].mean().reset_index()
        hourly_bias["color"] = hourly_bias["residual"].apply(lambda v: "#dc2626" if v > 0 else "#0d9488")
        res_hour_bar = px.bar(
            hourly_bias,
            x="hour",
            y="residual",
            title="Mean residual by hour of day (positive = over-prediction)",
            template="plotly_white",
            color="residual",
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0,
        )
        res_hour_bar.add_hline(y=0, line_dash="dash", line_color="#475569", line_width=1)
        res_hour_bar.update_layout(
            height=420,
            xaxis={"title": "Hour", "gridcolor": "#f1f5f9", "dtick": 2},
            yaxis={"title": "Mean residual (AED)", "gridcolor": "#f1f5f9"},
            coloraxis_showscale=False,
            **_chart_layout,
        )
        st.plotly_chart(res_hour_bar, use_container_width=True)

    st.caption(
        "Residuals are computed on the 5,000-ride SHAP sample using the trained XGBoost model. "
        "These are in-sample residuals for the SHAP pool — the full held-out test set RMSE is AED "
        f"{metrics['test']['rmse']:.2f}."
    )