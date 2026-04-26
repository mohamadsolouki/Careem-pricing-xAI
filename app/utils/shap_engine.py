from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb


def _percentile_scale(series: pd.Series) -> np.ndarray:
    lower = np.nanpercentile(series, 5)
    upper = np.nanpercentile(series, 95)
    if np.isclose(lower, upper):
        return np.full(len(series), 0.5)
    clipped = np.clip(series, lower, upper)
    return (clipped - lower) / (upper - lower)


def compute_contributions(model, feature_frame: pd.DataFrame):
    booster = model.get_booster()
    matrix = xgb.DMatrix(feature_frame, feature_names=list(feature_frame.columns))
    contrib_matrix = booster.predict(matrix, pred_contribs=True)
    values = contrib_matrix[:, :-1]
    base_values = contrib_matrix[:, -1]
    return values, base_values


def compute_local_contributions(model, feature_frame: pd.DataFrame):
    values, base_values = compute_contributions(model, feature_frame)
    contribution_series = pd.Series(values[0], index=feature_frame.columns)
    prediction = float(base_values[0] + contribution_series.sum())
    return contribution_series, float(base_values[0]), prediction


def plot_waterfall(contribution_series: pd.Series, base_value: float, predicted_price: float, max_display: int = 8):
    top = contribution_series.reindex(contribution_series.abs().sort_values(ascending=False).head(max_display).index)
    remainder = contribution_series.drop(top.index).sum()
    if abs(remainder) > 1e-6:
        top.loc["Other factors"] = remainder

    ordered = top.sort_values()
    figure, axis = plt.subplots(figsize=(10, 6))
    running = base_value
    y_positions = np.arange(len(ordered))

    for position, (label, value) in enumerate(ordered.items()):
        start = running if value >= 0 else running + value
        color = "#d97706" if value >= 0 else "#0f766e"
        axis.barh(position, abs(value), left=start, color=color, edgecolor="white", height=0.65)
        axis.text(start + abs(value) + 0.6, position, f"{value:+.2f}", va="center", fontsize=9)
        running += value

    axis.axvline(base_value, color="#475569", linestyle="--", linewidth=1, label="Baseline")
    axis.axvline(predicted_price, color="#111827", linewidth=1.4, label="Prediction")
    axis.set_yticks(y_positions)
    axis.set_yticklabels(ordered.index)
    axis.set_xlabel("Fare impact (AED)")
    axis.set_title(f"Local price decomposition | Baseline AED {base_value:.2f} → Prediction AED {predicted_price:.2f}")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    plt.tight_layout()
    return figure


def plot_beeswarm(contrib_values: np.ndarray, feature_frame: pd.DataFrame, max_display: int = 20):
    mean_abs = np.abs(contrib_values).mean(axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:max_display]
    ordered_indices = top_indices[::-1]

    figure, axis = plt.subplots(figsize=(12, 8))
    scatter = None
    for row_index, feature_index in enumerate(ordered_indices):
        x_values = contrib_values[:, feature_index]
        y_values = np.random.normal(loc=row_index, scale=0.10, size=len(x_values))
        color_scale = _percentile_scale(feature_frame.iloc[:, feature_index])
        scatter = axis.scatter(x_values, y_values, c=color_scale, cmap="coolwarm", s=8, alpha=0.45, linewidths=0)

    axis.axvline(0, color="#475569", linestyle="--", linewidth=1)
    axis.set_yticks(range(len(ordered_indices)))
    axis.set_yticklabels([feature_frame.columns[index] for index in ordered_indices], fontsize=9)
    axis.set_xlabel("Feature contribution to price (AED)")
    axis.set_title("Global price drivers across the filtered ride sample")
    axis.spines[["top", "right"]].set_visible(False)
    if scatter is not None:
        colorbar = figure.colorbar(scatter, ax=axis, pad=0.01)
        colorbar.set_label("Relative feature value")
    plt.tight_layout()
    return figure


def plot_importance_bar(contrib_values: np.ndarray, feature_names: list[str], max_display: int = 20):
    mean_abs = np.abs(contrib_values).mean(axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:max_display]
    ordered_indices = top_indices[::-1]

    figure, axis = plt.subplots(figsize=(10, 7))
    axis.barh(
        [feature_names[index] for index in ordered_indices],
        mean_abs[ordered_indices],
        color="#0f766e",
        edgecolor="white",
        linewidth=0.6,
    )
    axis.set_xlabel("Mean absolute contribution (AED)")
    axis.set_title("Global contribution ranking")
    axis.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return figure


def plot_dependence(contrib_values: np.ndarray, feature_frame: pd.DataFrame, feature_name: str):
    feature_index = feature_frame.columns.get_loc(feature_name)
    x_values = feature_frame[feature_name]
    y_values = contrib_values[:, feature_index]
    color_scale = _percentile_scale(x_values)

    figure, axis = plt.subplots(figsize=(10, 6))
    scatter = axis.scatter(x_values, y_values, c=color_scale, cmap="viridis", s=12, alpha=0.45, linewidths=0)
    axis.axhline(0, color="#475569", linestyle="--", linewidth=1)
    axis.set_xlabel(feature_name)
    axis.set_ylabel("Contribution (AED)")
    axis.set_title(f"Contribution dependence: {feature_name}")
    axis.spines[["top", "right"]].set_visible(False)
    colorbar = figure.colorbar(scatter, ax=axis, pad=0.01)
    colorbar.set_label("Relative feature value")
    plt.tight_layout()
    return figure


def build_top_driver_frame(contrib_values: np.ndarray, feature_names: list[str], max_display: int = 12):
    mean_abs = np.abs(contrib_values).mean(axis=0)
    ranking = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_contribution": mean_abs,
    }).sort_values("mean_abs_contribution", ascending=False).head(max_display)
    ranking["mean_abs_contribution"] = ranking["mean_abs_contribution"].round(3)
    return ranking.reset_index(drop=True)