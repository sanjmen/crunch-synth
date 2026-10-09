"""
Visualization utilities for predictive densities, realized price paths, and CRPS trajectories.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from densitypdf import density_pdf


def plot_mixture_density(
    density_dict: dict,
    realized_return: Optional[float] = None,
    bound: float = 500.0,
    num_points: int = 1000,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """
    Plots the probability density function (PDF) of a predicted mixture distribution,
    optionally superimposing the realized market price increment.

    Parameters
    ----------
    density_dict : dict
        Mixture dictionary complying with densitypdf / CrunchDAO format.
    realized_return : Optional[float]
        Realized price return / increment observed during the forecast interval.
    bound : float
        Symmetric bound [-bound, +bound] over which to evaluate the PDF.
    num_points : int
        Grid resolution.
    title : Optional[str]
        Title for the figure.
    save_path : Optional[Union[str, Path]]
        If provided, writes the plot to an HTML file.

    Returns
    -------
    go.Figure
    """
    x_grid = np.linspace(-bound, bound, num_points)
    pdf_values = [density_pdf(density_dict, float(xi)) for xi in x_grid]

    fig = go.Figure()

    # Predicted density curve
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=pdf_values,
            mode="lines",
            line=dict(color="#1f77b4", width=2.5),
            name="Forecasted PDF",
            fill="tozeroy",
            fillcolor="rgba(31, 119, 180, 0.15)",
        )
    )

    # Realized observation indicator
    if realized_return is not None:
        realized_density = float(density_pdf(density_dict, float(realized_return)))
        fig.add_trace(
            go.Scatter(
                x=[realized_return],
                y=[realized_density],
                mode="markers",
                marker=dict(color="red", size=10, symbol="x"),
                name=f"Observed Return ({realized_return:+.2f})",
            )
        )
        fig.add_vline(
            x=realized_return,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Realized: {realized_return:+.2f}",
            annotation_position="top right",
        )

    fig.update_layout(
        title=dict(text=title or "Predicted Return Density Mixture", x=0.5, font=dict(size=18)),
        xaxis=dict(title="Price Increment ΔP", showgrid=True, gridcolor="rgba(200,200,200,0.4)"),
        yaxis=dict(title="Probability Density", showgrid=True, gridcolor="rgba(200,200,200,0.4)"),
        plot_bgcolor="white",
        hovermode="x unified",
        margin=dict(l=50, r=40, t=60, b=50),
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path))

    return fig


def plot_crps_timeline(
    score_df: pd.DataFrame,
    asset: str,
    rolling_window: int = 10,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """
    Plots the longitudinal CRPS evaluation score trajectory across time.

    Parameters
    ----------
    score_df : pd.DataFrame
        DataFrame with columns ['timestamp', 'datetime', 'score'].
    asset : str
        Asset symbol.
    rolling_window : int
        Window for rolling moving average CRPS.
    title : Optional[str]
        Title for the figure.
    save_path : Optional[Union[str, Path]]
        HTML export destination.
    """
    if score_df.empty or "score" not in score_df.columns:
        raise ValueError("score_df must contain non-empty 'score' column")

    df = score_df.copy().sort_values("timestamp")
    df["rolling_crps"] = df["score"].rolling(rolling_window, min_periods=1).mean()
    mean_val = float(df["score"].mean())

    fig = go.Figure()

    # Instantaneous score points
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["score"],
            mode="markers",
            marker=dict(color="rgba(100, 100, 100, 0.4)", size=5),
            name="Step CRPS",
        )
    )

    # Rolling mean
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["rolling_crps"],
            mode="lines",
            line=dict(color="#2ca02c", width=2.5),
            name=f"Rolling Mean (w={rolling_window})",
        )
    )

    # Global mean line
    fig.add_hline(
        y=mean_val,
        line_dash="dot",
        line_color="crimson",
        annotation_text=f"Mean: {mean_val:.4f}",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title=dict(text=title or f"{asset} — CRPS Score Progression", x=0.5, font=dict(size=18)),
        xaxis=dict(title="Time (UTC)", showgrid=True, gridcolor="rgba(200,200,200,0.4)"),
        yaxis=dict(title="CRPS (Lower is better)", showgrid=True, gridcolor="rgba(200,200,200,0.4)"),
        plot_bgcolor="white",
        hovermode="x unified",
        margin=dict(l=50, r=40, t=60, b=50),
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path))

    return fig


def plot_leaderboard_ranks(
    df_comparison: pd.DataFrame,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """
    Generates a grouped bar chart comparing Tracker vs Official Benchmark CRPS per asset.
    """
    if df_comparison.empty:
        raise ValueError("df_comparison cannot be empty")

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df_comparison["asset"],
            y=df_comparison["my_score"],
            name="My Tracker",
            marker_color="#1f77b4",
        )
    )

    if "benchmark_score" in df_comparison.columns and not df_comparison["benchmark_score"].isna().all():
        fig.add_trace(
            go.Bar(
                x=df_comparison["asset"],
                y=df_comparison["benchmark_score"],
                name="Benchmark Model",
                marker_color="#ff7f0e",
            )
        )

    fig.update_layout(
        title=dict(text=title or "CRPS Comparison: Tracker vs Benchmark", x=0.5, font=dict(size=18)),
        xaxis=dict(title="Asset", showgrid=True),
        yaxis=dict(title="Mean CRPS (Lower is better)", showgrid=True, gridcolor="rgba(200,200,200,0.4)"),
        barmode="group",
        plot_bgcolor="white",
        margin=dict(l=50, r=40, t=60, b=50),
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path))

    return fig
