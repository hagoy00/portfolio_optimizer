import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


# =========================================================
# SAFE SECTOR PIE CHART
# =========================================================
def plot_sector_pie(sector_weights):
    """
    Creates a crash-proof sector allocation pie chart.
    sector_weights: dict like {"Technology": 0.42, "Energy": 0.10, ...}
    """

    # Guard clause — empty or invalid input
    if not sector_weights or len(sector_weights) == 0:
        fig = go.Figure()
        fig.update_layout(
            title="Sector Allocation (No Data)",
            annotations=[dict(text="No Data", x=0.5, y=0.5, showarrow=False)]
        )
        return fig

    sectors = list(sector_weights.keys())
    weights = list(sector_weights.values())

    # Normalize weights if they don't sum to 1
    total = sum(weights)
    if total <= 0 or pd.isna(total):
        weights = [1 / len(weights)] * len(weights)
    else:
        weights = [w / total for w in weights]

    fig = px.pie(
        names=sectors,
        values=weights,
        title="Portfolio Sector Allocation",
        hole=0.4
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        pull=[0.02] * len(sectors)
    )

    fig.update_layout(
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig


# =========================================================
# SAFE DRAWDOWN CHART
# =========================================================
def plot_drawdown(dd_df):
    """
    Drawdown chart with dual-axis:
    - Portfolio value (left axis)
    - Drawdown % (right axis)
    dd_df must contain columns: value, peak, drawdown
    """

    # Guard clause — empty or missing columns
    if dd_df is None or len(dd_df) == 0:
        fig = go.Figure()
        fig.update_layout(
            title="Portfolio Drawdown (No Data)",
            annotations=[dict(text="No Data", x=0.5, y=0.5, showarrow=False)]
        )
        return fig

    required_cols = {"value", "peak", "drawdown"}
    if not required_cols.issubset(dd_df.columns):
        fig = go.Figure()
        fig.update_layout(
            title="Portfolio Drawdown (Invalid Data)",
            annotations=[dict(text="Invalid Data", x=0.5, y=0.5, showarrow=False)]
        )
        return fig

    fig = go.Figure()

    # Portfolio value
    fig.add_trace(go.Scatter(
        x=dd_df.index,
        y=dd_df["value"],
        name="Portfolio Value",
        line=dict(color="#1f77b4", width=2)
    ))

    # Drawdown
    fig.add_trace(go.Scatter(
        x=dd_df.index,
        y=dd_df["drawdown"],
        name="Drawdown",
        line=dict(color="#d62728", width=2, dash="dot"),
        yaxis="y2"
    ))

    fig.update_layout(
        title="Portfolio Drawdown",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        yaxis2=dict(
            title="Drawdown",
            overlaying="y",
            side="right",
            tickformat=".0%",
        ),
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig
