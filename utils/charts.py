import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# SECTOR PIE CHART
# ---------------------------------------------------------
def plot_sector_pie(sector_weights):
    """
    sector_weights: dict like {"Technology": 0.42, "Energy": 0.10, ...}
    """
    sectors = list(sector_weights.keys())
    weights = list(sector_weights.values())

    fig = px.pie(
        names=sectors,
        values=weights,
        title="Portfolio Sector Allocation",
        hole=0.4
    )

    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(showlegend=False)

    return fig


# ---------------------------------------------------------
# DRAWDOWN CHART
# ---------------------------------------------------------
def plot_drawdown(dd_df):
    """
    dd_df must contain:
    - value
    - peak
    - drawdown
    """
    fig = go.Figure()

    # Portfolio value
    fig.add_trace(go.Scatter(
        x=dd_df.index,
        y=dd_df["value"],
        name="Portfolio Value",
        line=dict(color="blue")
    ))

    # Drawdown (secondary axis)
    fig.add_trace(go.Scatter(
        x=dd_df.index,
        y=dd_df["drawdown"],
        name="Drawdown",
        line=dict(color="red"),
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
        height=500
    )

    return fig
