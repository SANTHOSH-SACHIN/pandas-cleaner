import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List, Dict, Any

def create_histogram(df, column: str, color_scheme: str = 'YlOrRd', nbins: int = 30) -> go.Figure:
    """Create an interactive histogram using Plotly."""
    colors = getattr(px.colors.sequential, color_scheme)
    fig = px.histogram(df, x=column, nbins=nbins, color_discrete_sequence=colors)
    fig.update_layout(
        title=f'Histogram of {column}',
        xaxis_title=column,
        yaxis_title='Count',
        hovermode='x',
        template='plotly_white'
    )
    return fig

def create_box_plot(df, column: str, group_by: Optional[str] = None, color_scheme: str = 'YlOrRd') -> go.Figure:
    """Create an interactive box plot using Plotly."""
    if group_by:
        colors = getattr(px.colors.sequential, color_scheme)
        fig = px.box(df, x=group_by, y=column, color=group_by, color_discrete_sequence=colors)
    else:
        colors = getattr(px.colors.sequential, color_scheme)
        fig = px.box(df, y=column, color_discrete_sequence=colors)

    fig.update_layout(
        title=f'Box Plot of {column}',
        yaxis_title=column,
        template='plotly_white'
    )
    return fig

def create_scatter_plot(
    df, x_column: str, y_column: str,
    color_column: Optional[str] = None,
    size_column: Optional[str] = None,
    color_scheme: str = 'YlOrRd'
) -> go.Figure:
    """Create an interactive scatter plot using Plotly."""
    fig = px.scatter(
        df, x=x_column, y=y_column,
        color=color_column,
        size=size_column,
        color_discrete_sequence=getattr(px.colors.sequential, color_scheme)
    )

    fig.update_layout(
        title=f'{y_column} vs {x_column}',
        xaxis_title=x_column,
        yaxis_title=y_column,
        template='plotly_white'
    )
    return fig

def create_line_plot(
    df, x_column: str, y_column: str,
    color_column: Optional[str] = None,
    color_scheme: str = 'YlOrRd'
) -> go.Figure:
    """Create an interactive line plot using Plotly."""
    fig = px.line(
        df, x=x_column, y=y_column,
        color=color_column,
        color_discrete_sequence=getattr(px.colors.sequential, color_scheme)
    )

    fig.update_layout(
        title=f'{y_column} vs {x_column}',
        xaxis_title=x_column,
        yaxis_title=y_column,
        template='plotly_white'
    )
    return fig

def create_bar_plot(
    df, x_column: str, y_column: str,
    color_column: Optional[str] = None,
    color_scheme: str = 'YlOrRd'
) -> go.Figure:
    """Create an interactive bar plot using Plotly."""
    fig = px.bar(
        df, x=x_column, y=y_column,
        color=color_column,
        color_discrete_sequence=getattr(px.colors.sequential, color_scheme)
    )

    fig.update_layout(
        title=f'{y_column} by {x_column}',
        xaxis_title=x_column,
        yaxis_title=y_column,
        template='plotly_white'
    )
    return fig

def create_correlation_heatmap(df, numeric_columns: List[str]) -> go.Figure:
    """Create a correlation heatmap using Plotly."""
    corr_matrix = df[numeric_columns].corr()

    fig = px.imshow(
        corr_matrix,
        color_continuous_scale='RdBu',
        aspect='auto'
    )

    fig.update_layout(
        title='Correlation Heatmap',
        template='plotly_white'
    )
    return fig

# List of available color schemes in plotly express
COLOR_SCHEMES = [
    'Viridis', 'Plasma', 'Inferno', 'Magma', 'Reds', 'YlOrRd', 'YlOrBr',
    'YlGnBu', 'YlGn', 'Blues', 'Greens', 'Purples'
]

PLOT_TYPES = {
    'Histogram': create_histogram,
    'Box Plot': create_box_plot,
    'Scatter Plot': create_scatter_plot,
    'Line Plot': create_line_plot,
    'Bar Plot': create_bar_plot,
    'Correlation Heatmap': create_correlation_heatmap
}
