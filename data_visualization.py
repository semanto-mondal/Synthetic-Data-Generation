import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sdv.evaluation.single_table import get_column_pair_plot




def plot_feature_histogram(original_df, synthetic_df, column):
    """
    Plot box plots comparing original and synthetic data for a specific column.
    
    Args:
        original_df: Original pandas DataFrame
        synthetic_df: Synthetic pandas DataFrame
        column: Column name to plot
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    # Box plot for original data
    fig.add_trace(go.Box(
        y=original_df[column],
        name='Original',
        marker_color='blue',
        boxmean=True  # shows mean in addition to median
    ))

    # Box plot for synthetic data
    fig.add_trace(go.Box(
        y=synthetic_df[column],
        name='Synthetic',
        marker_color='red',
        boxmean=True
    ))

    fig.update_layout(
        title=f"Box Plot Comparison: {column}",
        yaxis_title=column,
        height=400
    )

    return fig


def plot_feature_kde(original_df, synthetic_df, column):
    """
    Create KDE (violin) plot comparing original and synthetic data distributions
    
    Args:
        original_df: Original pandas DataFrame
        synthetic_df: Synthetic pandas DataFrame
        column: Column name to plot
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Original data KDE
    fig.add_trace(
        go.Violin(
            x=original_df[column],
            name="Original",
            side='positive',
            line_color='blue',
            fillcolor='lightblue',
            opacity=0.6,
            points=False
        )
    )
    
    # Synthetic data KDE
    fig.add_trace(
        go.Violin(
            x=synthetic_df[column],
            name="Synthetic",
            side='negative',
            line_color='red',
            fillcolor='lightpink',
            opacity=0.6,
            points=False
        )
    )
    
    fig.update_layout(
        title_text=f"KDE Comparison: {column}",
        height=400,
        violingap=0,
        violinmode='overlay'
    )
    
    return fig

def plot_categorical_comparison(combined_counts, column):
    """
    Create bar chart for categorical feature comparison
    
    Args:
        combined_counts: DataFrame with combined categorical counts
        column: Column name being plotted
        
    Returns:
        Plotly figure object
    """
    fig = px.bar(
        combined_counts,
        x='value',
        y='proportion',
        color='dataset',
        barmode='group',
        title=f"Categorical Distribution: {column}",
        color_discrete_map={'Original': 'blue', 'Synthetic': 'red'}
    )
    
    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Proportion",
        legend_title="Dataset"
    )
    
    return fig

def plot_scatter_comparison(original_df, synthetic_df, x_col, y_col):
    """
    Create scatter plot comparing feature relationships
    
    Args:
        original_df: Original pandas DataFrame
        synthetic_df: Synthetic pandas DataFrame
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Original data scatter
    fig.add_trace(
        go.Scatter(
            x=original_df[x_col],
            y=original_df[y_col],
            mode='markers',
            name='Original',
            marker=dict(color='blue', opacity=0.7, size=8)
        )
    )
    
    # Synthetic data scatter
    fig.add_trace(
        go.Scatter(
            x=synthetic_df[x_col],
            y=synthetic_df[y_col],
            mode='markers',
            name='Synthetic',
            marker=dict(color='red', opacity=0.7, size=8)
        )
    )
    
    fig.update_layout(
        title=f"Scatter Plot: {x_col} vs {y_col}",
        xaxis_title=x_col,
        yaxis_title=y_col,
        height=600
    )
    
    return fig

def plot_density_comparison(original_df, synthetic_df, x_col, y_col):
    """
    Create density contour plots comparing feature relationships
    
    Args:
        original_df: Original pandas DataFrame
        synthetic_df: Synthetic pandas DataFrame
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        
    Returns:
        Plotly figure object
    """
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Original Data", "Synthetic Data"))
    
    # Original data density
    fig.add_trace(
        go.Histogram2dContour(
            x=original_df[x_col],
            y=original_df[y_col],
            colorscale='Blues',
            showscale=False,
            contours=dict(showlabels=True)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=original_df[x_col],
            y=original_df[y_col],
            mode='markers',
            marker=dict(color='blue', opacity=0.3, size=5),
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Synthetic data density
    fig.add_trace(
        go.Histogram2dContour(
            x=synthetic_df[x_col],
            y=synthetic_df[y_col],
            colorscale='Reds',
            showscale=False,
            contours=dict(showlabels=True)
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=synthetic_df[x_col],
            y=synthetic_df[y_col],
            mode='markers',
            marker=dict(color='red', opacity=0.3, size=5),
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title=f"Density Comparison: {x_col} vs {y_col}",
        height=500
    )
    
    return fig

def plot_correlation_matrix(corr_matrix, title="Correlation Matrix"):
    """
    Create correlation matrix heatmap
    
    Args:
        corr_matrix: Correlation matrix as pandas DataFrame
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    if corr_matrix.empty:
        return None
        
    fig = px.imshow(
        corr_matrix, 
        text_auto=True, 
        color_continuous_scale='RdBu_r',
        title=title
    )
    
    return fig

def plot_ks_test_results(ks_results):
    """
    Create bar chart visualizing KS test results
    
    Args:
        ks_results: DataFrame with KS test results
        
    Returns:
        Plotly figure object
    """
    fig = px.bar(
        ks_results, 
        x='Feature', 
        y='KS Statistic',
        color='Similarity',
        color_discrete_map={'High': 'green', 'Medium': 'orange', 'Low': 'red'},
        title="KS Test Statistics (Lower is Better)"
    )
    
    fig.update_layout(
        xaxis_title="Feature",
        yaxis_title="KS Statistic",
        height=500
    )
    
    return fig

def plot_parallel_coordinates(original_df, synthetic_df, selected_cols):
    """
    Create parallel coordinates plot for multiple features
    
    Args:
        original_df: Original pandas DataFrame
        synthetic_df: Synthetic pandas DataFrame
        selected_cols: List of column names to include
        
    Returns:
        Plotly figure object
    """
    # Prepare data for parallel coordinates
    orig_data = original_df[selected_cols].copy()
    orig_data['Dataset'] = 'Original'
    
    synth_data = synthetic_df[selected_cols].copy()
    synth_data['Dataset'] = 'Synthetic'
    
    combined_data = pd.concat([orig_data, synth_data])
    
    # Create parallel coordinates plot
    fig = px.parallel_coordinates(
        combined_data,
        color="Dataset",
        dimensions=selected_cols,
        color_discrete_map={'Original': 'blue', 'Synthetic': 'red'},
        title="Parallel Coordinates Plot"
    )
    
    fig.update_layout(height=600)
    return fig
