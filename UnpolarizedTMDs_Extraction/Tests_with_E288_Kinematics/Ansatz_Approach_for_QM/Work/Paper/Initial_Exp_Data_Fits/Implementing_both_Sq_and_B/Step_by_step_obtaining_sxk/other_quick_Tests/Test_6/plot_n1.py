import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot
import os

def plot_n1_3d():
    """
    Load n1 results from CSV and create interactive 3D plots.
    """
    # Check if the CSV file exists
    if not os.path.exists('csvs/n1_results.csv'):
        print("Error: n1_results.csv file not found in comparison_plots directory.")
        return
    
    # Load the data
    print("Loading n1 results data...")
    df = pd.read_csv('csvs/n1_results.csv')
    
    # Create directory if it doesn't exist
    os.makedirs('comparison_plots', exist_ok=True)
    
    # Get unique model indices
    unique_models = df['model'].unique()
    print(f"Found {len(unique_models)} unique models.")
    
    # First, create a combined 3D plot with all models
    create_combined_3d_plot(df)
    
    # Then create individual model plots
    for model_idx in unique_models:
        create_model_3d_plot(df, model_idx)

def create_combined_3d_plot(df):
    """Create a 3D plot with data from all models."""
    # First, calculate the mean n1 value for each x1, k combination across all models
    grouped = df.groupby(['x1', 'k'])
    n1_mean = grouped['n1'].mean().reset_index()
    n1_std = grouped['n1'].std().reset_index()
    
    # Merge the mean and std dataframes
    result_df = pd.merge(n1_mean, n1_std, on=['x1', 'k'], suffixes=('_mean', '_std'))
    
    # Create a 3D scatter plot with color based on n1 value
    fig = go.Figure(data=[go.Scatter3d(
        x=result_df['x1'],
        y=result_df['k'],
        z=result_df['n1_mean'],
        mode='markers',
        marker=dict(
            size=5,
            color=result_df['n1_mean'],
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title='n1 value')
        ),
        hovertemplate='x1: %{x:.4f}<br>k: %{y:.2f}<br>n1: %{z:.6f}<br>std: %{text:.6f}',
        text=result_df['n1_std']
    )])
    
    # Update layout
    fig.update_layout(
        title='3D Visualization of n1 Distribution (Mean across all models)',
        scene=dict(
            xaxis_title='x1',
            yaxis_title='k',
            zaxis_title='n1 value',
            xaxis=dict(type='linear'),
            yaxis=dict(type='linear'),
            zaxis=dict(type='linear')
        ),
        width=1000,
        height=800,
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    # Save the figure as an HTML file
    plot(fig, filename='comparison_plots/n1_3d_all_models.html', auto_open=False)
    print("Saved combined 3D plot: n1_3d_all_models.html")
    
    # Also create a 2D surface plot for better visualization
    create_surface_plot(result_df, 'all')

def create_model_3d_plot(df, model_idx):
    """Create a 3D plot for a specific model."""
    # Filter data for this model
    model_df = df[df['model'] == model_idx]
    
    # Create a 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=model_df['x1'],
        y=model_df['k'],
        z=model_df['n1'],
        mode='markers',
        marker=dict(
            size=5,
            color=model_df['n1'],
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title='n1 value')
        ),
        hovertemplate='x1: %{x:.4f}<br>k: %{y:.2f}<br>n1: %{z:.6f}'
    )])
    
    # Update layout
    fig.update_layout(
        title=f'3D Visualization of n1 Distribution (Model {model_idx})',
        scene=dict(
            xaxis_title='x1',
            yaxis_title='k',
            zaxis_title='n1 value',
            xaxis=dict(type='linear'),
            yaxis=dict(type='linear'),
            zaxis=dict(type='linear')
        ),
        width=1000,
        height=800,
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    # Save the figure as an HTML file
    plot(fig, filename=f'comparison_plots/n1_3d_model_{model_idx}.html', auto_open=False)
    print(f"Saved model {model_idx} 3D plot: n1_3d_model_{model_idx}.html")
    
    # Also create a 2D surface plot for better visualization
    create_surface_plot(model_df, model_idx)

def create_surface_plot(df, model_label):
    """Create a surface plot for better visualization of the 3D data."""
    # Pivot the data to get a grid
    if 'n1_mean' in df.columns:
        # For combined model data
        pivot_df = df.pivot(index='x1', columns='k', values='n1_mean')
    else:
        # For individual model data
        pivot_df = df.pivot(index='x1', columns='k', values='n1')
    
    # Create x and y grids
    x_values = pivot_df.index.values
    y_values = pivot_df.columns.values
    
    # Create the surface plot
    fig = go.Figure(data=[go.Surface(
        z=pivot_df.values,
        x=y_values,  # Using k for x-axis in the plot
        y=x_values,  # Using x1 for y-axis in the plot
        colorscale='Viridis'
    )])
    
    # Update layout
    if model_label == 'all':
        title = 'Surface Plot of n1 Distribution (Mean across all models)'
    else:
        title = f'Surface Plot of n1 Distribution (Model {model_label})'
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='k',
            yaxis_title='x1',
            zaxis_title='n1 value'
        ),
        width=1000,
        height=800,
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    # Save the figure as an HTML file
    if model_label == 'all':
        filename = 'comparison_plots/n1_surface_all_models.html'
    else:
        filename = f'comparison_plots/n1_surface_model_{model_label}.html'
    
    plot(fig, filename=filename, auto_open=False)
    print(f"Saved surface plot: {filename}")

if __name__ == "__main__":
    plot_n1_3d()