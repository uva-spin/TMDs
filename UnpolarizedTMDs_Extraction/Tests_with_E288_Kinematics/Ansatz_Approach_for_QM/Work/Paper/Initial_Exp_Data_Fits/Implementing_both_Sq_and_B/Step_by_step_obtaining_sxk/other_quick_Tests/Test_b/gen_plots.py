import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import griddata

# Create a folder for the plots
def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")

# Function to create a 2D heatmap plot with contours
def create_2d_plot(x_data, y_data, z_data, x_label, y_label, title, filename, 
                  cmap='viridis', point_size=10, interpolation_method='cubic'):
    
    # Create a figure with a specific size
    plt.figure(figsize=(12, 10))
    
    # Create a grid for interpolation
    x_min, x_max = np.min(x_data), np.max(x_data)
    y_min, y_max = np.min(y_data), np.max(y_data)
    
    # Create a grid with 100x100 points
    xi = np.linspace(x_min, x_max, 100)
    yi = np.linspace(y_min, y_max, 100)
    xi, yi = np.meshgrid(xi, yi)
    
    # Interpolate the data to the grid
    zi = griddata((x_data, y_data), z_data, (xi, yi), method=interpolation_method, fill_value=np.nan)
    
    # Plot the data points as a scatter plot
    scatter = plt.scatter(x_data, y_data, c=z_data, cmap=cmap, s=point_size, edgecolor='black', linewidth=0.5)
    
    # Plot the interpolated surface
    contour = plt.contourf(xi, yi, zi, 15, cmap=cmap, alpha=0.7)
    
    # Add contour lines
    contour_lines = plt.contour(xi, yi, zi, 15, colors='k', linewidths=0.5, alpha=0.5)
    plt.clabel(contour_lines, inline=1, fontsize=8, fmt='%.2f')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('SqT Mean Value')
    
    # Set labels and title
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.title(title, fontsize=16)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Tight layout to use space efficiently
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {filename}")
    
    # Show the plot
    plt.close()

def main():
    # Create plots folder
    plots_folder = 'SqT_Mean_Plots'
    create_folders(plots_folder)
    
    # Files directory - adjust if needed
    results_folder = 'Results_csvs'
    
    # Find the most recent CSV file in the results folder
    try:
        csv_files = [f for f in os.listdir(results_folder) if f.endswith('.csv')]
        if not csv_files:
            print(f"No CSV files found in {results_folder}")
            return
            
        # Sort by modification time (newest first)
        csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(results_folder, x)), reverse=True)
        latest_csv = os.path.join(results_folder, csv_files[0])
        print(f"Using the most recent CSV file: {latest_csv}")
    except FileNotFoundError:
        print(f"Directory {results_folder} not found. Please specify the correct path.")
        return
    
    # Read the CSV file
    try:
        df = pd.read_csv(latest_csv)
        print(f"Successfully loaded data with {len(df)} rows and {len(df.columns)} columns")
        print(f"Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Check if the required columns exist
    required_columns = ['x1', 'x2', 'qT', 'SqT_mean']
    for col in required_columns:
        if col not in df.columns:
            print(f"Required column '{col}' not found in the CSV file")
            return
    
    # Plot SqT_mean with respect to x1 and qT for fixed values of x2
    print("Generating x1 vs qT plots for fixed x2 values...")
    
    # Get all unique x2 values
    x2_values = sorted(df['x2'].unique())
    print(f"Found {len(x2_values)} unique x2 values in the dataset")
    
    # Plot for each unique x2 value
    for i, x2_val in enumerate(x2_values):
        # Filter data for this x2 value (with some tolerance for floating point)
        x2_df = df[np.isclose(df['x2'], x2_val, rtol=1e-5)]
        
        # Skip if not enough data points for a meaningful plot
        if len(x2_df) < 10:
            print(f"Skipping x2 = {x2_val:.4f} (only {len(x2_df)} data points)")
            continue
            
        # Generate the 2D heatmap plot
        filename = f"{plots_folder}/x1_qT_x2_{x2_val:.4f}.png"
        create_2d_plot(
            x2_df['x1'], x2_df['qT'], x2_df['SqT_mean'],
            'x1', 'qT', f'SqT Mean vs x1, qT (x2 = {x2_val:.4f})', 
            filename
        )
        
        # Print progress update every 5 plots or for the last one
        if (i+1) % 5 == 0 or i == len(x2_values) - 1:
            print(f"Progress: {i+1}/{len(x2_values)} x2 values processed")
    
    # Create a summary plot with sample x2 values
    if len(x2_values) > 5:
        # Select 5 representative x2 values
        selected_indices = np.linspace(0, len(x2_values)-1, 5, dtype=int)
        selected_x2_values = [x2_values[i] for i in selected_indices]
        
        plt.figure(figsize=(14, 10))
        
        # Create a plot showing the distribution of x2 values
        plt.subplot(121)
        plt.hist(x2_values, bins=20, alpha=0.7)
        plt.xlabel('x2 values')
        plt.ylabel('Count')
        plt.title('Distribution of x2 values in dataset')
        
        # Mark the selected values
        for x2_val in selected_x2_values:
            plt.axvline(x=x2_val, color='r', linestyle='--', alpha=0.5)
        
        # Create a second subplot showing the range of x1 and qT
        plt.subplot(122)
        plt.scatter(df['x1'], df['qT'], c=df['SqT_mean'], 
                   cmap='viridis', s=5, alpha=0.5)
        plt.xlabel('x1')
        plt.ylabel('qT')
        plt.title('Overview of x1, qT, and SqT_mean')
        plt.colorbar(label='SqT Mean')
        
        plt.tight_layout()
        plt.savefig(f"{plots_folder}/summary_plot.png", dpi=300, bbox_inches='tight')
        print(f"Summary plot saved as {plots_folder}/summary_plot.png")
        plt.close()
    
    print(f"All plots generated successfully in folder: {plots_folder}")

if __name__ == "__main__":
    main()