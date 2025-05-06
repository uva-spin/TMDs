import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
import sys

# Check if seaborn is available
HAS_SEABORN = True
try:
    import seaborn as sns
    print("Successfully imported seaborn")
except ImportError:
    print("Warning: seaborn is not installed. Using matplotlib for heatmap.")
    HAS_SEABORN = False

# Configuration
RESULTS_FOLDER = 'Results_csvs'
FIGURE_FOLDER = 'Figures'
GRID_FILE = 'grid_results_xF5_x110_qT20_QM20.csv'  # Adjust this based on your actual filename

# Create figure folder if it doesn't exist
if not os.path.exists(FIGURE_FOLDER):
    os.makedirs(FIGURE_FOLDER)
    print(f"Created folder '{FIGURE_FOLDER}' for saving visualizations")

# Load the grid data
print(f"Loading grid data from {os.path.join(RESULTS_FOLDER, GRID_FILE)}...")
try:
    grid_data = pd.read_csv(os.path.join(RESULTS_FOLDER, GRID_FILE))
    print(f"Loaded {len(grid_data)} grid points successfully")
except FileNotFoundError:
    raise FileNotFoundError(f"Grid file not found. Make sure the path is correct: {os.path.join(RESULTS_FOLDER, GRID_FILE)}")

# Check if 3D plotting is available
HAS_3D = True
try:
    from mpl_toolkits.mplot3d import Axes3D
    print("Successfully imported 3D plotting capabilities")
except ImportError:
    print("Warning: 3D plotting capabilities not available. 3D plots will be skipped.")
    HAS_3D = False

# 1. Create S(qT) plots for different fixed values of x1 and x2
def plot_S_vs_qT():
    plt.figure(figsize=(12, 8))
    
    # Get unique values of x1 and x2 (rounded to 3 decimal places to handle floating point issues)
    grid_data['x1_rounded'] = grid_data['x1'].round(3)
    grid_data['x2_rounded'] = grid_data['x2'].round(3)
    unique_x1 = sorted(grid_data['x1_rounded'].unique())
    unique_x2 = sorted(grid_data['x2_rounded'].unique())
    
    # Select a subset of x1, x2 combinations for clarity
    selected_x1 = unique_x1[::len(unique_x1)//5][:5]  # Choose ~5 values
    selected_x2 = unique_x2[::len(unique_x2)//5][:5]  # Choose ~5 values
    
    # Plot S vs qT for different (x1, x2) pairs
    colors = plt.cm.viridis(np.linspace(0, 1, len(selected_x1)))
    markers = ['o', 's', '^', 'd', 'x']
    
    for i, x1 in enumerate(selected_x1):
        for j, x2 in enumerate(selected_x2):
            subset = grid_data[
                (grid_data['x1_rounded'] == x1) & 
                (grid_data['x2_rounded'] == x2)
            ]
            
            if len(subset) > 0:
                # Get a fixed QM value for consistency
                median_QM = subset['QM'].median()
                qm_subset = subset[np.isclose(subset['QM'], median_QM, rtol=1e-2)]
                
                if len(qm_subset) > 0:
                    plt.plot(
                        qm_subset['qT'], 
                        qm_subset['SqT_mean'], 
                        marker=markers[j % len(markers)],
                        color=colors[i],
                        label=f'x1={x1:.3f}, x2={x2:.3f}',
                        linewidth=2,
                        markersize=6
                    )
    
    plt.xlabel('qT', fontsize=14)
    plt.ylabel('S(qT, x1, x2)', fontsize=14)
    plt.title('S(qT) for Different x1, x2 Values', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{FIGURE_FOLDER}/S_vs_qT.png', dpi=300, bbox_inches='tight')
    print(f"Saved S vs qT plot to {FIGURE_FOLDER}/S_vs_qT.png")

# 2. Create 3D surface plot of S(qT, x1) for fixed x2
def plot_S_surface_qT_x1():
    if not HAS_3D:
        print("Skipping 3D surface plot due to missing 3D plotting capabilities")
        return
        
    # Select a middle value of x2
    unique_x2 = grid_data['x2'].round(3).unique()
    selected_x2 = unique_x2[len(unique_x2)//2]  # Middle value
    
    # Filter for the selected x2 and a fixed QM value
    filtered_data = grid_data[np.isclose(grid_data['x2'], selected_x2, rtol=1e-2)]
    median_QM = filtered_data['QM'].median()
    filtered_data = filtered_data[np.isclose(filtered_data['QM'], median_QM, rtol=1e-2)]
    
    if len(filtered_data) > 10:  # Ensure we have enough data points
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create a 2D grid for the surface
        x1_unique = sorted(filtered_data['x1'].unique())
        qT_unique = sorted(filtered_data['qT'].unique())
        
        X1, QT = np.meshgrid(x1_unique, qT_unique)
        S_values = np.zeros(X1.shape)
        
        # Fill the grid with S values
        for i, qT in enumerate(qT_unique):
            for j, x1 in enumerate(x1_unique):
                matching_rows = filtered_data[
                    (np.isclose(filtered_data['qT'], qT, rtol=1e-3)) & 
                    (np.isclose(filtered_data['x1'], x1, rtol=1e-3))
                ]
                if len(matching_rows) > 0:
                    S_values[i, j] = matching_rows['SqT_mean'].values[0]
        
        # Plot the surface
        surf = ax.plot_surface(X1, QT, S_values, cmap=cm.viridis,
                              linewidth=0, antialiased=True, alpha=0.8)
        
        # Add color bar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='S(qT, x1, x2)')
        
        ax.set_xlabel('x1', fontsize=14)
        ax.set_ylabel('qT', fontsize=14)
        ax.set_zlabel('S(qT, x1, x2)', fontsize=14)
        ax.set_title(f'S(qT, x1) Surface for x2={selected_x2:.3f}, QM={median_QM:.2f}', fontsize=16)
        
        plt.savefig(f'{FIGURE_FOLDER}/S_surface_qT_x1.png', dpi=300, bbox_inches='tight')
        print(f"Saved 3D surface plot to {FIGURE_FOLDER}/S_surface_qT_x1.png")
    else:
        print("Not enough data points for 3D surface plot")

# 3. Create contour plot of S in the (x1, x2) plane for fixed qT
def plot_S_contour_x1_x2():
    # Select a middle value of qT
    unique_qT = sorted(grid_data['qT'].unique())
    selected_qT = unique_qT[len(unique_qT)//2]  # Middle value
    
    # Filter for the selected qT and a fixed QM
    filtered_data = grid_data[np.isclose(grid_data['qT'], selected_qT, rtol=1e-2)]
    median_QM = filtered_data['QM'].median()
    filtered_data = filtered_data[np.isclose(filtered_data['QM'], median_QM, rtol=1e-2)]
    
    if len(filtered_data) > 10:
        plt.figure(figsize=(10, 8))
        
        # Create a pivot table for the contour plot
        pivot_data = filtered_data.pivot_table(
            values='SqT_mean', 
            index='x1', 
            columns='x2', 
            aggfunc='mean'
        )
        
        # Create the contour plot
        contour = plt.contourf(
            pivot_data.columns, 
            pivot_data.index, 
            pivot_data.values, 
            20, 
            cmap='viridis'
        )
        
        plt.colorbar(contour, label='S(qT, x1, x2)')
        plt.xlabel('x2', fontsize=14)
        plt.ylabel('x1', fontsize=14)
        plt.title(f'S(x1, x2) Contour for qT={selected_qT:.2f}, QM={median_QM:.2f}', fontsize=16)
        plt.grid(True, alpha=0.3)
        
        # Add contour lines for better visibility
        contour_lines = plt.contour(
            pivot_data.columns, 
            pivot_data.index, 
            pivot_data.values, 
            10, 
            colors='black', 
            alpha=0.5,
            linewidths=0.5
        )
        plt.clabel(contour_lines, inline=True, fontsize=8)
        
        plt.savefig(f'{FIGURE_FOLDER}/S_contour_x1_x2.png', dpi=300, bbox_inches='tight')
        print(f"Saved contour plot to {FIGURE_FOLDER}/S_contour_x1_x2.png")
    else:
        print("Not enough data points for contour plot")

# 4. Create heatmap showing S vs qT and xF
def plot_S_heatmap_qT_xF():
    plt.figure(figsize=(12, 10))
    
    # Get a fixed QM value for consistency
    median_QM = grid_data['QM'].median()
    filtered_data = grid_data[np.isclose(grid_data['QM'], median_QM, rtol=1e-2)]
    
    # Create a pivot table for the heatmap
    pivot_data = filtered_data.pivot_table(
        values='SqT_mean', 
        index='qT', 
        columns='xF', 
        aggfunc='mean'
    )
    
    # Create the heatmap
    if HAS_SEABORN:
        # Use seaborn if available
        ax = sns.heatmap(
            pivot_data, 
            cmap='viridis', 
            cbar_kws={'label': 'S(qT, x1, x2)'}
        )
    else:
        # Use matplotlib's pcolormesh as alternative
        plt.pcolormesh(
            pivot_data.columns, 
            pivot_data.index, 
            pivot_data.values, 
            cmap='viridis'
        )
        plt.colorbar(label='S(qT, x1, x2)')
    
    plt.xlabel('xF', fontsize=14)
    plt.ylabel('qT', fontsize=14)
    plt.title(f'S(qT, xF) Heatmap for QM={median_QM:.2f}', fontsize=16)
    
    plt.savefig(f'{FIGURE_FOLDER}/S_heatmap_qT_xF.png', dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to {FIGURE_FOLDER}/S_heatmap_qT_xF.png")

# 5. Create S vs x1/x2 ratio plot for different qT values
def plot_S_vs_x1_x2_ratio():
    plt.figure(figsize=(12, 8))
    
    # Calculate x1/x2 ratio
    grid_data['x1_x2_ratio'] = grid_data['x1'] / grid_data['x2']
    
    # Get unique values of qT (rounded for stability)
    grid_data['qT_rounded'] = grid_data['qT'].round(2)
    unique_qT = sorted(grid_data['qT_rounded'].unique())
    
    # Select a subset of qT values
    selected_qT = unique_qT[::len(unique_qT)//5][:5]  # Choose ~5 values
    
    # Get a fixed QM value for consistency
    median_QM = grid_data['QM'].median()
    filtered_data = grid_data[np.isclose(grid_data['QM'], median_QM, rtol=1e-2)]
    
    # Plot S vs x1/x2 ratio for different qT values
    colors = plt.cm.plasma(np.linspace(0, 1, len(selected_qT)))
    
    for i, qT in enumerate(selected_qT):
        subset = filtered_data[np.isclose(filtered_data['qT_rounded'], qT, rtol=1e-2)]
        
        if len(subset) > 0:
            # Sort by ratio for smooth lines
            subset = subset.sort_values('x1_x2_ratio')
            
            plt.plot(
                subset['x1_x2_ratio'], 
                subset['SqT_mean'], 
                'o-',
                color=colors[i],
                label=f'qT={qT:.2f}',
                linewidth=2,
                markersize=5
            )
    
    plt.xlabel('x1/x2 Ratio', fontsize=14)
    plt.ylabel('S(qT, x1, x2)', fontsize=14)
    plt.title(f'S vs x1/x2 Ratio for Different qT Values (QM={median_QM:.2f})', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xscale('log')
    plt.tight_layout()
    
    plt.savefig(f'{FIGURE_FOLDER}/S_vs_x1_x2_ratio.png', dpi=300, bbox_inches='tight')
    print(f"Saved x1/x2 ratio plot to {FIGURE_FOLDER}/S_vs_x1_x2_ratio.png")

# 6. Alternative to 3D surface: Create 2D version with multiple lines
def plot_S_lines_qT_x1():
    # This is a 2D alternative to the 3D surface plot
    plt.figure(figsize=(12, 8))
    
    # Select a middle value of x2
    unique_x2 = grid_data['x2'].round(3).unique()
    selected_x2 = unique_x2[len(unique_x2)//2]  # Middle value
    
    # Filter for the selected x2 and a fixed QM value
    filtered_data = grid_data[np.isclose(grid_data['x2'], selected_x2, rtol=1e-2)]
    median_QM = filtered_data['QM'].median()
    filtered_data = filtered_data[np.isclose(filtered_data['QM'], median_QM, rtol=1e-2)]
    
    # Get unique values of x1 (rounded for stability)
    filtered_data['x1_rounded'] = filtered_data['x1'].round(3)
    unique_x1 = sorted(filtered_data['x1_rounded'].unique())
    
    # Select a subset of x1 values
    selected_x1 = unique_x1[::len(unique_x1)//5][:5]  # Choose ~5 values
    
    # Plot S vs qT for different x1 values
    colors = plt.cm.cool(np.linspace(0, 1, len(selected_x1)))
    
    for i, x1 in enumerate(selected_x1):
        subset = filtered_data[np.isclose(filtered_data['x1_rounded'], x1, rtol=1e-3)]
        
        if len(subset) > 0:
            # Sort by qT for smooth lines
            subset = subset.sort_values('qT')
            
            plt.plot(
                subset['qT'], 
                subset['SqT_mean'], 
                'o-',
                color=colors[i],
                label=f'x1={x1:.3f}',
                linewidth=2,
                markersize=5
            )
    
    plt.xlabel('qT', fontsize=14)
    plt.ylabel('S(qT, x1, x2)', fontsize=14)
    plt.title(f'S vs qT for Different x1 Values (x2={selected_x2:.3f}, QM={median_QM:.2f})', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f'{FIGURE_FOLDER}/S_lines_qT_x1.png', dpi=300, bbox_inches='tight')
    print(f"Saved qT-x1 lines plot to {FIGURE_FOLDER}/S_lines_qT_x1.png")

# Main execution
if __name__ == "__main__":
    print("Generating visualizations for SqT model output...")
    
    try:
        # Generate all plots
        print("1. Creating S vs qT plot...")
        plot_S_vs_qT()
        
        print("2. Creating 3D surface plot (if 3D plotting is available)...")
        plot_S_surface_qT_x1()
        
        # Add 2D alternative to 3D surface plot
        print("3. Creating 2D alternative to 3D surface plot...")
        plot_S_lines_qT_x1()
        
        print("4. Creating contour plot...")
        plot_S_contour_x1_x2()
        
        print("5. Creating heatmap...")
        plot_S_heatmap_qT_xF()
        
        print("6. Creating x1/x2 ratio plot...")
        plot_S_vs_x1_x2_ratio()
        
        print(f"All visualizations completed and saved to {FIGURE_FOLDER}/")
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()