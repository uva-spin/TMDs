import pandas as pd
import plotly.graph_objects as go
import os
import numpy as np

CSV_FOLDER = "csvs"
PLOTS_FOLDER = "HTML_plots"
os.makedirs(PLOTS_FOLDER, exist_ok=True)

def plot_3d_with_error_surfaces(df, x_col, y_col, z_avg_col, z_err_col, title, filename):
    # Pivot data into 2D arrays for surface plotting
    x_vals = np.sort(df[x_col].unique())
    y_vals = np.sort(df[y_col].unique())

    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')  # shape: (len(x_vals), len(y_vals))

    def reshape_z(col):
        return df[col].values.reshape(len(x_vals), len(y_vals))

    Z_avg = reshape_z(z_avg_col)
    Z_std = reshape_z(z_err_col)
    Z_min = Z_avg - Z_std
    Z_max = Z_avg + Z_std

    fig = go.Figure()

    # Avg surface (yellow)
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z_avg,
        colorscale=[[0, 'yellow'], [1, 'yellow']],
        cmin=Z_avg.min(), cmax=Z_avg.max(),
        colorbar=dict(title='Average'),
        name='Avg',
        showscale=True,
        opacity=0.9
    ))

    # Min surface (blue)
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z_min,
        colorscale=[[0, 'blue'], [1, 'blue']],
        cmin=Z_min.min(), cmax=Z_min.max(),
        name='Min',
        showscale=False,
        opacity=0.3
    ))

    # Max surface (red)
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z_max,
        colorscale=[[0, 'red'], [1, 'red']],
        cmin=Z_max.min(), cmax=Z_max.max(),
        name='Max',
        showscale=False,
        opacity=0.3
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_avg_col
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(itemsizing='constant')
    )

    output_path = os.path.join(PLOTS_FOLDER, filename)
    fig.write_html(output_path)
    print(f"{title} saved to {output_path}")



def main():
    df = pd.read_csv(os.path.join(CSV_FOLDER, "n1n2_grid.csv"))

    # Plot n1 (a) with error surfaces
    plot_3d_with_error_surfaces(df,
                                x_col='x',
                                y_col='k',
                                z_avg_col='nna_mean',
                                z_err_col='nna_std',
                                title='n₁(x, k) with error surfaces',
                                filename='nna_x_k_surface.html')

    # Plot n2 (b) with error surfaces
    plot_3d_with_error_surfaces(df,
                                x_col='x',
                                y_col='k',
                                z_avg_col='nnb_mean',
                                z_err_col='nnb_std',
                                title='n₂(x, k) with error surfaces',
                                filename='nnb_x_k_surface.html')


if __name__ == "__main__":
    main()
