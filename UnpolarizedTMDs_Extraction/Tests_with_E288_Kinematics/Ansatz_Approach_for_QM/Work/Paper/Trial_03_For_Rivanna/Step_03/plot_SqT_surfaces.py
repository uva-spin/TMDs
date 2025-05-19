import pandas as pd
import plotly.graph_objects as go
import os
import numpy as np

CSV_FOLDER = "comp_results_csvs"
PLOTS_FOLDER = "HTML_plots"
os.makedirs(PLOTS_FOLDER, exist_ok=True)

def plot_3d_multiple_surfaces(data_info, x_col, y_col, z_col, title, filename):
    fig = go.Figure()

    for csv_file, surface_name, color in data_info:
        df = pd.read_csv(os.path.join(CSV_FOLDER, csv_file))
        x_vals = np.sort(df[x_col].unique())
        y_vals = np.sort(df[y_col].unique())

        X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')

        Z = df[z_col].values.reshape(len(x_vals), len(y_vals))

        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            colorscale=[[0, color], [1, color]],
            cmin=Z.min(), cmax=Z.max(),
            name=surface_name,
            showscale=False,
            opacity=0.9
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(itemsizing='constant')
    )

    output_path = os.path.join(PLOTS_FOLDER, filename)
    fig.write_html(output_path)
    print(f"{title} saved to {output_path}")


def plot_3d_comparison(data_info, x_col, y_col, z_1, z_2, title, filename):
    fig = go.Figure()

    for csv_file, surface_name, colors in data_info:
        df = pd.read_csv(os.path.join(CSV_FOLDER, csv_file))
        x_vals = np.sort(df[x_col].unique())
        y_vals = np.sort(df[y_col].unique())

        X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')

        Z1 = df[z_1].values.reshape(len(x_vals), len(y_vals))
        Z2 = df[z_2].values.reshape(len(x_vals), len(y_vals))

        # Surface for z_1
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z1,
            colorscale=[[0, colors[0]], [1, colors[0]]],
            cmin=Z1.min(), cmax=Z1.max(),
            name=f"{surface_name} - {z_1}",
            showscale=False,
            opacity=0.9
        ))

        # Surface for z_2
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z2,
            colorscale=[[0, colors[1]], [1, colors[1]]],
            cmin=Z2.min(), cmax=Z2.max(),
            name=f"{surface_name} - {z_2}",
            showscale=False,
            opacity=0.9
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title="Value"
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(itemsizing='constant')
    )

    output_path = os.path.join(PLOTS_FOLDER, filename)
    fig.write_html(output_path)
    print(f"{title} saved to {output_path}")


def main():
    data_info = [
        ("SqT_xb_02.csv", "xₛ = 0.2", "yellow"),
        ("SqT_xb_04.csv", "xₛ = 0.4", "orange"),
        ("SqT_xb_06.csv", "xₛ = 0.6", "red")
    ]

    data_info_comp = [
        ("SqT_xb_02.csv", "xₛ = 0.2", ["yellow", "gold"]),
        ("SqT_xb_04.csv", "xₛ = 0.4", ["orange", "darkorange"]),
        ("SqT_xb_06.csv", "xₛ = 0.6", ["red", "darkred"])
    ]


    plot_3d_multiple_surfaces(data_info,
                              x_col='x1',
                              y_col='qT',
                              z_col='SqT_true_mean',
                              title='SqT from CS models for x2 = 0.2, 0.4, 0.6',
                              filename='SqT_True_surface.html')
    
    plot_3d_multiple_surfaces(data_info,
                              x_col='x1',
                              y_col='qT',
                              z_col='SqT_pred_mean',
                              title='SqT from integral models for x2 = 0.2, 0.4, 0.6',
                              filename='SqT_Pred_surface.html')
    
    plot_3d_comparison(data_info_comp,
                              x_col='x1',
                              y_col='qT',
                              z_1='SqT_true_mean',
                              z_2='SqT_pred_mean',
                              title='SqT from integral models for x2 = 0.2, 0.4, 0.6',
                              filename='SqT_comparison_surface.html')

if __name__ == "__main__":
    main()
