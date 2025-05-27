import pandas as pd
import plotly.graph_objects as go
import os
import numpy as np

CSV_FOLDER = "Results_csvs"
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




def main():
    data_info_fixed_x2 = [
        ("SqT_x2_20.csv", "x2 = 0.2", "yellow"),
        ("SqT_x2_40.csv", "x2 = 0.4", "orange"),
        ("SqT_x2_60.csv", "x2 = 0.6", "red")
    ]

    data_info_fixed_x1 = [
        ("SqT_x1_20.csv", "x2 = 0.2", "yellow"),
        ("SqT_x1_40.csv", "x2 = 0.4", "orange"),
        ("SqT_x1_60.csv", "x2 = 0.6", "red")
    ]


    plot_3d_multiple_surfaces(data_info_fixed_x2,
                              x_col='x1',
                              y_col='qT',
                              z_col='SqT_true_mean',
                              title='SqT from CS models for x2 = 0.2, 0.4, 0.6',
                              filename='SqT_Surface_fixed_x2.html')
    
    plot_3d_multiple_surfaces(data_info_fixed_x1,
                              x_col='x2',
                              y_col='qT',
                              z_col='SqT_true_mean',
                              title='SqT from CS models for x1 = 0.2, 0.4, 0.6',
                              filename='SqT_Surface_fixed_x1.html')
    

if __name__ == "__main__":
    main()
