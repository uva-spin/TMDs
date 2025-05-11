import pandas as pd
import plotly.graph_objects as go
import os


CSV_FOLDER = "csvs"
PLOTS_FOLDER = "HTML_plots"
os.makedirs(PLOTS_FOLDER, exist_ok=True)

def plot_3d(df, x_col, y_col, z_col, title, filename):
    fig = go.Figure(data=[go.Scatter3d(
        x=df[x_col],
        y=df[y_col],
        z=df[z_col],
        mode='markers',
        marker=dict(
            size=3,
            color=df[z_col],
            colorscale='Viridis',
            opacity=0.8
        )
    )])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    output_path = os.path.join(PLOTS_FOLDER, filename)
    fig.write_html(output_path)
    print(f"{title} saved to {output_path}")


def main():

    # Plot 2: n1 vs x1, k
    n1_file = os.path.join(CSV_FOLDER, "n1n2_grid.csv")
    df_n1 = pd.read_csv(n1_file)
    plot_3d(df_n1, x_col='x', y_col='k', z_col='n1',
            title="n1(x1, k)", filename="n1_x1_k.html")

    # Plot 3: n2 vs x2, k
    n2_file = os.path.join(CSV_FOLDER, "n1n2_grid.csv")
    df_n2 = pd.read_csv(n2_file)
    plot_3d(df_n2, x_col='x', y_col='k', z_col='n2',
            title="n2(x2, k)", filename="n2_x2_k.html")


if __name__ == "__main__":
    main()

