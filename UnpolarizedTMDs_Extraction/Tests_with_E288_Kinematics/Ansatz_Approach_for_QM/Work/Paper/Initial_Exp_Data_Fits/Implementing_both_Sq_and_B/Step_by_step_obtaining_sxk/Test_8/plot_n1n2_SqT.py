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




def plot_3d_with_true_pred(df, x_col, y_col, z_true_col='SqT_true', z_pred_col='SqT_pred', title='SqT_true vs SqT_pred', filename='S_true_vs_S_pred.html'):
    fig = go.Figure()

    # Add S_true points
    fig.add_trace(go.Scatter3d(
        x=df[x_col],
        y=df[y_col],
        z=df[z_true_col],
        mode='markers',
        name='S_true',
        marker=dict(
            size=3,
            color='blue',
            opacity=0.7,
            symbol='circle'
        )
    ))

    # Add S_pred points
    fig.add_trace(go.Scatter3d(
        x=df[x_col],
        y=df[y_col],
        z=df[z_pred_col],
        mode='markers',
        name='S_pred',
        marker=dict(
            size=3,
            color='red',
            opacity=0.7,
            symbol='diamond'
        )
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title='S',
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=0, y=1)
    )

    output_path = os.path.join(PLOTS_FOLDER, filename)
    fig.write_html(output_path)
    print(f"{title} saved to {output_path}")




def main():

    # Plot 1: n1 vs x1, k
    n1_file = os.path.join(CSV_FOLDER, "n1n2_grid.csv")
    df_n1 = pd.read_csv(n1_file)
    plot_3d(df_n1, x_col='x', y_col='k', z_col='n1',
            title="n1(x1, k)", filename="n1_x1_k.html")

    # Plot 2: n2 vs x2, k
    n2_file = os.path.join(CSV_FOLDER, "n1n2_grid.csv")
    df_n2 = pd.read_csv(n2_file)
    plot_3d(df_n2, x_col='x', y_col='k', z_col='n2',
            title="n2(x2, k)", filename="n2_x2_k.html")


    SqT_file = os.path.join(CSV_FOLDER, "SqT_comp.csv")
    df_SqT = pd.read_csv(SqT_file)

    # Plot 3: SqT vs x1, qT
    plot_3d_with_true_pred(df_SqT, x_col='x1', y_col='qT',
            title="SqT with x1 and qT", filename="SqT_x1_qT.html")
    
    # Plot 4: SqT vs x2, qT
    plot_3d_with_true_pred(df_SqT, x_col='x2', y_col='qT',
            title="SqT with x2 and qT", filename="SqT_x2_qT.html")
    
    # Plot 5: SqT vs x1, x2
    plot_3d_with_true_pred(df_SqT, x_col='x1', y_col='x2',
            title="SqT with x1 and x2", filename="SqT_x1_x2.html")

if __name__ == "__main__":
    main()

