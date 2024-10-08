import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
#import mpld3  # Library to save matplotlib plots as interactive HTML files
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting toolkit


k_lower = 0.001
k_upper = 6.0

def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")

# Folder to store Comparison comparison plots
plots_folder = 'Comparison_Comparison_Plots'
create_folders(str(plots_folder))

models_path = '/scratch/cee9hc/Unpolarized_TMD/E288/flavor_1/Phase_2/Test_05_k_0-001_6_phi_0'
Models_folder = str(models_path) + '/' + 'DNNmodels'
folders_array = os.listdir(Models_folder)
numreplicas = len(folders_array)

def mse_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    return mse_loss

modelsArray = []
for i in range(numreplicas):
    testmodel = tf.keras.models.load_model(
        str(Models_folder) + '/' + str(folders_array[i]),
        custom_objects={'mse_loss': mse_loss}
    )
    modelsArray.append(testmodel)

modelsArray = np.array(modelsArray)

########### Import pseudodata file 
df = pd.read_csv('E288_pseudo_data.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')

#### S(k) used for pseudo-data ####

def Skq(k):
    return np.exp(-4 * k**2 / (4 * k**2 + 4))

def Skqbar(k):
    return np.exp(-4 * k**2 / (4 * k**2 + 1))

def Generate_Comparison_Data(df, num_replicas, k_values):
    x1vals = df['x1']
    x2vals = df['x2']
    QMvals = df['QM']

    fu_xA = df['fu_xA']
    fubar_xB = df['fubar_xB']

    fu_xB = df['fu_xB']
    fubar_xA = df['fubar_xA']

    true_fxAk = []
    true_fbxBk = []
    true_fxBk = []
    true_fbxAk = []

    pred_fxAk = []
    pred_fbxBk = []
    pred_fxBk = []
    pred_fbxAk = []

    for kv in k_values:
        Kvals = np.linspace(kv, kv, len(x1vals))

        # Calculate true values
        true_fxAk.append(fu_xA * Skq(Kvals))
        true_fbxBk.append(fubar_xB * Skqbar(Kvals))
        true_fxBk.append(fu_xB * Skq(Kvals))
        true_fbxAk.append(fubar_xA * Skqbar(Kvals))

        # Prepare model input
        concat_inputs_xA = np.column_stack((x1vals, Kvals, QMvals))
        concat_inputs_xB = np.column_stack((x2vals, Kvals, QMvals))

        tempfu = []
        tempfubar = []
        tempfu_rev = []
        tempfubar_rev = []

        for i in range(num_replicas):
            t = modelsArray[i]
            modnnu = t.get_layer('nnu')
            modnnubar = t.get_layer('nnubar')

            tempfu.append(list(modnnu.predict(concat_inputs_xA)))
            tempfubar.append(list(modnnubar.predict(concat_inputs_xB)))

            tempfu_rev.append(list(modnnu.predict(concat_inputs_xA)))
            tempfubar_rev.append(list(modnnubar.predict(concat_inputs_xB)))

        # Calculate predicted values
        pred_fxAk.append(np.array(tempfu).mean(axis=0))
        pred_fbxBk.append(np.array(tempfubar).mean(axis=0))
        pred_fxBk.append(np.array(tempfu_rev).mean(axis=0))
        pred_fbxAk.append(np.array(tempfubar_rev).mean(axis=0))

    return (
        np.array(k_values), 
        np.array(true_fxAk), np.array(pred_fxAk), 
        np.array(true_fbxBk), np.array(pred_fbxBk),
        np.array(true_fxBk), np.array(pred_fxBk),
        np.array(true_fbxAk), np.array(pred_fbxAk)
    )


def create_3D_Comparison_dfs(df, numreplicas, k_values):
    Kvals, true_fxAk, pred_fxAk, true_fbxBk, pred_fbxBk, true_fxBk, pred_fxBk, true_fbxAk, pred_fbxAk = Generate_Comparison_Data(df, numreplicas, k_values)
    
    return {
        'fxAk_true': pd.DataFrame({'xA': np.tile(df['x1'], len(k_values)), 'k': np.repeat(Kvals, len(df['x1'])), 'fxk': true_fxAk.flatten()}),
        'fxAk_pred': pd.DataFrame({'xA': np.tile(df['x1'], len(k_values)), 'k': np.repeat(Kvals, len(df['x1'])), 'fxk': pred_fxAk.flatten()}),
        'fbxBk_true': pd.DataFrame({'xB': np.tile(df['x2'], len(k_values)), 'k': np.repeat(Kvals, len(df['x2'])), 'fxk': true_fbxBk.flatten()}),
        'fbxBk_pred': pd.DataFrame({'xB': np.tile(df['x2'], len(k_values)), 'k': np.repeat(Kvals, len(df['x2'])), 'fxk': pred_fbxBk.flatten()}),
        'fxBk_true': pd.DataFrame({'xB': np.tile(df['x2'], len(k_values)), 'k': np.repeat(Kvals, len(df['x2'])), 'fxk': true_fxBk.flatten()}),
        'fxBk_pred': pd.DataFrame({'xB': np.tile(df['x2'], len(k_values)), 'k': np.repeat(Kvals, len(df['x2'])), 'fxk': pred_fxBk.flatten()}),
        'fbxAk_true': pd.DataFrame({'xA': np.tile(df['x1'], len(k_values)), 'k': np.repeat(Kvals, len(df['x1'])), 'fxk': true_fbxAk.flatten()}),
        'fbxAk_pred': pd.DataFrame({'xA': np.tile(df['x1'], len(k_values)), 'k': np.repeat(Kvals, len(df['x1'])), 'fxk': pred_fbxAk.flatten()})
    }


def plot_3D_comparison_matplotlib(df_true, df_pred, title, x_label, y_label, z_label, file_name, save_html=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot true values
    ax.scatter(df_true[x_label], df_true[y_label], df_true['fxk'], c='blue', label='True')

    # Plot predicted values
    ax.scatter(df_pred[x_label], df_pred[y_label], df_pred['fxk'], c='red', label='Predicted')

    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.set_title(title)

    # Show legend
    ax.legend()

    # Save plot as an image file
    plt.savefig(os.path.join(plots_folder, file_name + ".png"))
    print(f"PNG plot saved as {file_name}.png in {plots_folder}")

    # Optionally save the plot as an interactive HTML file
    if save_html:
        html_file = os.path.join(plots_folder, file_name + ".html")
        mpld3.save_html(fig, html_file)
        print(f"HTML plot saved as {file_name}.html in {plots_folder}")

    plt.close()


# Generate and save comparison data for k values from 0.0 to 6.0
k_values = np.arange(0.0, 6.1, 1.0)  # Generate K values from 0.0 to 6.0 with step 1.0
df_comparison = create_3D_Comparison_dfs(df, numreplicas, k_values)

# Save 3D plots as PNG and HTML files
plot_3D_comparison_matplotlib(df_comparison['fxAk_true'], df_comparison['fxAk_pred'], 
                              'fxAk True vs Predicted', 'xA', 'k', 'fxk', 'fxAk_true_vs_pred', save_html=False)

plot_3D_comparison_matplotlib(df_comparison['fbxBk_true'], df_comparison['fbxBk_pred'], 
                              'fbxBk True vs Predicted', 'xB', 'k', 'fxk', 'fbxBk_true_vs_pred', save_html=False)

plot_3D_comparison_matplotlib(df_comparison['fxBk_true'], df_comparison['fxBk_pred'], 
                              'fxBk True vs Predicted', 'xB', 'k', 'fxk', 'fxBk_true_vs_pred', save_html=False)

plot_3D_comparison_matplotlib(df_comparison['fbxAk_true'], df_comparison['fbxAk_pred'], 
                              'fbxAk True vs Predicted', 'xA', 'k', 'fxk', 'fbxAk_true_vs_pred', save_html=False)



# Function to create 2D subplots for comparisons
def plot_2D_comparison_matplotlib(df_true, df_pred, x_label_1, x_label_2, y_label, title, file_name):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot 1: y vs k
    axs[0].scatter(df_true['k'], df_true['fxk'], c='blue', label='True')
    axs[0].scatter(df_pred['k'], df_pred['fxk'], c='red', label='Predicted')
    axs[0].set_xlabel('k')
    axs[0].set_ylabel(y_label)
    axs[0].set_title(f'{title} vs k')
    axs[0].legend()

    # Plot 2: y vs x (either xA or xB)
    axs[1].scatter(df_true[x_label_2], df_true['fxk'], c='blue', label='True')
    axs[1].scatter(df_pred[x_label_2], df_pred['fxk'], c='red', label='Predicted')
    axs[1].set_xlabel(x_label_2)
    axs[1].set_ylabel(y_label)
    axs[1].set_title(f'{title} vs {x_label_2}')
    axs[1].legend()

    # Adjust layout and save plot
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, file_name + ".png"))
    print(f"2D comparison plot saved as {file_name}.png in {plots_folder}")
    plt.close()


# Add 2D plots for each case in addition to the 3D plots
def create_2D_plots(df_comparison):
    # fxAk_ vs k and vs xA
    plot_2D_comparison_matplotlib(
        df_comparison['fxAk_true'], df_comparison['fxAk_pred'],
        x_label_1='k', x_label_2='xA', y_label='fxk',
        title='fxAk True vs Predicted', file_name='fxAk_true_vs_pred_2D'
    )

    # fbxBk_ vs k and vs xB
    plot_2D_comparison_matplotlib(
        df_comparison['fbxBk_true'], df_comparison['fbxBk_pred'],
        x_label_1='k', x_label_2='xB', y_label='fxk',
        title='fbxBk True vs Predicted', file_name='fbxBk_true_vs_pred_2D'
    )

    # fxBk_ vs k and vs xB
    plot_2D_comparison_matplotlib(
        df_comparison['fxBk_true'], df_comparison['fxBk_pred'],
        x_label_1='k', x_label_2='xB', y_label='fxk',
        title='fxBk True vs Predicted', file_name='fxBk_true_vs_pred_2D'
    )

    # fbxAk_ vs k and vs xA
    plot_2D_comparison_matplotlib(
        df_comparison['fbxAk_true'], df_comparison['fbxAk_pred'],
        x_label_1='k', x_label_2='xA', y_label='fxk',
        title='fbxAk True vs Predicted', file_name='fbxAk_true_vs_pred_2D'
    )


# Calling the 2D plotting function
create_2D_plots(df_comparison)
