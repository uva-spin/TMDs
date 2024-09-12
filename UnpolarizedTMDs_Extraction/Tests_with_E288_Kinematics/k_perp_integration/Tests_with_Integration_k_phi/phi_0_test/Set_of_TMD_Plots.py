import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
#import mpld3  # Library to save matplotlib plots as interactive HTML files
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting toolkit


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

models_path = '/home/ishara/Documents/TMDs/Tests_with_Pseudo_data/Tests_with_E288_Kinematics/k_perp_integration/Tests_with_k_phi/phi_0_test'
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
    ax.legend()

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


### PDFs Comparison ###


#def Generate_PDFs(model, x, qM, N_values):
#    k_values = tf.linspace(0.0, k_upper, N_values)
#    dk = k_values[1] - k_values[0]
#    product_sum = []  
#    for k_val in k_values:
#        concatenated_inputs = np.column_stack((x, k_val, qM))      
#        model_output = model(concatenated_inputs)
#        product_sum.append(model_output)
#    tmd_product_sum = tf.reduce_sum(product_sum, axis=0) * dk
#    tmd_product_sum = np.array(tmd_product_sum) 
#    return tmd_product_sum.flatten()[0]

def Generate_PDFs(model, x, qM, N_values):
    #k_upper = 6.0  # Upper bound for k-values
    k_values = tf.linspace(0.0, k_upper, N_values)
    dk = k_values[1] - k_values[0]
    product_sum = []  
    for k_val in k_values:
        concatenated_inputs = np.column_stack((x, k_val, qM))      
        model_output = model(concatenated_inputs)
        product_sum.append(tf.multiply(k_val,model_output))
    tmd_product_sum = tf.reduce_sum(product_sum, axis=0) * dk
    tmd_product_sum = np.array(tmd_product_sum) 
    return tmd_product_sum.flatten()[0]


def Generate_PDFs_Comparison_Plots(df, modelsArray, num_replicas, output_folder):
    x1vals = df['x1']
    x2vals = df['x2']
    QMvals = df['QM']
    
    fu_xA = df['fu_xA']
    fubar_xB = df['fubar_xB']
    fu_xB = df['fu_xB']
    fubar_xA = df['fubar_xA']
    
    # Store predictions
    predfuxA, predfubarxB, predfuxB, predfubarxA = [], [], [], []
    
    # Iterate over models and replicas
    for i in range(num_replicas):
        model = modelsArray[i]
        modnnu = model.get_layer('nnu')
        modnnubar = model.get_layer('nnubar')
        tempfuxA, tempfubarxB, tempfuxB, tempfubarxA = [], [], [], []
        for j in range(len(x1vals)):
            tempxA, tempxB = x1vals[j], x2vals[j]
            tempQM = QMvals[j]
            tempfuxA.append(Generate_PDFs(modnnu, tempxA, tempQM, len(x1vals)))
            tempfubarxB.append(Generate_PDFs(modnnubar, tempxB, tempQM, len(x1vals)))
            tempfuxB.append(Generate_PDFs(modnnu, tempxB, tempQM, len(x1vals)))
            tempfubarxA.append(Generate_PDFs(modnnubar, tempxA, tempQM, len(x1vals)))
        predfuxA.append(tempfuxA)
        predfubarxB.append(tempfubarxB)
        predfuxB.append(tempfuxB)
        predfubarxA.append(tempfubarxA)
    
    # Averages and standard deviations for each
    tempfu_mean = np.mean(np.array(predfuxA), axis=0)
    tempfubar_mean = np.mean(np.array(predfubarxB), axis=0)
    tempfu_mean_rev = np.mean(np.array(predfuxB), axis=0)
    tempfubar_mean_rev = np.mean(np.array(predfubarxA), axis=0)

    # Create 4 subplots in one figure
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Plot fxA vs xA
    axs[0, 0].plot(x1vals, fu_xA, 'b.', label='Actual fxA')
    axs[0, 0].plot(x1vals, tempfu_mean, 'r.', label='Predicted fxA')
    axs[0, 0].set_title('fxA vs xA')
    axs[0, 0].legend()

    # Plot fbxB vs xB
    axs[0, 1].plot(x2vals, fubar_xB, 'b.', label='Actual fbxB')
    axs[0, 1].plot(x2vals, tempfubar_mean, 'r.', label='Predicted fbxB')
    axs[0, 1].set_title('fbxB vs xB')
    axs[0, 1].legend()

    # Plot fxB vs xB
    axs[1, 0].plot(x2vals, fu_xB, 'b.', label='Actual fxB')
    axs[1, 0].plot(x2vals, tempfu_mean_rev, 'r.', label='Predicted fxB')
    axs[1, 0].set_title('fxB vs xB')
    axs[1, 0].legend()

    # Plot fbxA vs xA
    axs[1, 1].plot(x1vals, fubar_xA, 'b.', label='Actual fbxA')
    axs[1, 1].plot(x1vals, tempfubar_mean_rev, 'r.', label='Predicted fbxA')
    axs[1, 1].set_title('fbxA vs xA')
    axs[1, 1].legend()

    # Adjust layout
    plt.tight_layout()

    # Save the figure in the specified folder
    plot_path = os.path.join(output_folder, 'PDF_Comparison_Plots.pdf')
    plt.savefig(plot_path)
    print(f"PDF Comparison plots saved at: {plot_path}")
    plt.close()

# Usage
# output_folder = 'TMDs_Comparison_Plots'
Generate_PDFs_Comparison_Plots(df, modelsArray, num_replicas=numreplicas, output_folder=str(plots_folder))




'''
def plot_cross_section_comparison_3D(df, predictions, folder_name):
    # Extract the necessary values from the DataFrame
    x1Vals = df['x1'].values
    pTVals = df['pT'].values
    A_vals_actual = df['A'].values  # Actual A values
    
    # Create a new figure for 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the actual data points
    ax.scatter(x1Vals, pTVals, A_vals_actual, c='r', marker='o', label='Actual')
    
    # Plot the predicted data points
    ax.scatter(x1Vals, pTVals, predictions, c='b', marker='^', label='Predicted')
    
    # Set axis labels and plot title
    ax.set_xlabel('x1')
    ax.set_ylabel('pT')
    ax.set_zlabel('A')
    ax.set_title('Actual vs Predicted Cross-Section')
    
    # Add legend
    ax.legend()
    
    # Save the figure in the specified folder
    plot_path = os.path.join(folder_name, 'Actual_vs_Predicted_cross-section.pdf')
    plt.savefig(plot_path)
    print(f"Cross-section comparison plot saved at: {plot_path}")
    plt.close()

# Example of usage after predictions are made
model = modelsArray[0]
predictions = model.predict([df['x1'], df['x2'], df['pT'], df['QM']])[0]
plot_cross_section_comparison_3D(df, predictions, str(plots_folder))
'''
