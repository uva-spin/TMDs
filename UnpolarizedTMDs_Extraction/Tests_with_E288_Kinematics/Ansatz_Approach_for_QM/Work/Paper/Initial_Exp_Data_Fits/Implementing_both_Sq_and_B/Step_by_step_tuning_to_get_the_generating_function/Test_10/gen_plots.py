import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from functions_and_constants import *



import matplotlib.font_manager as fm

# # Set Times New Roman as the global font
# plt.rcParams["font.family"] = "Times New Roman"

# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["mathtext.fontset"] = "custom"
# plt.rcParams["mathtext.rm"] = "Times New Roman"
# plt.rcParams["mathtext.it"] = "Times New Roman:italic"
# plt.rcParams["mathtext.bf"] = "Times New Roman:bold"



def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")

# Create Results Folder
results_folder = 'plots'
create_folders(results_folder)


# Load Data
E288_200_results = pd.read_csv("Results_csvs/E288_200_results.csv")
E288_400_results = pd.read_csv("Results_csvs/E288_400_results.csv")


E288_200_with_cuts = pd.read_csv('Results_csvs/E288_200_results_with_cuts.csv')
E288_400_with_cuts = pd.read_csv('Results_csvs/E288_400_results_with_cuts.csv')


functions_results = pd.read_csv("Results_csvs/comparison_results.csv")



def generate_combined_subplots(datasets, filenames, output_filename):
    num_rows = len(datasets)
    num_cols_per_row = [len(np.unique(df['QM'])) for df in datasets]
    max_cols = max(num_cols_per_row)
    
    fig, axes = plt.subplots(nrows=num_rows, ncols=max_cols, figsize=(4 * max_cols, 6 * num_rows))
    
    for row, (df, title, num_cols) in enumerate(zip(datasets, filenames, num_cols_per_row)):
        unique_QM = np.unique(df['QM'])
        for col, QM_val in enumerate(unique_QM):
            ax = axes[row, col] if num_rows > 1 else axes[col]
            mask = df['QM'] == QM_val
            ax.errorbar(df['qT'][mask], df['A_true'][mask], yerr=df['A_true_err'][mask], fmt='bo', label='Pseudo-data', capsize=3)
            ax.errorbar(df['qT'][mask], df['A_pred'][mask], yerr=df['A_pred_err'][mask], fmt='rx', label='DNN', capsize=3)
            ax.set_xlabel(r'$q_T$')
            ax.set_ylabel(r'$E\frac{d\sigma}{d^3q}$')
            ax.legend()
            ax.grid(True)
            ax.text(0.05, 0.1, f'QM={QM_val:.2f}', transform=ax.transAxes, fontsize=12, verticalalignment='top')
        
        # Hide empty subplots
        for col in range(num_cols, max_cols):
            fig.delaxes(axes[row, col]) if num_rows > 1 else fig.delaxes(axes[col])
        
        # Add title text in the bottom-left corner of each row
        fig.text(0.05, 0.1 + (num_rows - row - 1) * (1 / num_rows), title, fontsize=14, ha='left', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{output_filename}.pdf")
    plt.close(fig)
    print(f"Subplots saved successfully in {output_filename}.pdf")



datasets = [E288_200_results,E288_400_results]
filenames = ['E288_200','E288_400']
generate_combined_subplots(datasets, filenames, f'{results_folder}/E288_combined_cross_sections')


datasets_with_cuts = [E288_200_with_cuts,E288_400_with_cuts]
filenames_with_cuts = ['E288 200','E288 400']
generate_combined_subplots(datasets_with_cuts, filenames_with_cuts, f'{results_folder}/E288_combined_cross_sections_with_cuts')


# datasets = [E288_200_results, E288_300_results, E288_400_results]
# filenames = ['E288_200', 'E288_300', 'E288_400']
# generate_combined_subplots(datasets, filenames, f'{results_folder}/E288_combined_cross_sections')


# datasets_with_cuts = [E288_200_with_cuts, E288_300_with_cuts, E288_400_with_cuts]
# filenames_with_cuts = ['E288 200', 'E288 300', 'E288 400']
# generate_combined_subplots(datasets_with_cuts, filenames_with_cuts, f'{results_folder}/E288_combined_cross_sections_with_cuts')



# ######### B(QM) #################


# # Generate QM Range for Comparison
# QM_values = np.linspace(data['QM'].min(), data['QM'].max(), 200)
# fDNNQ_values = fDNNQ(QM_values)


# # Get Model Predictions
# dnnQ_contributions = np.array([model.predict(QM_values, verbose=0).flatten() for model in models_list])
# dnnQ_mean = np.mean(dnnQ_contributions, axis=0)
# dnnQ_std = np.std(dnnQ_contributions, axis=0)



# #fDNNQ_values = fDNNQ(QM_values)


Bmean = np.array(functions_results['B_calc_mean'])
Bstd = np.array(functions_results['B_calc_std'])

qT = np.array(functions_results['qT'])
QM = np.array(functions_results['QM'])

B2mean = np.array(functions_results['B2_calc_mean'])
B2std = np.array(functions_results['B2_calc_std'])

SqTmean = np.array(functions_results['SqT_mean'])
SqTstd = np.array(functions_results['SqT_std'])

SB2mean = np.array(functions_results['SB2mean'])
SB2std = np.array(functions_results['SB2std'])




# ####### Plot SqT ############
plt.figure(1,figsize=(10, 6))
plt.plot(qT, SqTmean, label='$\mathcal{S}(q_T)$ DNN model (mean)', linestyle='-', color='red')
plt.fill_between(qT, SqTmean - SqTstd, SqTmean + SqTstd, color='red', alpha=0.2, label='$\mathcal{S}(q_T)$ DNN model (std)')
plt.xlabel(r'$q_T$', fontsize=14)
plt.ylabel(r'$\mathcal{S}(q_T)$', fontsize=14)
plt.title('$\mathcal{S}(q_T)$ vs $q_T$', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig(f"{results_folder}/SqT_plot.pdf")
plt.close()


# ####### Plot B2(QM) vs QM ############
plt.figure(2,figsize=(10, 6))
#plt.plot(QM, B2true, label=r'$\mathcal{B}^2(Q_M)$ True', linestyle='--', color='blue')
plt.plot(QM, B2mean, label='$\mathcal{B}^2(Q_M)$ DNN model (mean)', linestyle='-', color='red')
plt.fill_between(QM, B2mean - B2std, B2mean + B2std, color='red', alpha=0.2, label='$\mathcal{B}^2(Q_M)$ DNN model (std)')
plt.plot(QM, Bmean, label='$\mathcal{B}(Q_M)$ DNN model (mean)', linestyle='-', color='blue')
plt.fill_between(QM, Bmean - Bstd, Bmean + Bstd, color='blue', alpha=0.2, label='$\mathcal{B}(Q_M)$ DNN model (std)')
plt.xlabel(r'$Q_M$', fontsize=14)
plt.ylabel(r'$\mathcal{B}^2(Q_M)$', fontsize=14)
plt.title('$\mathcal{B}^2(Q_M)$ vs $Q_M$', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig(f"{results_folder}/B2QM_plot.pdf")
plt.close()


# ####### Plot SB2(QM) vs QM ############
plt.figure(3,figsize=(10, 6))
#plt.plot(QM, SB2true, label=r'$\mathcal{S}(q_T)\mathcal{B}^2(Q_M)$ True', linestyle='--', color='blue')
plt.plot(QM, SB2mean, label='$\mathcal{S}(q_T)\mathcal{B}^2(Q_M)$ DNN model (mean)', linestyle='-', color='red')
plt.fill_between(QM, SB2mean - SB2std, SB2mean + SB2std, color='red', alpha=0.2, label='$\mathcal{S}(q_T)\mathcal{B}^2(Q_M)$ DNN model (std)')
plt.xlabel(r'$Q_M$', fontsize=14)
plt.ylabel(r'$\mathcal{S}(q_T)\mathcal{B}^2(Q_M)$', fontsize=14)
plt.title('$\mathcal{S}(q_T)\mathcal{B}^2(Q_M)$ vs $Q_M$', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig(f"{results_folder}/SB2QM_plot.pdf")
plt.close()




# ####### Plot SqT ############
plt.figure(4,figsize=(10, 6))
plt.plot(QM, SqTmean, label='$\mathcal{S}(q_T)$ DNN model (mean)', linestyle='-', color='red')
plt.fill_between(QM, SqTmean - SqTstd, SqTmean + SqTstd, color='red', alpha=0.2, label='$\mathcal{S}(q_T)$ DNN model (std)')
plt.xlabel(r'$Q_M$', fontsize=14)
plt.ylabel(r'$\mathcal{S}(q_T)$', fontsize=14)
plt.title('$\mathcal{S}(q_T)$ vs $Q_M$', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig(f"{results_folder}/SqT_vs_QM_plot.pdf")
plt.close()





