import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from matplotlib import rcParams
#from functions_and_constants import *

eu2 = (2/3)**2
ed2 = (-1/3)**2
es2 = (-1/3)**2
alpha = 1/137
hc_factor = 3.89 * 10**8
factor = ((4*np.pi*alpha)**2)/(9*2*np.pi)

import matplotlib.font_manager as fm

# Set Times New Roman as the global font
plt.rcParams["font.family"] = "Times New Roman"

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"


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
E288_300_results = pd.read_csv("Results_csvs/E288_300_results.csv")
E288_400_results = pd.read_csv("Results_csvs/E288_400_results.csv")
E605_results = pd.read_csv("Results_csvs/E605_results.csv")



E288_200_with_cuts = pd.read_csv('Results_csvs/E288_200_results_with_cuts.csv')
E288_300_with_cuts = pd.read_csv('Results_csvs/E288_300_results_with_cuts.csv')
E288_400_with_cuts = pd.read_csv('Results_csvs/E288_400_results_with_cuts.csv')
E605_with_cuts = pd.read_csv('Results_csvs/E605_results_with_cuts.csv')



functions_results = pd.read_csv("Results_csvs/comparison_results.csv")



def generate_combined_subplots(datasets, filenames, output_filename):
    # Publication-quality settings
    rcParams.update({
        'font.family': 'serif',
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'text.usetex': False,
        'figure.dpi': 300
    })

    num_rows = len(datasets)
    num_cols_per_row = [len(np.unique(df['QM'])) for df in datasets]
    max_cols = max(num_cols_per_row)
    
    fig, axes = plt.subplots(nrows=num_rows, ncols=max_cols, figsize=(3 * max_cols, 4 * num_rows), constrained_layout=True)
    
    if num_rows == 1 and max_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = np.array([axes])
    elif max_cols == 1:
        axes = np.array([[ax] for ax in axes])

    for row, (df, title, num_cols) in enumerate(zip(datasets, filenames, num_cols_per_row)):
        unique_QM = np.unique(df['QM'])
        for col, QM_val in enumerate(unique_QM):
            ax = axes[row, col]
            ax.tick_params(labelsize=20)
            mask = df['QM'] == QM_val

            # Plotting
            ax.errorbar(df['qT'][mask], df['A_true'][mask], yerr=df['A_true_err'][mask],
                        fmt='o', color='blue', label='Experiment', capsize=3, markersize=5, linewidth=1.2)
            ax.errorbar(df['qT'][mask], df['A_pred'][mask], yerr=df['A_pred_err'][mask],
                        fmt='x', color='red', label='DNN', capsize=3, markersize=5, linewidth=1.2)

            # # Plotting
            # ax.errorbar(df['qT'][mask], df['A_true'][mask], yerr=df['A_true_err'][mask],
            #             fmt='o', color='blue', label='Experiment', capsize=3, markersize=5, linewidth=1.2)

            # DNN: plot mean line + fill between error
            ax.plot(df['qT'][mask], df['A_pred'][mask], 'x-', color='red', markersize=5, linewidth=1.2)
            ax.fill_between(df['qT'][mask],
                            df['A_pred'][mask] - df['A_pred_err'][mask],
                            df['A_pred'][mask] + df['A_pred_err'][mask],
                            color='red', alpha=0.1, linewidth=0)


            ax.set_xlabel(r'$q_T$ [GeV]',fontsize=20)
            if col == 0:
                ax.set_ylabel(r'$E \, \mathrm{d}\sigma/\mathrm{d}^3q$',fontsize=20)

            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            ax.text(0.2, 0.88, fr'$Q_M$ = {QM_val:.2f}', transform=ax.transAxes,
                    fontsize=18, verticalalignment='top', horizontalalignment='left')
            

            if col == 0 and col == 0:
                ax.text(0.95, 0.98, title, transform=ax.transAxes,
                        fontsize=14, fontweight='bold', va='top', ha='right')

            if row == 0 and col == 4:
                ax.legend(loc='upper left', bbox_to_anchor=(0.15, 0.7), frameon=True)


        # Hide unused axes
        for col in range(num_cols, max_cols):
            fig.delaxes(axes[row, col])

    plt.savefig(f"{output_filename}.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"Subplots saved successfully in {output_filename}.pdf")


E288_datasets = [E288_200_results, E288_300_results, E288_400_results]
E288_filenames = ['E288 200 GeV', 'E288 300 GeV', 'E288 400 GeV']
generate_combined_subplots(E288_datasets, E288_filenames, f'{results_folder}/E288_combined_cross_sections')


E_288_datasets_with_cuts = [E288_200_with_cuts, E288_300_with_cuts, E288_400_with_cuts]
E_288_filenames_with_cuts = ['E288 200 GeV', 'E288 300  GeV', 'E288 400  GeV']
generate_combined_subplots(E_288_datasets_with_cuts, E_288_filenames_with_cuts, f'{results_folder}/E288_combined_cross_sections_with_cuts')


E605_datasets = [E605_results]
E605_filenames = ['E605']
generate_combined_subplots(E605_datasets, E605_filenames, f'{results_folder}/E605_combined_cross_sections')


E605_datasets_with_cuts = [E605_with_cuts]
E605_filenames_with_cuts = ['E605']
generate_combined_subplots(E605_datasets_with_cuts, E605_filenames_with_cuts, f'{results_folder}/E605_combined_cross_sections_with_cuts')


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


PDFs_x1x2 = np.array(functions_results['PDFs_x1x2'])
PDFs_x2x1 = np.array(functions_results['PDFs_x2x1'])
PDFs = np.array(functions_results['PDFs'])


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



# ####### Plot SqT ############
plt.figure(5,figsize=(10, 6))
#plt.plot(QM, PDFs_x1x2, label='PDFs_x1x2', linestyle='-', color='blue')
#plt.plot(QM, PDFs_x2x1, label='PDFs_x2x1', linestyle='-', color='green')
plt.plot(QM, PDFs, label='PDFs', linestyle='-', color='red')
plt.xlabel(r'$Q_M$', fontsize=14)
plt.ylabel(r'PDFs', fontsize=14)
plt.title('PDFs vs $Q_M$', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.yscale('log')
plt.savefig(f"{results_folder}/PDFs_vs_QM_plot.pdf")
plt.close()

