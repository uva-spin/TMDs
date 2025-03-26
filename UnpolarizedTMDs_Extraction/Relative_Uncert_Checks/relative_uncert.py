import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from function import fDNNQ

# Create necessary folders
def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
        
       
       
pseudo_data_folder="plots_and_csvs"
create_folders(pseudo_data_folder)




# Load Data
E288_200 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Test/E288_200.csv")
E288_300 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Test/E288_300.csv")
E288_400 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Test/E288_400.csv")
E605 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Test/E605.csv")
E772 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Test/E772.csv")




def gen_pseudo(df):
    x1_temp = df['x1'].values
    x2_temp = df['x2'].values
    qT_temp = df['qT'].values
    QM_temp = df['QM'].values
    dA_temp = df['dA'].values
    A_temp = df['A'].values
    A_rel_uncrt = dA_temp/A_temp

    results_df = pd.DataFrame({
        'x1': x1_temp,
        'x2': x2_temp,
        'qT': qT_temp,
        'QM': QM_temp,
        'A': A_temp,
        'dA': dA_temp,
        'A_rel_uncert': abs(A_rel_uncrt)
    })

    return pd.DataFrame(results_df)




def gen_plots(df1, filename):
    # Get unique QM values and assign colors
    unique_QM = df1["QM"].unique()
    colormap = cm.get_cmap("tab10", len(unique_QM))
    colors = [colormap(i) for i in range(len(unique_QM))]
    color_map = dict(zip(unique_QM, colors))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for QM in unique_QM:
        subset = df1[df1["QM"] == QM]
        qT1 = subset['qT'].values
        Arel = subset['A_rel_uncert'].values
        
        ax.plot(qT1, Arel, label=f'$Q_M$ = {QM:.2f} GeV', color=color_map[QM])
    
    ax.set_xlabel('qT')
    ax.set_ylabel('A_rel_uncert')
    ax.legend()
    ax.grid(True)
    ax.set_title('A_rel_uncert vs qT for Different QM')
    
    plt.tight_layout()
    plt.savefig(os.path.join(pseudo_data_folder, f"{filename}.pdf"))


E288_200_pseudo = gen_pseudo(E288_200)
E288_200_pseudo.to_csv(os.path.join(pseudo_data_folder,"pseudodata_E288_200.csv"))

E228_300_pseudo = gen_pseudo(E288_300)
E228_300_pseudo.to_csv(os.path.join(pseudo_data_folder,"pseudodata_E288_300.csv"))

E228_400_pseudo = gen_pseudo(E288_400)
E228_400_pseudo.to_csv(os.path.join(pseudo_data_folder,"pseudodata_E288_400.csv"))

E605_pseudo = gen_pseudo(E605)
E605_pseudo.to_csv(os.path.join(pseudo_data_folder,"pseudodata_E605.csv"))

E772_pseudo = gen_pseudo(E772)
E772_pseudo.to_csv(os.path.join(pseudo_data_folder,"pseudodata_E772.csv"))


gen_plots(E288_200_pseudo,"E288_200")
gen_plots(E228_300_pseudo,"E288_300")
gen_plots(E228_400_pseudo,"E288_400")
gen_plots(E605_pseudo,"E605")
gen_plots(E772_pseudo,"E772")


######### B(QM) #################

data = E288_200_pseudo


QM_values = np.array(data['QM'].unique())
fDNNQ_values = fDNNQ(QM_values)

# Extract the first A_rel_uncert value for each QM
# Here we use the first value in qT
A_rel_uncert_first = data.groupby('QM')['A_rel_uncert'].first().reindex(QM_values).values
print(A_rel_uncert_first)
err_B = A_rel_uncert_first * fDNNQ_values

## Plot Analytical vs. Model Predictions
plt.figure(figsize=(10, 6))
plt.plot(QM_values, fDNNQ_values, label=r'Analytical $\mathcal{B}(Q_M)$', linestyle='--', color='blue')
plt.fill_between(QM_values, fDNNQ_values - err_B, fDNNQ_values + err_B, color='blue', alpha=0.2, label='Uncertainty Band')

plt.xlabel(r'$Q_M$', fontsize=14)
plt.ylabel(r'$f_{DNNQ}(Q_M)$', fontsize=14)
plt.title('Comparison of Analytical $\mathcal{B}(Q_M)$ and DNNQ Model', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig(os.path.join(pseudo_data_folder, "QM_comparison_plot.pdf"))

# Generate QM Range for Comparison
# QM_values = np.linspace(data['QM'].min(), data['QM'].max(), 200)
# QM_values = np.array(data['QM'].unique())
# fDNNQ_values = fDNNQ(QM_values)

# ## Plot Analytical vs. Model Predictions
# plt.figure(figsize=(10, 6))
# plt.plot(QM_values, fDNNQ_values, label=r'Analytical $\mathcal{B}(Q_M)$', linestyle='--', color='blue')
# plt.xlabel(r'$Q_M$', fontsize=14)
# plt.ylabel(r'$f_{DNNQ}(Q_M)$', fontsize=14)
# plt.title('Comparison of Analytical $\mathcal{B}(Q_M)$ and DNNQ Model', fontsize=16)
# plt.legend(fontsize=12)
# plt.grid(True)
# plt.savefig(os.path.join(pseudo_data_folder,"QM_comparison_plot.pdf"))
# plt.show()
