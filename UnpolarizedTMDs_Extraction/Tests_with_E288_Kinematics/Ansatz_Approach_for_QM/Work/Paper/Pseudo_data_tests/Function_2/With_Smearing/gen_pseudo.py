import os
import numpy as np
import matplotlib.pyplot as plt
import lhapdf
import pandas as pd
from function import fDNNQ
#from scipy.integrate import simps

# Create necessary folders
def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
        
       
       
pseudo_data_folder="gen_pseudo_data"
create_folders(pseudo_data_folder)


NNPDF4_nlo = lhapdf.mkPDF('NNPDF40_nlo_as_01180')


# Load Data
E288_200 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Test/E288_200.csv")
E288_300 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Test/E288_300.csv")
E288_400 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Test/E288_400.csv")
E605 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Test/E605.csv")
E772 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Test/E772.csv")

#data = pd.concat([E288_200])

alpha = 1/137

mm = 0.5

def pdf(pdfset, flavor, x, QQ):
    return pdfset.xfxQ(flavor, x, QQ)

def S(k):
    return ((k**2)/(mm*np.pi))*np.exp(-(k**2)/mm)



def QM_int(QM):
    return (-1)/(2*QM**2)

def compute_QM_integrals(QM_array):
    QM_array = np.atleast_1d(QM_array) 
    QM_integrated = QM_int(QM_array + 0.5) - QM_int(QM_array - 0.5)
    
    return QM_integrated[0] if QM_integrated.size == 1 else QM_integrated



def compute_A(x1, x2, qT, QM):
    f_u_x1 = pdf(NNPDF4_nlo, 2, x1, QM) 
    f_ubar_x2 = pdf(NNPDF4_nlo, -2, x2, QM)
    f_u_x2 = pdf(NNPDF4_nlo, 2, x2, QM)
    f_ubar_x1 = pdf(NNPDF4_nlo, -2, x1, QM)
    f_d_x1 = pdf(NNPDF4_nlo, 1, x1, QM) 
    f_dbar_x2 = pdf(NNPDF4_nlo, -1, x2, QM)
    f_d_x2 = pdf(NNPDF4_nlo, 1, x2, QM)
    f_dbar_x1 = pdf(NNPDF4_nlo, -1, x1, QM)
    f_s_x1 = pdf(NNPDF4_nlo, 3, x1, QM) 
    f_sbar_x2 = pdf(NNPDF4_nlo, -3, x2, QM)
    f_s_x2 = pdf(NNPDF4_nlo, 3, x2, QM)
    f_sbar_x1 = pdf(NNPDF4_nlo, -3, x1, QM)

    # # Sk_contribution = (1/2)*(np.pi)*(np.exp(-qT*qT/2))
    Sk_contribution = (8*mm*mm + qT*qT*qT*qT)/(32*np.pi*mm)*(np.exp(-(qT*qT)/(2*mm)))

    fDNN_contribution = fDNNQ(QM)

    # eu2 = (2/3)**2
    # ed2 = (-1/3)**2
    # es2 = (-1/3)**2

    eu2 = 1
    ed2 = 1
    es2 = 1


    ux1ubarx2_term = eu2*f_u_x1*f_ubar_x2
    ubarx1ux2_term = eu2*f_u_x2*f_ubar_x1
    dx1dbarx2_term = ed2*f_d_x1*f_dbar_x2
    dbarx1dx2_term = ed2*f_d_x2*f_dbar_x1
    sx1sbarx2_term = es2*f_s_x1*f_sbar_x2
    sbarx1sx2_term = es2*f_s_x2*f_sbar_x1
    
    PDFs = (ux1ubarx2_term + ubarx1ux2_term + dx1dbarx2_term + dbarx1dx2_term + sx1sbarx2_term + sbarx1sx2_term)
    fDNN = fDNN_contribution
    # factor = qT*((4*np.pi*alpha)**2)/(9*QM*QM*QM)/(2*np.pi*qT)
    # factor = ((4*np.pi*alpha)**2)/(9*QM*QM*QM)/(2*np.pi)
    factor = ((4*np.pi*alpha)**2)/(9*2*np.pi)
    QM_integral = compute_QM_integrals(QM)
    cross_section =  fDNN * factor * PDFs * Sk_contribution * QM_integral
    hc_factor = 3.89*10**8
    cross_section = cross_section * hc_factor 
    return cross_section



def gen_pseudo(df):
    x1_temp = df['x1'].values
    x2_temp = df['x2'].values
    qT_temp = df['qT'].values
    QM_temp = df['QM'].values
    dA_temp = df['dA'].values

    A_temp = np.array([
        compute_A(x1, x2, qT, QM)
        for x1, x2, qT, QM in zip(x1_temp, x2_temp, qT_temp, QM_temp)
    ])

    results_df = pd.DataFrame({
        'x1': x1_temp,
        'x2': x2_temp,
        'qT': qT_temp,
        'QM': QM_temp,
        'A': A_temp,
        'dA': dA_temp
    })

    return pd.DataFrame(results_df)



def gen_plots(df1, df2, filename):
    df1["unique_group"] = df1["QM"].astype(str) + "_" + df1["x1"].astype(str) + "_" + df1["x2"].astype(str)
    df2["unique_group"] = df2["QM"].astype(str) + "_" + df2["x1"].astype(str) + "_" + df2["x2"].astype(str)

    groups_df1 = df1.groupby("unique_group")
    groups_df2 = df2.groupby("unique_group")

    n_groups = groups_df1.ngroups
    ncols = 3
    nrows = (n_groups + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))
    axes = axes.flatten()

    for idx, (group_name, group_df1) in enumerate(groups_df1):
        qT1 = group_df1['qT'].values
        A1 = group_df1['A'].values
        A1_err = group_df1['dA'].values
        QM = group_df1['QM'].values[0]

        axes[idx].errorbar(qT1, A1, yerr=A1_err, fmt='o', color='blue', label='Experiment')


        group_df2 = groups_df2.get_group(group_name)
        qT2 = group_df2['qT'].values
        A2 = group_df2['A'].values
        A2_err = group_df2['dA'].values

        axes[idx].errorbar(qT2, A2, yerr=A2_err, fmt='s', color='red', label='Pseudo Data')

        axes[idx].set_title(f'$Q_M$ = {QM:.2f} GeV')
        axes[idx].set_xlabel('qT')
        axes[idx].set_ylabel('A')
        axes[idx].legend()
        axes[idx].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(pseudo_data_folder,str(filename) + ".pdf"))
    #plt.show()

#results_df.to_csv("pseudodata_E288_200.csv", index=False)
#print("Computed A values saved csv file")


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


gen_plots(E288_200,E288_200_pseudo,"E288_200")
gen_plots(E288_300,E228_300_pseudo,"E288_300")
gen_plots(E288_400,E228_400_pseudo,"E288_400")
gen_plots(E605,E605_pseudo,"E605")
gen_plots(E772,E772_pseudo,"E772")


######### B(QM) #################

data = E288_200_pseudo

# Generate QM Range for Comparison
QM_values = np.linspace(data['QM'].min(), data['QM'].max(), 200)
fDNNQ_values = fDNNQ(QM_values)

## Plot Analytical vs. Model Predictions
plt.figure(figsize=(10, 6))
plt.plot(QM_values, fDNNQ_values, label=r'Analytical $\mathcal{B}(Q_M)$', linestyle='--', color='blue')
plt.xlabel(r'$Q_M$', fontsize=14)
plt.ylabel(r'$f_{DNNQ}(Q_M)$', fontsize=14)
plt.title('Comparison of Analytical $\mathcal{B}(Q_M)$ and DNNQ Model', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig(os.path.join(pseudo_data_folder,"QM_comparison_plot.pdf"))
# plt.show()
