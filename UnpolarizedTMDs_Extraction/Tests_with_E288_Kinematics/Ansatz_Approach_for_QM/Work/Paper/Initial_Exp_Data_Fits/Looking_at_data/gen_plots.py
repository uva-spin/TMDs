import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import lhapdf
# from function import fDNNQ
from matplotlib.backends.backend_pdf import PdfPages

# Load PDF Set
NNPDF4_nlo = lhapdf.mkPDF('NNPDF40_nlo_as_01180')
alpha = 1/137
hc_factor = 3.89 * 10**8

def pdf(pdfset, flavor, x, QQ):
    return pdfset.xfxQ(flavor, x, QQ)

# Create necessary folders
def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")


plots_folder = 'Plots'
csvs_folder = 'csvs'
create_folders(plots_folder)
create_folders(csvs_folder)


# Load Data
E288_200 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated/E288_200.csv")
E288_300 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated/E288_300.csv")
E288_400 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated/E288_400.csv")
E605 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated/E605.csv")
E772 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated/E772.csv")


# data = pd.concat([E288_200,E288_300,E288_400,E605,E772], ignore_index=True)
data = pd.concat([E288_300], ignore_index=True)


# Compute S(qT) Contribution
mm = 0.5
def S(qT):
    return (8 * mm * mm + qT**4) / (32 * np.pi * mm) * tf.exp(-qT**2 / (2 * mm))

def QM_int(QM):
    return (-1) / (2 * QM**2)

def compute_QM_integrals(QM_array):
    QM_integrated = QM_int(QM_array + 0.5) - QM_int(QM_array - 0.5)
    return QM_integrated


## This is a dummy function
def fDNNQ(QM):
    return 1 + 0*QM


def GenerateReplicaData(df):
    pseudodata_df = {'x1': [],
                     'x2': [],
                     'qT': [],
                     'QM': [],
                     'A_true': [],
                     'A_true_err': [],
                     'A_replica': [],
                     'A_ratio':[],
                     'factor':[],
                     'PDFs':[],
                     'QM_int':[],
                     'SqT':[],
                     'B_true':[],
                     'B_calc':[],
                     'B_ratio':[]}
    pseudodata_df['x1'] = df['x1']
    pseudodata_df['x2'] = df['x2']
    pseudodata_df['qT'] = df['qT']
    pseudodata_df['QM'] = df['QM']
    pseudodata_df['A_true'] = df['A']
    pseudodata_df['A_true_err'] = df['dA']
    tempQM = np.array(df['QM'])
    tempA = np.array(df['A'])
    tempAerr = np.abs(np.array(df['dA'])) 
    tempBtrue = np.array(fDNNQ(tempQM))
    while True:
        #ReplicaA = np.random.uniform(low=tempA - 1.0*tempAerr, high=tempA + 1.0*tempAerr)
        ReplicaA = np.random.normal(loc=tempA, scale=0.0*tempAerr)
        if np.all(ReplicaA > 0):  # Correct way to check all elements
            break
    pseudodata_df['A_replica'] = ReplicaA

    pseudodata_df['A_ratio'] = ReplicaA/tempA


    # Extract Values
    x1_values = tf.constant(df['x1'].values, dtype=tf.float32)
    x2_values = tf.constant(df['x2'].values, dtype=tf.float32)
    qT_values = tf.constant(df['qT'].values, dtype=tf.float32)
    QM_values = tf.constant(df['QM'].values, dtype=tf.float32)

    # Constants
    alpha = tf.constant(1/137, dtype=tf.float32)  # Fine structure constant
    hc_factor = 3.89 * 10**8
    factor = ((4 * np.pi * alpha) ** 2) / (9 * 2 * np.pi)

    pseudodata_df['factor'] = factor

    # Compute PDFs
    f_u_x1 = pdf(NNPDF4_nlo, 2, x1_values, QM_values)
    f_ubar_x2 = pdf(NNPDF4_nlo, -2, x2_values, QM_values)
    f_u_x2 = pdf(NNPDF4_nlo, 2, x2_values, QM_values)
    f_ubar_x1 = pdf(NNPDF4_nlo, -2, x1_values, QM_values)
    f_d_x1 = pdf(NNPDF4_nlo, 1, x1_values, QM_values)
    f_dbar_x2 = pdf(NNPDF4_nlo, -1, x2_values, QM_values)
    f_d_x2 = pdf(NNPDF4_nlo, 1, x2_values, QM_values)
    f_dbar_x1 = pdf(NNPDF4_nlo, -1, x1_values, QM_values)
    f_s_x1 = pdf(NNPDF4_nlo, 3, x1_values, QM_values)
    f_sbar_x2 = pdf(NNPDF4_nlo, -3, x2_values, QM_values)
    f_s_x2 = pdf(NNPDF4_nlo, 3, x2_values, QM_values)
    f_sbar_x1 = pdf(NNPDF4_nlo, -3, x1_values, QM_values)

    PDFs = (np.array(f_u_x1) * np.array(f_ubar_x2) + 
            np.array(f_u_x2) * np.array(f_ubar_x1) + 
            np.array(f_d_x1) * np.array(f_dbar_x2) + 
            np.array(f_d_x2) * np.array(f_dbar_x1) + 
            np.array(f_s_x1) * np.array(f_sbar_x2) + 
            np.array(f_s_x2) * np.array(f_sbar_x1))

    pseudodata_df['PDFs'] = PDFs

    Sk_contribution = S(qT_values)
    pseudodata_df['SqT'] = Sk_contribution

    QM_integral = compute_QM_integrals(tempQM)
    pseudodata_df['QM_int'] = QM_integral
    B_QM = ReplicaA / (hc_factor * factor * PDFs * Sk_contribution * QM_integral)

    pseudodata_df['B_calc'] = B_QM
    pseudodata_df['B_true'] = tempBtrue

    pseudodata_df['B_ratio'] = B_QM/tempBtrue

    return pd.DataFrame(pseudodata_df)



def generate_replica_A_subplots(df,folder,filename):
    prepared_df = df
    unique_QM = np.unique(prepared_df['QM'])
    
    # Create a PDF to store subplots
    with PdfPages(f"{folder}/{filename}_Comparison_A_Subplots.pdf") as pdf:
        fig, axes = plt.subplots(nrows=len(unique_QM) // 2 + len(unique_QM) % 2, ncols=2, figsize=(12, 6 * (len(unique_QM) // 2)))
        axes = axes.flatten()
        
        for i, QM_val in enumerate(unique_QM):
            mask = prepared_df['QM'] == QM_val
            # axes[i].scatter(prepared_df['qT'][mask], prepared_df['A_true'][mask], color='b', label=f'QM = {QM_val:.2f} (True)')
            axes[i].scatter(prepared_df['qT'][mask], prepared_df['A_replica'][mask], color='r', marker='x', label=f'QM = {QM_val:.2f} (replica)')
            axes[i].errorbar(prepared_df['qT'][mask], prepared_df['A_true'][mask], yerr=prepared_df['A_true_err'][mask], fmt='bo', label=f'QM = {QM_val:.2f} (True)', capsize=3)
            #axes[i].errorbar(prepared_df['qT'][mask], prepared_df['A_replica'][mask], yerr=prepared_df['A_pred_err'][mask], fmt='rx', label=f'QM = {QM_val:.2f} (Pred)', capsize=3)
            axes[i].set_xlabel(r'$q_T$')
            axes[i].set_ylabel(r'$A(q_T)$')
            axes[i].legend()
            axes[i].grid(True)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        print("Replica Subplots for cross-section saved successfully in Subplots.pdf")


def generate_replica_B_subplots(df,folder,filename):
    prepared_df = df
    unique_QM = np.unique(prepared_df['QM'])
    
    # Create a PDF to store subplots
    with PdfPages(f"{folder}/{filename}_Comparison_B_Subplots.pdf") as pdf:
        fig, axes = plt.subplots(nrows=len(unique_QM) // 2 + len(unique_QM) % 2, ncols=2, figsize=(12, 6 * (len(unique_QM) // 2)))
        axes = axes.flatten()
        
        for i, QM_val in enumerate(unique_QM):
            mask = prepared_df['QM'] == QM_val
            axes[i].scatter(prepared_df['qT'][mask], prepared_df['B_calc'][mask], color='r', marker='x', label=f'QM = {QM_val:.2f} (Calc)')
            axes[i].scatter(prepared_df['qT'][mask], prepared_df['B_true'][mask], color='b', marker='x', label=f'QM = {QM_val:.2f} (True)')
            axes[i].set_xlabel(r'$q_T$')
            axes[i].set_ylabel(r'$B(Q_M)$')
            #axes[i].set_ylim(0,14)
            axes[i].legend()
            axes[i].grid(True)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        print("Replica Subplots for B(QM) saved successfully in Subplots.pdf")


def generate_ratio_subplots(df,folder,filename):
    prepared_df = df
    unique_QM = np.unique(prepared_df['QM'])
    
    # Create a PDF to store subplots
    with PdfPages(f"{folder}/{filename}_Comparison_Ratio_Subplots.pdf") as pdf:
        fig, axes = plt.subplots(nrows=len(unique_QM) // 2 + len(unique_QM) % 2, ncols=2, figsize=(12, 6 * (len(unique_QM) // 2)))
        axes = axes.flatten()
        
        for i, QM_val in enumerate(unique_QM):
            mask = prepared_df['QM'] == QM_val
            axes[i].scatter(prepared_df['qT'][mask], prepared_df['A_ratio'][mask], color='r', marker='x', label=f'QM = {QM_val:.2f} (A ratio)')
            axes[i].scatter(prepared_df['qT'][mask], prepared_df['B_ratio'][mask], color='b', marker='x', label=f'QM = {QM_val:.2f} (B ratio)')
            axes[i].set_xlabel(r'$q_T$')
            axes[i].set_ylabel(r'$B(Q_M)$')
            #axes[i].set_ylim(0,6)
            axes[i].legend()
            axes[i].grid(True)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        print("Replica Subplots for B(QM) saved successfully in Subplots.pdf")



## This section is to generate sanity plots for each replica training

# Compute A Predictions
def compute_A(model, x1, x2, qT, QM):
    # Get Predictions from All Models
    fDNN_mean = model.predict(QM, verbose=0).flatten()

    factor_temp = ((4*np.pi*alpha)**2)/(9*2*np.pi)


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

    ux1ubarx2_temp = np.array(f_u_x1)*np.array(f_ubar_x2)
    ubarx1ux2_temp = np.array(f_u_x2)*np.array(f_ubar_x1)
    dx1dbarx2_temp = np.array(f_d_x1)*np.array(f_dbar_x2)
    dbarx1dx2_temp = np.array(f_d_x2)*np.array(f_dbar_x1)
    sx1sbarx2_temp = np.array(f_s_x1)*np.array(f_sbar_x2)
    sbarx1sx2_temp = np.array(f_s_x2)*np.array(f_sbar_x1)
    PDFs_temp = ux1ubarx2_temp + ubarx1ux2_temp + dx1dbarx2_temp + dbarx1dx2_temp + sx1sbarx2_temp + sbarx1sx2_temp

    Sk_temp = S(qT)

    QM_integral_temp = compute_QM_integrals(QM)
    A_pred = fDNN_mean * factor_temp * PDFs_temp * Sk_temp * hc_factor * QM_integral_temp
    return A_pred


def prep_data_for_plots(model,df):
    temp_df = {'x1': [],
        'x2': [],
        'qT': [],
        'QM': [],
        'A_true_err': [],
        'A_replica': [],
        'A_pred': []}

    qT = df['qT'].values
    QM = df['QM'].values
    x1 = df['x1'].values
    x2 = df['x2'].values
    A_replica = df['A_replica'].values
    A_true_err = df['A_true_err'].values
    A_pred = compute_A(model, x1, x2, qT, QM)

    temp_df['x1'] = x1
    temp_df['x2'] = x2
    temp_df['qT'] = qT
    temp_df['QM'] = QM
    temp_df['A_replica'] = A_replica
    temp_df['A_true_err'] = A_true_err
    temp_df['A_pred'] = A_pred

    return pd.DataFrame(temp_df)




####################################


# Generate replica data
E288_200_Sample = GenerateReplicaData(E288_200)
E288_300_Sample = GenerateReplicaData(E288_300)
E288_400_Sample = GenerateReplicaData(E288_400)
E605_Sample = GenerateReplicaData(E605)
E772_Sample = GenerateReplicaData(E772)

E288_200_Sample.to_csv(f"{csvs_folder}/E288_200_Sample.csv")
E288_300_Sample.to_csv(f"{csvs_folder}/E288_300_Sample.csv")
E288_400_Sample.to_csv(f"{csvs_folder}/E288_400_Sample.csv")
E605_Sample.to_csv(f"{csvs_folder}/E605_Sample.csv")
E772_Sample.to_csv(f"{csvs_folder}/E772_Sample.csv")

# E288_200 comprison plots
generate_replica_A_subplots(E288_200_Sample,plots_folder,'E288_200_Sample')
generate_replica_B_subplots(E288_200_Sample,plots_folder,'E288_200_Sample')
generate_ratio_subplots(E288_200_Sample,plots_folder,'E288_200_Sample')

# E288_300 comprison plots
generate_replica_A_subplots(E288_300_Sample,plots_folder,'E288_300_Sample')
generate_replica_B_subplots(E288_300_Sample,plots_folder,'E288_300_Sample')
generate_ratio_subplots(E288_300_Sample,plots_folder,'E288_300_Sample')

# E288_400 comprison plots
generate_replica_A_subplots(E288_400_Sample,plots_folder,'E288_400_Sample')
generate_replica_B_subplots(E288_400_Sample,plots_folder,'E288_400_Sample')
generate_ratio_subplots(E288_400_Sample,plots_folder,'E288_400_Sample')

# E605 comprison plots
generate_replica_A_subplots(E605_Sample,plots_folder,'E605_Sample')
generate_replica_B_subplots(E605_Sample,plots_folder,'E605_Sample')
generate_ratio_subplots(E605_Sample,plots_folder,'E605_Sample')

# E772 comprison plots
generate_replica_A_subplots(E772_Sample,plots_folder,'E772_Sample')
generate_replica_B_subplots(E772_Sample,plots_folder,'E772_Sample')
generate_ratio_subplots(E772_Sample,plots_folder,'E772_Sample')
