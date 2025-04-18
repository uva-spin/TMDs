import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import lhapdf
from functions_and_constants import *
from matplotlib.backends.backend_pdf import PdfPages
#from DNN_model import *
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)


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

models_folder = 'Models'
loss_plot_folder = 'Loss_Plots'
replica_data_folder = 'Replica_Data'
create_folders(models_folder)
create_folders(loss_plot_folder)
create_folders(replica_data_folder)


# Load Data
E288_200 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated/E288_200.csv")
E288_300 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated/E288_300.csv")
E288_400 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated/E288_400.csv")
# E605 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated/E605.csv")
# E772 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated/E772.csv")


data = pd.concat([E288_200,E288_300,E288_400], ignore_index=True)
# data = pd.concat([E772], ignore_index=True)
# data = pd.concat([E288_400,E772], ignore_index=True)


def QM_int(QM):
    return (-1) / (2 * QM**2)

def compute_QM_integrals(QM_array):
    QM_integrated = QM_int(QM_array + 0.5) - QM_int(QM_array - 0.5)
    return QM_integrated


def fDNNQ(QM):
    return 1 + 0*QM


def GenerateReplicaData(df):
    df=df[df['qT'] < 0.2 * df['QM']]
    df = df[(9.0 > df['QM']) | (df['QM'] > 11.0)]

    temp_df = {'x1': [],
            'x2': [],
            'qT': [],
            'QM': [],
            'eu2_fu_x1_fubar_x2': [],
            'eu2_fu_x2_fubar_x1': [],
            'ed2_fd_x1_fdbar_x2': [],
            'ed2_fd_x2_fdbar_x1': [],
            'es2_fs_x1_fsbar_x2': [],
            'es2_fs_x2_fsbar_x1':[],
            'A_true_err': [],
            'A_replica': [],
            'A_pred': []}
    pseudodata_df = {'x1': [],
                     'x2': [],
                     'qT': [],
                     'QM': [],
                     'eu2_fu_x1_fubar_x2': [],
                     'eu2_fu_x2_fubar_x1': [],
                     'ed2_fd_x1_fdbar_x2': [],
                     'ed2_fd_x2_fdbar_x1': [],
                     'es2_fs_x1_fsbar_x2': [],
                     'es2_fs_x2_fsbar_x1':[],
                     'A_true': [],
                     'A_true_err': [],
                     'A_replica': [],
                     'A_ratio':[],
                     'factor':[],
                     'PDFs':[],
                     'QM_int':[],
                     'SB_calc':[]}
    pseudodata_df['x1'] = df['x1']
    pseudodata_df['x2'] = df['x2']
    pseudodata_df['qT'] = df['qT']
    pseudodata_df['QM'] = df['QM']
    pseudodata_df['A_true'] = df['A']
    pseudodata_df['A_true_err'] = df['dA']
    tempQM = np.array(df['QM'])
    tempA = np.array(df['A'])
    tempAerr = np.abs(np.array(df['dA'])) 
    while True:
        #ReplicaA = np.random.uniform(low=tempA - 1.0*tempAerr, high=tempA + 1.0*tempAerr)
        ReplicaA = np.random.normal(loc=tempA, scale=1.0*tempAerr)
        print("Sampling...")
        if np.all(ReplicaA > 0):  # Correct way to check all elements
            break
    pseudodata_df['A_replica'] = ReplicaA

    pseudodata_df['A_ratio'] = ReplicaA/tempA


    # Extract Values
    x1_values = tf.constant(df['x1'].values, dtype=tf.float32)
    x2_values = tf.constant(df['x2'].values, dtype=tf.float32)
    qT_values = tf.constant(df['qT'].values, dtype=tf.float32)
    QM_values = tf.constant(df['QM'].values, dtype=tf.float32)

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

    eu2_fu_x1_fubar_x2 = eu2*np.array(f_u_x1) * np.array(f_ubar_x2)
    eu2_fu_x2_fubar_x1 = eu2*np.array(f_u_x2) * np.array(f_ubar_x1)
    ed2_fd_x1_fdbar_x2 = ed2*np.array(f_d_x1) * np.array(f_dbar_x2)
    ed2_fd_x2_fdbar_x1 = ed2*np.array(f_d_x2) * np.array(f_dbar_x1)
    es2_fs_x1_fsbar_x2 = es2*np.array(f_s_x1) * np.array(f_sbar_x2)
    es2_fs_x2_fsbar_x1 = es2*np.array(f_s_x2) * np.array(f_sbar_x1)


    PDFs = (eu2_fu_x1_fubar_x2 + 
            eu2_fu_x2_fubar_x1 + 
            ed2_fd_x1_fdbar_x2 + 
            ed2_fd_x2_fdbar_x1 + 
            es2_fs_x1_fsbar_x2 + 
            es2_fs_x2_fsbar_x1)
    
    pseudodata_df['eu2_fu_x1_fubar_x2'] = eu2_fu_x1_fubar_x2
    pseudodata_df['eu2_fu_x2_fubar_x1'] = eu2_fu_x2_fubar_x1
    pseudodata_df['ed2_fd_x1_fdbar_x2'] = ed2_fd_x1_fdbar_x2
    pseudodata_df['ed2_fd_x2_fdbar_x1'] = ed2_fd_x2_fdbar_x1
    pseudodata_df['es2_fs_x1_fsbar_x2'] = es2_fs_x1_fsbar_x2
    pseudodata_df['es2_fs_x2_fsbar_x1'] = es2_fs_x2_fsbar_x1


    pseudodata_df['PDFs'] = PDFs

    QM_integral = compute_QM_integrals(tempQM)
    pseudodata_df['QM_int'] = QM_integral
    # B_QM = ReplicaA / (hc_factor * factor * PDFs * Sk_contribution * QM_integral)

    SB = ReplicaA / (hc_factor * factor * PDFs * QM_integral)

    pseudodata_df['SB_calc'] = SB

    return pd.DataFrame(pseudodata_df)



def generate_replica_A_subplots(df,rep_num):
    prepared_df = df
    unique_QM = np.unique(prepared_df['QM'])
    
    # Create a PDF to store subplots
    with PdfPages(f"{replica_data_folder}/Replica_{rep_num}_Comparison_A_Subplots.pdf") as pdf:
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


def generate_replica_B_subplots(df,rep_num):
    prepared_df = df
    unique_QM = np.unique(prepared_df['QM'])
    
    # Create a PDF to store subplots
    with PdfPages(f"{replica_data_folder}/Replica_{rep_num}_Comparison_SB_Subplots.pdf") as pdf:
        fig, axes = plt.subplots(nrows=len(unique_QM) // 2 + len(unique_QM) % 2, ncols=2, figsize=(12, 6 * (len(unique_QM) // 2)))
        axes = axes.flatten()
        
        for i, QM_val in enumerate(unique_QM):
            mask = prepared_df['QM'] == QM_val
            axes[i].scatter(prepared_df['qT'][mask], prepared_df['SB_calc'][mask], color='r', marker='x', label=f'QM = {QM_val:.2f} (Calc)')
            # axes[i].scatter(prepared_df['qT'][mask], prepared_df['B_true'][mask], color='b', marker='x', label=f'QM = {QM_val:.2f} (True)')
            axes[i].set_xlabel(r'$q_T$')
            axes[i].set_ylabel(r'$B(Q_M)$')
            #axes[i].set_ylim(0,14)
            axes[i].legend()
            axes[i].grid(True)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        print("Replica Subplots for SB saved successfully in Subplots.pdf")


def generate_ratio_subplots(df,rep_num):
    prepared_df = df
    unique_QM = np.unique(prepared_df['QM'])
    
    # Create a PDF to store subplots
    with PdfPages(f"{replica_data_folder}/Replica_{rep_num}_Comparison_Ratio_Subplots.pdf") as pdf:
        fig, axes = plt.subplots(nrows=len(unique_QM) // 2 + len(unique_QM) % 2, ncols=2, figsize=(12, 6 * (len(unique_QM) // 2)))
        axes = axes.flatten()
        
        for i, QM_val in enumerate(unique_QM):
            mask = prepared_df['QM'] == QM_val
            axes[i].scatter(prepared_df['qT'][mask], prepared_df['A_ratio'][mask], color='r', marker='x', label=f'QM = {QM_val:.2f} (A ratio)')
            # axes[i].scatter(prepared_df['qT'][mask], prepared_df['B_ratio'][mask], color='b', marker='x', label=f'QM = {QM_val:.2f} (B ratio)')
            axes[i].set_xlabel(r'$q_T$')
            axes[i].set_ylabel(r'$B(Q_M)$')
            #axes[i].set_ylim(0,6)
            axes[i].legend()
            axes[i].grid(True)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        print("Replica Subplots for B(QM) saved successfully in Subplots.pdf")


initial_lr = 0.0005

epochs = 1000 # 1000 #500  

batch_size = 12


def DNNB(name):
    inp = tf.keras.Input(shape=(1,))
    initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=None)
    x1 = tf.keras.layers.Dense(250, activation='relu6', kernel_initializer = initializer)(inp)
    x2 = tf.keras.layers.Dense(100, activation='tanh', kernel_initializer = initializer)(x1)
    x3 = tf.keras.layers.Dense(100, activation='relu6', kernel_initializer = initializer)(x2)
    out = tf.keras.layers.Dense(1, activation='linear', kernel_initializer = initializer)(x3)
    return tf.keras.Model(inp, out, name=name)


def DNNS(name):
    inp = tf.keras.Input(shape=(1,))
    initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=None)
    x1 = tf.keras.layers.Dense(250, activation='relu6', kernel_initializer = initializer)(inp)
    x2 = tf.keras.layers.Dense(250, activation='tanh', kernel_initializer = initializer)(x1)
    x3 = tf.keras.layers.Dense(250, activation='relu6', kernel_initializer = initializer)(x2)
    out = tf.keras.layers.Dense(1, activation='softplus', kernel_initializer = initializer)(x3)
    return tf.keras.Model(inp, out, name=name)


def sigma_model():
    x1= tf.keras.Input(shape=(1,), name='x1')
    x2= tf.keras.Input(shape=(1,), name='x2')
    qT= tf.keras.Input(shape=(1,), name='qT')
    QM = tf.keras.Input(shape=(1,), name='QM')
    eu2_fu_x1_fubar_x2 = tf.keras.Input(shape=(1,), name='eu2_fu_x1_fubar_x2')
    eu2_fu_x2_fubar_x1 = tf.keras.Input(shape=(1,), name='eu2_fu_x2_fubar_x1')
    ed2_fd_x1_fdbar_x2 = tf.keras.Input(shape=(1,), name='ed2_fd_x1_fdbar_x2')
    ed2_fd_x2_fdbar_x1 = tf.keras.Input(shape=(1,), name='ed2_fd_x2_fdbar_x1')
    es2_fs_x1_fsbar_x2 = tf.keras.Input(shape=(1,), name='es2_fs_x1_fsbar_x2')
    es2_fs_x2_fsbar_x1 = tf.keras.Input(shape=(1,), name='es2_fs_x2_fsbar_x1')

    # These are tensor inputs, convert it to np

    SModel = DNNS('SqT')
    BModel = DNNB('BQM')

    Sq = SModel(qT)
    BQM = BModel(QM)

    SB = tf.keras.layers.Multiply(name='SB2')([Sq, BQM])

    factor_temp = factor

    PDFs = tf.keras.layers.Add()([eu2_fu_x1_fubar_x2,eu2_fu_x2_fubar_x1,ed2_fd_x1_fdbar_x2,ed2_fd_x2_fdbar_x1,es2_fs_x1_fsbar_x2,es2_fs_x2_fsbar_x1])

    QM_integral_temp = compute_QM_integrals(QM)
    A_pred = SB * factor_temp * PDFs * hc_factor * QM_integral_temp

    return tf.keras.Model([x1,x2,qT,QM,eu2_fu_x1_fubar_x2,eu2_fu_x2_fubar_x1,ed2_fd_x1_fdbar_x2,ed2_fd_x2_fdbar_x1,es2_fs_x1_fsbar_x2,es2_fs_x2_fsbar_x1],A_pred)



## This section is to generate sanity plots for each replica training

# Compute A Predictions
def compute_A(model, x1, x2, qT, QM,eu2_fu_x1_fubar_x2,eu2_fu_x2_fubar_x1,ed2_fd_x1_fdbar_x2,ed2_fd_x2_fdbar_x1,es2_fs_x1_fsbar_x2,es2_fs_x2_fsbar_x1):
    # Get Predictions from All Models
    A_pred = model.predict([x1,x2,qT,QM,eu2_fu_x1_fubar_x2,eu2_fu_x2_fubar_x1,ed2_fd_x1_fdbar_x2,ed2_fd_x2_fdbar_x1,es2_fs_x1_fsbar_x2,es2_fs_x2_fsbar_x1], verbose=0).flatten()
    return A_pred



def prep_data_for_plots(model,df):


    temp_df = {'x1': [],
            'x2': [],
            'qT': [],
            'QM': [],
            'eu2_fu_x1_fubar_x2': [],
            'eu2_fu_x2_fubar_x1': [],
            'ed2_fd_x1_fdbar_x2': [],
            'ed2_fd_x2_fdbar_x1': [],
            'es2_fs_x1_fsbar_x2': [],
            'es2_fs_x2_fsbar_x1':[],
            'A_true_err': [],
            'A_replica': [],
            'A_pred': []}
        
    eu2_fu_x1_fubar_x2 = df['eu2_fu_x1_fubar_x2'].values
    eu2_fu_x2_fubar_x1 = df['eu2_fu_x2_fubar_x1'].values
    ed2_fd_x1_fdbar_x2 = df['ed2_fd_x1_fdbar_x2'].values
    ed2_fd_x2_fdbar_x1 = df['ed2_fd_x2_fdbar_x1'].values
    es2_fs_x1_fsbar_x2 = df['es2_fs_x1_fsbar_x2'].values
    es2_fs_x2_fsbar_x1 = df['es2_fs_x2_fsbar_x1'].values


    qT = df['qT'].values
    QM = df['QM'].values
    x1 = df['x1'].values
    x2 = df['x2'].values
    A_replica = df['A_replica'].values
    A_true_err = df['A_true_err'].values
    A_pred = compute_A(model, x1, x2, qT, QM,eu2_fu_x1_fubar_x2,eu2_fu_x2_fubar_x1,ed2_fd_x1_fdbar_x2,ed2_fd_x2_fdbar_x1,es2_fs_x1_fsbar_x2,es2_fs_x2_fsbar_x1)

    temp_df['x1'] = x1
    temp_df['x2'] = x2
    temp_df['qT'] = qT
    temp_df['QM'] = QM
    temp_df['A_replica'] = A_replica
    temp_df['A_true_err'] = A_true_err
    temp_df['A_pred'] = A_pred

    temp_df['eu2_fu_x1_fubar_x2'] = eu2_fu_x1_fubar_x2
    temp_df['eu2_fu_x2_fubar_x1'] = eu2_fu_x2_fubar_x1
    temp_df['ed2_fd_x1_fdbar_x2'] = ed2_fd_x1_fdbar_x2
    temp_df['ed2_fd_x2_fdbar_x1'] = ed2_fd_x2_fdbar_x1
    temp_df['es2_fs_x1_fsbar_x2'] = es2_fs_x1_fsbar_x2
    temp_df['es2_fs_x2_fsbar_x1'] = es2_fs_x2_fsbar_x1

    return pd.DataFrame(temp_df)


def generate_subplots(model,df,rep_num):
    prepared_df = prep_data_for_plots(model,df)
    unique_QM = np.unique(prepared_df['QM'])
    
    # Create a PDF to store subplots
    with PdfPages(f"{replica_data_folder}/Replica_{rep_num}_Result.pdf") as pdf:
        fig, axes = plt.subplots(nrows=len(unique_QM) // 2 + len(unique_QM) % 2, ncols=2, figsize=(12, 6 * (len(unique_QM) // 2)))
        axes = axes.flatten()
        
        for i, QM_val in enumerate(unique_QM):
            mask = prepared_df['QM'] == QM_val
            axes[i].errorbar(prepared_df['qT'][mask], prepared_df['A_replica'][mask], yerr=prepared_df['A_true_err'][mask], fmt='bo', label=f'QM = {QM_val:.2f} (True)', capsize=3)
            axes[i].scatter(prepared_df['qT'][mask], prepared_df['A_pred'][mask], color='r', marker='x', label=f'QM = {QM_val:.2f} (Pred)')
            axes[i].set_xlabel(r'$q_T$')
            axes[i].set_ylabel(r'$A(q_T)$')
            axes[i].legend()
            axes[i].grid(True)
            
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        print("Subplots saved successfully in Subplots.pdf")


def gen_SB_plots(model, df,replica_id):
    # Generate QM Range for Comparison
    qT_values = np.linspace(df['QM'].min(), df['QM'].max(), 200)
    QM_values = np.linspace(df['qT'].min(), df['qT'].max(), 200)
    #fDNNQ_values = fDNNQ(QM_test)
    SqT_model = model.get_layer('SqT')
    BQM_model = model.get_layer('BQM')
    # SB2_model = model.get_layer('SB2')
    
    # Predict with each sub-model separately
    sq_vals = SqT_model.predict(qT_values, verbose=0).flatten()
    bqm_vals = BQM_model.predict(QM_values, verbose=0).flatten()

    dnnQ_contributions = np.array(sq_vals)*np.array(bqm_vals)
    
    # Plot Analytical vs. Model Predictions
    plt.figure(figsize=(10, 6))
    #plt.plot(QM_test, fDNNQ_values, label=r'Analytical $\mathcal{B}(Q_M)$', linestyle='--', color='blue')
    plt.plot(QM_values, dnnQ_contributions, label='DNNQ Model Mean', linestyle='-', color='red')
    plt.xlabel(r'$Q_M$', fontsize=14)
    plt.ylabel(r'$f_{DNNQ}(Q_M)$', fontsize=14)
    plt.title('Comparison of Analytical $\mathcal{B}(Q_M)$ and DNNQ Model', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig(f"{replica_data_folder}/QM_comparison_plot_{replica_id}.pdf")
    plt.close()


class DataDY(object):
    def __init__(self, pdfset='NNPDF40_nlo_as_01180'):
        self.pdfData = lhapdf.mkPDF(pdfset)

        self.eu = 2/3
        self.eubar = -2/3
        self.ed = -1/3
        self.edbar = 1/3
        self.es = -1/3
        self.esbar = 1/3


    def pdf(self, flavor, x, QQ):
        return np.array([self.pdfData.xfxQ2(flavor, ax, qq) for ax, qq in zip(x, QQ)])
    
    def makeData(self, df):

        data = {'x1': [],
             'x2': [],
             'qT': [],
             'QM': [],
             'eu2_fu_x1_fubar_x2': [],
             'eu2_fu_x2_fubar_x1': [],
             'ed2_fd_x1_fdbar_x2': [],
             'ed2_fd_x2_fdbar_x1': [],
             'es2_fs_x1_fsbar_x2': [],
             'es2_fs_x2_fsbar_x1':[]}
        

        y = []
        err = []

        y = np.array(df['A_replica'])
        err = np.array(df['A_true_err'])

        x1 = np.array(df['x1'])
        x2 = np.array(df['x2'])
        qT = np.array(df['qT'])
        QM = np.array(df['QM'])
        
        data['eu2_fu_x1_fubar_x2'] = np.array(self.eu**2 * self.pdf(2, x1, QM) * self.pdf(-2, x2, QM))
        data['eu2_fu_x2_fubar_x1'] = np.array(self.eu**2 * self.pdf(2, x2, QM) * self.pdf(-2, x1, QM))
        data['ed2_fd_x1_fdbar_x2'] = np.array(self.ed**2 * self.pdf(1, x1, QM) * self.pdf(-1, x2, QM))
        data['ed2_fd_x2_fdbar_x1'] = np.array(self.ed**2 * self.pdf(1, x2, QM) * self.pdf(-1, x1, QM))
        data['es2_fs_x1_fsbar_x2'] = np.array(self.es**2 * self.pdf(3, x1, QM) * self.pdf(-3, x2, QM))
        data['es2_fs_x2_fsbar_x1'] = np.array(self.es**2 * self.pdf(3, x2, QM) * self.pdf(-3, x1, QM))


        data['x1'] = df['x1']
        data['x2'] = df['x2']
        data['qT'] = df['qT']
        data['QM'] = df['QM']
        

        for key in data.keys():
            data[key] = np.array(data[key])

        return data, np.array(y), np.array(err)



# Define Loss Function
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true))  # MSE loss




def split_data(X, y, err, split=0.25):
    tstidxs = np.random.choice(list(range(len(y))), size=int(len(y)*split), replace=False)
    
    tst_X = {k: v[tstidxs] for k, v in X.items()}
    trn_X = {k: np.delete(v, tstidxs) for k, v in X.items()}
    
    tst_y = y[tstidxs]
    trn_y = np.delete(y, tstidxs)
    
    tst_err = err[tstidxs]
    trn_err = np.delete(err, tstidxs)
    
    return trn_X, tst_X, trn_y, tst_y, trn_err, tst_err


# Train the Model
def replica_model(i):

    sigma_dnn = sigma_model()
    data_dnn = DataDY()

    # Generate replica data

    E288_200_Replica = GenerateReplicaData(E288_200)
    E288_300_Replica = GenerateReplicaData(E288_300)
    E288_400_Replica = GenerateReplicaData(E288_400)
    # E605_Replica = GenerateReplicaData(E605)
    # E772_Replica = GenerateReplicaData(E772)


    # replica_data = GenerateReplicaData(data)
    replica_data = pd.concat([E288_200_Replica,E288_300_Replica,E288_400_Replica], ignore_index=True)
    replica_data.to_csv(f"{replica_data_folder}/replica_data_{i}.csv")

    generate_replica_A_subplots(replica_data,i)
    generate_replica_B_subplots(replica_data,i)
    generate_ratio_subplots(replica_data,i)

    T_Xplt, T_yplt, T_errplt = data_dnn.makeData(replica_data)

    trn_X, tst_X, trn_y, tst_y, trn_err, tst_err = split_data(T_Xplt, T_yplt, T_errplt)


    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    sigma_dnn.compile(optimizer=optimizer, loss=custom_loss)
 
    history = sigma_dnn.fit(trn_X, trn_y, validation_data=(tst_X, tst_y), epochs=epochs, batch_size=batch_size, verbose=2)
    # Save Model
    model_path = os.path.join(models_folder, f'DNNB_model_{i}.h5')
    sigma_dnn.save(model_path)
    print(f"Model {i} saved successfully at {model_path}!")
    ###########################  This section is to test for investigating outlier

    # optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    # dnnSB.compile(optimizer=optimizer, loss=custom_loss)
    
    # history = dnnSB.fit(
    #     x=[x1_train,x2_train,qT_train,QM_train],  
    #     y=SB_train,
    #     validation_data = ([x1_test,x2_test,qT_test,QM_test], SB_test),  
    #     epochs=epochs,
    #     batch_size=batch_size,
    #     verbose=2
    # )
    
    # # Save Model
    # model_path = os.path.join(models_folder, f'DNNB_model_{i}.h5')
    # dnnSB.save(model_path)
    # print(f"Model {i} saved successfully at {model_path}!")
    generate_subplots(sigma_dnn,replica_data,i)
    gen_SB_plots(sigma_dnn, replica_data, i)
    
    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Loss', color='b')
    plt.plot(history.history['val_loss'], label='Loss', color='r')
    plt.title(f'Model {i} Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(loss_plot_folder, f'loss_plot_model_{i}.pdf')
    plt.savefig(loss_plot_path)
    print(f"Loss plot for Model {i} saved successfully at {loss_plot_path}!")

# Train multiple replicas
for i in range(5):
    replica_model(i)
