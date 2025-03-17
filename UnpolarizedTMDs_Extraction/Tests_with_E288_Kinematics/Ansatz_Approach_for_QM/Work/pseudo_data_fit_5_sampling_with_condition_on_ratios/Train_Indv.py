import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import lhapdf
from matplotlib.backends.backend_pdf import PdfPages

# Load PDF Set
NNPDF4_nlo = lhapdf.mkPDF('NNPDF40_nlo_as_01180')

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
E288_200 = pd.read_csv("gen_pseudo_data/pseudodata_E288_200.csv")
E288_300 = pd.read_csv("gen_pseudo_data/pseudodata_E288_300.csv")
E288_400 = pd.read_csv("gen_pseudo_data/pseudodata_E288_400.csv")
E605 = pd.read_csv("gen_pseudo_data/pseudodata_E605.csv")
E772 = pd.read_csv("gen_pseudo_data/pseudodata_E772.csv")

data = pd.concat([E288_200])


def fDNNQ(QM, b=0.5):
    return b * QM


# Compute S(qT) Contribution
mm = 0.5
def S(qT):
    return (8 * mm * mm + qT**4) / (32 * np.pi * mm) * tf.exp(-qT**2 / (2 * mm))

def QM_int(QM):
    return (-1) / (2 * QM**2)

def compute_QM_integrals(QM_array):
    QM_integrated = QM_int(QM_array + 0.5) - QM_int(QM_array - 0.5)
    return QM_integrated


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
        ReplicaA = np.random.normal(loc=tempA, scale=0.1*tempAerr)
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


    Sk_contribution = S(qT_values)
    pseudodata_df['SqT'] = Sk_contribution

    QM_integral = compute_QM_integrals(tempQM)
    pseudodata_df['QM_int'] = QM_integral
    B_QM = ReplicaA / (hc_factor * factor * PDFs * Sk_contribution * QM_integral)

    pseudodata_df['B_calc'] = B_QM
    pseudodata_df['B_true'] = tempBtrue

    pseudodata_df['B_ratio'] = B_QM/tempBtrue

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
    with PdfPages(f"{replica_data_folder}/Replica_{rep_num}_Comparison_B_Subplots.pdf") as pdf:
        fig, axes = plt.subplots(nrows=len(unique_QM) // 2 + len(unique_QM) % 2, ncols=2, figsize=(12, 6 * (len(unique_QM) // 2)))
        axes = axes.flatten()
        
        for i, QM_val in enumerate(unique_QM):
            mask = prepared_df['QM'] == QM_val
            axes[i].scatter(prepared_df['qT'][mask], prepared_df['B_calc'][mask], color='r', marker='x', label=f'QM = {QM_val:.2f} (Calc)')
            axes[i].scatter(prepared_df['qT'][mask], prepared_df['B_true'][mask], color='b', marker='x', label=f'QM = {QM_val:.2f} (True)')
            axes[i].set_xlabel(r'$q_T$')
            axes[i].set_ylabel(r'$B(Q_M)$')
            axes[i].set_ylim(0,14)
            axes[i].legend()
            axes[i].grid(True)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        print("Replica Subplots for B(QM) saved successfully in Subplots.pdf")


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
            axes[i].scatter(prepared_df['qT'][mask], prepared_df['B_ratio'][mask], color='b', marker='x', label=f'QM = {QM_val:.2f} (B ratio)')
            axes[i].set_xlabel(r'$q_T$')
            axes[i].set_ylabel(r'$B(Q_M)$')
            axes[i].set_ylim(0,6)
            axes[i].legend()
            axes[i].grid(True)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        print("Replica Subplots for B(QM) saved successfully in Subplots.pdf")


# Define the DNN Model
def DNNB():
    return tf.keras.Sequential([
        tf.keras.Input(shape=(1,)), 
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])


# Define Loss Function
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true))  # MSE loss

# Train the Model
def replica_model(i):
    # Generate replica data
    replica_data = GenerateReplicaData(data)
    replica_data.to_csv(f'replica_data_{i}.csv')
    # Let's apply the condition
    delta = 0.1
    # replica_data = replica_data[(replica_data['A_ratio'] <= 1 + delta) | (replica_data['A_ratio'] >= 1 - delta) | 
    #                         (replica_data['B_ratio'] <= 1 + delta) | (replica_data['B_ratio'] >= 1 - delta)]
    # replica_data = replica_data[(replica_data['A_ratio'] <= 1 + delta) & (replica_data['A_ratio'] >= 1 - delta) | 
    #                         (replica_data['B_ratio'] <= 1 + delta) & (replica_data['B_ratio'] >= 1 - delta)]
    QM_values = np.array(replica_data['QM'])
    B_QM = np.array(replica_data['B_calc'])
    # Generate comprison plots
    generate_replica_A_subplots(replica_data,i)
    generate_replica_B_subplots(replica_data,i)
    generate_ratio_subplots(replica_data,i)

    initial_lr = 0.01  
    epochs = 500  
    batch_size = 200

    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=initial_lr,
    #     decay_steps=50,
    #     decay_rate=0.96,
    #     staircase=True
    # )
    
    dnnB = DNNB()
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    dnnB.compile(optimizer=optimizer, loss=custom_loss)
    
    history = dnnB.fit(
        x=QM_values,  
        y=B_QM,  
        epochs=epochs,
        batch_size=batch_size,
        verbose=2
    )
    
    # Save Model
    model_path = os.path.join(models_folder, f'DNNB_model_{i}.h5')
    dnnB.save(model_path)
    print(f"Model {i} saved successfully at {model_path}!")
    
    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Loss', color='b')
    plt.title(f'Model {i} Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(loss_plot_folder, f'loss_plot_model_{i}.pdf')
    plt.savefig(loss_plot_path)
    print(f"Loss plot for Model {i} saved successfully at {loss_plot_path}!")

# Train multiple replicas
for i in range(3):
    replica_model(i)

