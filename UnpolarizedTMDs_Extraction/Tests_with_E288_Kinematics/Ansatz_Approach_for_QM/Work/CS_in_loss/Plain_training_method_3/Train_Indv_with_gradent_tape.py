import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import lhapdf
from matplotlib.backends.backend_pdf import PdfPages
from functools import partial

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

alpha = 1/137
hc_factor = 3.89 * 10**8

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
                     'PDFs':[],
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
        ReplicaA = np.random.normal(loc=tempA, scale=1.0*tempAerr)
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

def generate_replica_A_subplots(df,rep_num):
    prepared_df = df
    unique_QM = np.unique(prepared_df['QM'])
    
    # Create a PDF to store subplots
    with PdfPages(f"{replica_data_folder}/Replica_{rep_num}_Comparison_A_Subplots.pdf") as pdf:
        fig, axes = plt.subplots(nrows=len(unique_QM) // 2 + len(unique_QM) % 2, ncols=2, figsize=(12, 6 * (len(unique_QM) // 2)))
        axes = axes.flatten()
        
        for i, QM_val in enumerate(unique_QM):
            mask = prepared_df['QM'] == QM_val
            axes[i].scatter(prepared_df['qT'][mask], prepared_df['A_replica'][mask], color='r', marker='x', label=f'QM = {QM_val:.2f} (replica)')
            axes[i].errorbar(prepared_df['qT'][mask], prepared_df['A_true'][mask], yerr=prepared_df['A_true_err'][mask], fmt='bo', label=f'QM = {QM_val:.2f} (True)', capsize=3)
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

def Sk_cont(qT):
    return (8*mm*mm + qT*qT*qT*qT)/(32*np.pi*mm)*(np.exp(-(qT*qT)/(2*mm)))

def QM_int(QM):
    return (-1)/(2*QM**2)

def compute_QM_integrals(QM_array):
    QM_array = np.atleast_1d(QM_array) 
    QM_integrated = QM_int(QM_array + 0.5) - QM_int(QM_array - 0.5)
    return QM_integrated[0] if QM_integrated.size == 1 else QM_integrated

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

    Sk_temp = Sk_cont(qT)

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

# Define the DNN Model
def DNNB():
    return tf.keras.Sequential([
        tf.keras.Input(shape=(1,)), 
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

# Define Loss Function
def custom_loss(y_true, y_pred, PDFs_batch, SqT_integral_batch, QM_integral_batch):
    alpha = 1/137
    hc_factor = 3.89 * 10**8
    factor = ((4 * np.pi * alpha) ** 2) / (9 * 2 * np.pi)
    # Compute predicted cross-section
    cross_section = y_pred * hc_factor * factor * PDFs_batch * SqT_integral_batch * QM_integral_batch
    # Compute Mean Squared Error (MSE) loss
    return tf.reduce_mean(tf.square(cross_section - y_true))

def replica_model(i):
    # Generate replica data
    replica_data = GenerateReplicaData(data)
    replica_data.to_csv(f"{replica_data_folder}/replica_data_{i}.csv")
    delta = 0.1
    # replica_data = replica_data[(replica_data['A_ratio'] <= 1 + delta) & (replica_data['A_ratio'] >= 1 - delta) | 
    #                         (replica_data['B_ratio'] <= 1 + delta) & (replica_data['B_ratio'] >= 1 - delta)]
    
    # Let's apply the condition
    QM_values = np.array(replica_data['QM'])
    generate_replica_A_subplots(replica_data, i)
    generate_replica_B_subplots(replica_data, i)
    generate_ratio_subplots(replica_data, i)

    #initial_lr = 0.01 
    initial_lr = 0.00001  
    epochs = 10000  
    batch_size = 200

    # Extract Values
    PDFs = tf.convert_to_tensor(replica_data['PDFs'].values, dtype=tf.float32)
    SqT_integral = tf.convert_to_tensor(replica_data['SqT'].values, dtype=tf.float32)
    QM_integral = tf.convert_to_tensor(replica_data['QM_int'].values, dtype=tf.float32)
    A_replica = tf.convert_to_tensor(replica_data['A_replica'].values, dtype=tf.float32)
    
    # Create DNN Model
    dnnB = DNNB()
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)

    # Create arrays to track loss history
    loss_history = []

    @tf.function
    def train_step(x_batch, y_batch, PDFs_batch, SqT_batch, QM_int_batch):
        with tf.GradientTape() as tape:
            y_pred = dnnB(x_batch, training=True)
            loss = custom_loss(y_batch, y_pred, PDFs_batch, SqT_batch, QM_int_batch)
        gradients = tape.gradient(loss, dnnB.trainable_variables)
        optimizer.apply_gradients(zip(gradients, dnnB.trainable_variables))
        return loss

    # Training loop
    for epoch in range(epochs):
        epoch_losses = []
        indices = np.random.permutation(len(QM_values))
        for j in range(0, len(QM_values), batch_size):
            batch_indices = indices[j:min(j + batch_size, len(QM_values))]
            x_batch = tf.gather(QM_values, batch_indices)
            y_batch = tf.gather(A_replica, batch_indices)
            PDFs_batch = tf.gather(PDFs, batch_indices)
            SqT_batch = tf.gather(SqT_integral, batch_indices)
            QM_int_batch = tf.gather(QM_integral, batch_indices)
            
            loss = train_step(x_batch, y_batch, PDFs_batch, SqT_batch, QM_int_batch)
            epoch_losses.append(float(loss))
        
        # Calculate average loss for this epoch
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        loss_history.append(avg_loss)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss}")

    # Create a history-like object for compatibility with plotting code
    class History:
        def __init__(self, loss):
            self.history = {'loss': loss}

    history = History(loss_history)

    # Save the model
    model_path = os.path.join(models_folder, f'DNNB_model_{i}.h5')
    dnnB.save(model_path)
    print(f"Model {i} saved successfully at {model_path}!")
    generate_subplots(dnnB, replica_data, i)

    # Plot Loss
    plt.figure(1, figsize=(10, 6))
    plt.plot(history.history['loss'], label='Loss', color='b')
    plt.title(f'Model {i} Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(loss_plot_folder, f'loss_plot_model_{i}.pdf')
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Loss plot for Model {i} saved successfully at {loss_plot_path}!")

# Train multiple replicas
for i in range(3):
    replica_model(i)