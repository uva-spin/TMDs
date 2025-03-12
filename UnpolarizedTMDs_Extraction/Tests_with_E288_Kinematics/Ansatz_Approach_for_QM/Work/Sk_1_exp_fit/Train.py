import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import lhapdf
import sys

# Load PDF Set
NNPDF4_nlo = lhapdf.mkPDF('NNPDF40_nlo_as_01180')

def pdf(pdfset, flavor, x, QQ):
    return pdfset.xfxQ(flavor, x, QQ)

# Create necessary folders
def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")

models_folder = '/scratch/cee9hc/Unpolarized_TMDs/with_E288_E605_E702/Sk_1_exp/Models'
loss_plot_folder = '/scratch/cee9hc/Unpolarized_TMDs/with_E288_E605_E702/Sk_1_exp/Loss_Plots'
create_folders(models_folder)
create_folders(loss_plot_folder)

# Load and Preprocess Data
E288 = pd.read_csv("../Data/E288.csv")
E605 = pd.read_csv("../Data/E605.csv")
E772 = pd.read_csv("../Data/E772.csv")
data = pd.concat([E288,E605,E772])

x1_values = tf.constant(data['x1'].values, dtype=tf.float32)
x2_values = tf.constant(data['x2'].values, dtype=tf.float32)
qT_values = tf.constant(data['qT'].values, dtype=tf.float32)
QM_values = tf.constant(data['QM'].values, dtype=tf.float32)
A_true_values = tf.constant(data['A'].values, dtype=tf.float32)


alpha = tf.constant(1/137, dtype=tf.float32)  # Fine structure constant


#hc_factor = 3.89*10**8 * 1000000
hc_factor = 3.89*10**8 * 10000

A_true_values = A_true_values / hc_factor

factor = ((4*np.pi*alpha)**2)/(9*QM_values**3)/(2*np.pi)
A_true_values = A_true_values / factor


f_u_x1 = pdf(NNPDF4_nlo, 2,  x1_values, QM_values) 
f_ubar_x2 = pdf(NNPDF4_nlo, -2, x2_values, QM_values)
f_u_x2 = pdf(NNPDF4_nlo, 2, x2_values, QM_values)
f_ubar_x1 = pdf(NNPDF4_nlo, -2,  x1_values, QM_values)
f_d_x1 = pdf(NNPDF4_nlo, 1,  x1_values, QM_values) 
f_dbar_x2 = pdf(NNPDF4_nlo, -1, x2_values, QM_values)
f_d_x2 = pdf(NNPDF4_nlo, 1, x2_values, QM_values)
f_dbar_x1 = pdf(NNPDF4_nlo, -1,  x1_values, QM_values)
f_s_x1 = pdf(NNPDF4_nlo, 3,  x1_values, QM_values) 
f_sbar_x2 = pdf(NNPDF4_nlo, -3, x2_values, QM_values)
f_s_x2 = pdf(NNPDF4_nlo, 3, x2_values, QM_values)
f_sbar_x1 = pdf(NNPDF4_nlo, -3, x1_values, QM_values)


ux1ubarx2_term = np.array(f_u_x1)*np.array(f_ubar_x2)
ubarx1ux2_term = np.array(f_u_x2)*np.array(f_ubar_x1)
dx1dbarx2_term = np.array(f_d_x1)*np.array(f_dbar_x2)
dbarx1dx2_term = np.array(f_d_x2)*np.array(f_dbar_x1)
sx1sbarx2_term = np.array(f_s_x1)*np.array(f_sbar_x2)
sbarx1sx2_term = np.array(f_s_x2)*np.array(f_sbar_x1)
PDFs = (ux1ubarx2_term + ubarx1ux2_term + dx1dbarx2_term + dbarx1dx2_term + sx1sbarx2_term + sbarx1sx2_term)


A_true_values = A_true_values / PDFs


mm = 0.5

def S(k):
    return ((k**2)/(mm*np.pi))*np.exp(-(k**2)/mm)
    
def Sk_cont(qT):
    return (8*mm*mm + qT*qT*qT*qT)/(32*np.pi*mm)*(np.exp(-(qT*qT)/(2*mm)))
    
Sk_contribution = Sk_cont(qT_values)


A_true_values = A_true_values / Sk_contribution


# Normalize Inputs (QM) and Outputs (A_true)
qT_mean, qT_std = tf.reduce_mean(qT_values), tf.math.reduce_std(qT_values)
qT_values = (qT_values - qT_mean) / qT_std  # Standardization

# Normalize Inputs (QM) and Outputs (A_true)
QM_mean, QM_std = tf.reduce_mean(QM_values), tf.math.reduce_std(QM_values)
QM_values = (QM_values - QM_mean) / QM_std  # Standardization

A_mean, A_std = tf.reduce_mean(A_true_values), tf.math.reduce_std(A_true_values)
A_true_values = (A_true_values - A_mean) / A_std  # Standardization

# Define the Model
def DNNQ():
    return tf.keras.Sequential([
        tf.keras.Input(shape=(1,)), 
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')  # Output remains unbounded
    ])




# Define Custom Loss Function
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true))  # MSE loss

# Train the Model
def train_replica():
    i = sys.argv[1]
    initial_lr = 0.05  # Lower initial learning rate
    epochs = 500  # Increased training duration
    batch_size = 200

    # Learning Rate Scheduler (Exponential Decay)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=50,
        decay_rate=0.96,
        staircase=True
    )
    
    dnnQ = DNNQ()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    dnnQ.compile(optimizer=optimizer, loss=custom_loss)
    
    history = dnnQ.fit(
        x=QM_values,  # Scaled QM as input
        y=A_true_values,  # Scaled A_true as target
        epochs=epochs,
        batch_size=batch_size,
        verbose=2
    )
    
    # Save Model
    model_path = os.path.join(models_folder, f'DNNQ_model_{i}.h5')
    dnnQ.save(model_path)
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

train_replica()