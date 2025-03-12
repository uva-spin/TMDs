import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import lhapdf

def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")

# Create Results Folder
results_folder = 'Results_from_the_Model'
create_folders(results_folder)

# Load LHAPDF Set
NNPDF4_nlo = lhapdf.mkPDF('NNPDF40_nlo_as_01180')

def pdf(pdfset, flavor, x, QQ):
    return pdfset.xfxQ(flavor, x, QQ)

# Load and Preprocess Data
E288 = pd.read_csv("../Data/E288.csv")
E605 = pd.read_csv("../Data/E605.csv")
E772 = pd.read_csv("../Data/E772.csv")
data = pd.concat([E288,E605,E772])

models_folder = 'Models'


x1_values = tf.constant(data['x1'].values, dtype=tf.float32)
x2_values = tf.constant(data['x2'].values, dtype=tf.float32)
qT_values = tf.constant(data['qT'].values, dtype=tf.float32)
QM_values = tf.constant(data['QM'].values, dtype=tf.float32)
A_true_values = tf.constant(data['A'].values, dtype=tf.float32)




alpha = tf.constant(1/137, dtype=tf.float32)  # Fine structure constant

# hc_factor = 3.89*10**8 * 1000000
hc_factor = 3.89*10**8 * 1
# A_true_values = A_true_values / hc_factor

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


# Define Custom Loss Function
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true))  # MSE loss

# Load All Trained Models
model_files = [f for f in os.listdir(models_folder) if f.endswith('.h5')]
models_list = [tf.keras.models.load_model(os.path.join(models_folder, f), custom_objects={'custom_loss': custom_loss}) for f in model_files]


print(f"Loaded {len(models_list)} models from '{models_folder}'.")

# Load Data
pseudo_df = data

# Compute Mean and Std for Scaling (Must Match Training)
QM_mean, QM_std = np.mean(pseudo_df["QM"]), np.std(pseudo_df["QM"])
A_mean, A_std = np.mean(A_true_values), np.std(A_true_values)


# PDF Function
def pdf(pdfset, flavor, x, QQ):
    return pdfset.xfxQ(flavor, x, QQ)

### Cross-Section #####

# Compute A Predictions
def compute_A(x1, x2, qT, QM):
    QM_scaled = (QM - QM_mean) / QM_std  # Normalize QM for Model Input
    QM_input = QM_scaled.reshape(-1, 1)

    # Get Predictions from All Models
    fDNN_contributions = np.array([model.predict(QM_input, verbose=0).flatten() for model in models_list])
    fDNN_mean = np.mean(fDNN_contributions, axis=0)
    fDNN_std = np.std(fDNN_contributions, axis=0)


    A_pred = (fDNN_mean * np.mean(A_std)) + np.mean(A_mean)  # Ensures A_std is scalar
    A_std_restored = fDNN_std * np.mean(A_std)

    factor_temp = ((4*np.pi*alpha)**2)/(9*QM**3)/(2*np.pi)


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


    #Sk_temp = (8*mm*mm + qT**4)/(32*np.pi*mm)*(np.exp(-(qT**2)/(2*mm)))
    Sk_temp = Sk_cont(qT)


    A_pred = A_pred * factor_temp * PDFs_temp * Sk_temp * hc_factor
    A_std_restored = A_std_restored * factor_temp * PDFs_temp * Sk_temp * hc_factor

    return A_pred, A_std_restored

# Group Data by Unique Combinations of QM, x1, x2
pseudo_df["unique_group"] = (
    pseudo_df["QM"].astype(str) + "_" + pseudo_df["x1"].astype(str) + "_" + pseudo_df["x2"].astype(str)
)
groups = pseudo_df.groupby("unique_group")
n_groups = groups.ngroups
ncols = 3
nrows = (n_groups + ncols - 1) // ncols

# Create Subplots for Each Group
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))
axes = axes.flatten()

for idx, (group_name, group_df) in enumerate(groups):
    qT = group_df['qT'].values
    QM = group_df['QM'].values
    x1 = group_df['x1'].values
    x2 = group_df['x2'].values
    A_true = group_df['A'].values

    #A_true = A_true / hc_factor

    A_pred_temp, A_std_temp = compute_A(x1, x2, qT, QM)
    
    #axes[idx].errorbar(qT, A_pred_temp, yerr=A_std_temp, fmt='o', color='red', alpha=0.5, label='A_pred')
    axes[idx].plot(qT, A_true, '*', color='blue', label='A_true')
    axes[idx].set_title(f'A vs qT for $Q_M$ = {QM[0]:.2f} GeV')
    axes[idx].set_xlabel('qT')
    axes[idx].set_ylabel('A')
    axes[idx].legend()
    axes[idx].grid(True)

plt.tight_layout()
plt.savefig(f"{results_folder}/QM_subplots_with_predictions.pdf")
#plt.show()


######### B(QM) #################

# Analytical Function for fDNNQ
def fDNNQ(QM, b=0.5):
    return np.exp(-b * QM)

# Generate QM Range for Comparison
QM_values = np.linspace(pseudo_df['QM'].min(), pseudo_df['QM'].max(), 200)
QM_scaled = (QM_values - QM_mean) / QM_std  # Normalize QM for Model

QM_tensor = QM_scaled.reshape(-1, 1)

# Get Model Predictions
dnnQ_contributions = np.array([model.predict(QM_tensor, verbose=0).flatten() for model in models_list])
dnnQ_mean = np.mean(dnnQ_contributions, axis=0)
dnnQ_std = np.std(dnnQ_contributions, axis=0)

# Restore Original Scale
dnnQ_mean = (dnnQ_mean * A_std) + A_mean
dnnQ_std = dnnQ_std * A_std



fDNNQ_values = fDNNQ(QM_values)

# Plot Analytical vs. Model Predictions
plt.figure(figsize=(10, 6))
plt.plot(QM_values, fDNNQ_values, label=r'Analytical $\mathcal{B}(Q_M)$', linestyle='--', color='blue')
plt.plot(QM_values, dnnQ_mean, label='DNNQ Model Mean', linestyle='-', color='red')
plt.fill_between(QM_values, dnnQ_mean - dnnQ_std, dnnQ_mean + dnnQ_std, color='red', alpha=0.2, label="DNNQ Std Dev")
plt.xlabel(r'$Q_M$', fontsize=14)
plt.ylabel(r'$f_{DNNQ}(Q_M)$', fontsize=14)
plt.title('Comparison of Analytical $\mathcal{B}(Q_M)$ and DNNQ Model', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig(f"{results_folder}/QM_comparison_plot.pdf")
#plt.show()

# Compute and Save Final Results
pseudo_df["A_pred"], pseudo_df["A_std"] = compute_A(
    pseudo_df['x1'].values, pseudo_df['x2'].values, pseudo_df['qT'].values, pseudo_df['QM'].values
)
pseudo_df.to_csv(f"{results_folder}/Results_pseudodata_sk1.csv", index=False)

print("Results saved successfully!")