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

# Load Data
E288_200 = pd.read_csv("gen_pseudo_data/pseudodata_E288_200.csv")
E288_300 = pd.read_csv("gen_pseudo_data/pseudodata_E288_300.csv")
E288_400 = pd.read_csv("gen_pseudo_data/pseudodata_E288_400.csv")
E605 = pd.read_csv("gen_pseudo_data/pseudodata_E605.csv")
E772 = pd.read_csv("gen_pseudo_data/pseudodata_E772.csv")

#data = pd.concat([E288_200,E288_300,E288_400,E605,E772])
data = pd.concat([E288_200])

models_folder = 'Models'


x1_values = tf.constant(data['x1'].values, dtype=tf.float32)
x2_values = tf.constant(data['x2'].values, dtype=tf.float32)
qT_values = tf.constant(data['qT'].values, dtype=tf.float32)
QM_values = tf.constant(data['QM'].values, dtype=tf.float32)
A_true_values = tf.constant(data['A'].values, dtype=tf.float32)




alpha = tf.constant(1/137, dtype=tf.float32)  # Fine structure constant


hc_factor = 3.89*10**8


factor = ((4*np.pi*alpha)**2)/(9*2*np.pi)




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


mm = 0.5

def S(k):
    return ((k**2)/(mm*np.pi))*np.exp(-(k**2)/mm)
    
def Sk_cont(qT):
    return (8*mm*mm + qT*qT*qT*qT)/(32*np.pi*mm)*(np.exp(-(qT*qT)/(2*mm)))
    
Sk_contribution = Sk_cont(qT_values)

def QM_int(QM):
    return (-1)/(2*QM**2)

def compute_QM_integrals(QM_array):
    QM_array = np.atleast_1d(QM_array) 
    QM_integrated = QM_int(QM_array + 0.5) - QM_int(QM_array - 0.5)
    return QM_integrated[0] if QM_integrated.size == 1 else QM_integrated

QM_integral = compute_QM_integrals(QM_values)

# Assuming the additional parameters PDFs, SqT_integral, QM_integral are available here:
PDFs = PDFs
SqT_integral = Sk_contribution
QM_integral = QM_integral


# Directory where models are saved
models_folder = 'Models'

# Define the custom loss exactly as used during training
# def custom_loss(PDFs, SqT_integral, QM_integral, y_true, y_pred):
#     alpha = 1/137  # Fine-structure constant
#     hc_factor = 3.89 * 10**8
#     factor = ((4 * np.pi * alpha) ** 2) / (9 * 2 * np.pi)
#     # Compute predicted cross-section
#     cross_section = y_pred * hc_factor * factor * PDFs * SqT_integral * QM_integral
#     # Compute Mean Squared Error (MSE) loss
#     return tf.reduce_mean(tf.square(cross_section - y_true))

def custom_loss(y_pred, y_true, PDFs, SqT_integral, QM_integral):
    alpha = 1/137  # Fine-structure constant (define if missing)
    hc_factor = 3.89 * 10**8
    factor = ((4 * np.pi * alpha) ** 2) / (9 * 2 * np.pi)
    # Compute predicted cross-section
    cross_section = y_pred * hc_factor * factor * PDFs * SqT_integral * QM_integral
    # Compute Mean Squared Error (MSE) loss
    return tf.reduce_mean(tf.square(cross_section - y_true))

# Create a wrapper for the loss function to match TensorFlow's expected signature
class CustomLossWrapper(tf.keras.losses.Loss):
    # def __init__(self, PDFs=1.0, SqT_integral=1.0, QM_integral=1.0, **kwargs):
    def __init__(self, PDFs=PDFs, SqT_integral=SqT_integral, QM_integral=QM_integral, **kwargs):
        super().__init__(**kwargs)
        self.PDFs = PDFs
        self.SqT_integral = SqT_integral
        self.QM_integral = QM_integral
    
    def __call__(self, y_true, y_pred, sample_weight=None):
        return custom_loss(
            self.PDFs, 
            self.SqT_integral, 
            self.QM_integral, 
            y_true, 
            y_pred
        )

# # Create a wrapper for the loss function to match TensorFlow's expected signature
# class CustomLossWrapper(tf.keras.losses.Loss):
#     # def __init__(self, PDFs=1.0, SqT_integral=1.0, QM_integral=1.0, **kwargs):
#     def __init__(self, PDFs=PDFs, SqT_integral=SqT_integral, QM_integral=QM_integral, **kwargs):
#         super().__init__(**kwargs)
#         self.PDFs = PDFs
#         self.SqT_integral = SqT_integral
#         self.QM_integral = QM_integral
    
#     def __call__(self, y_true, y_pred, sample_weight=None):
#         return custom_loss(
#             y_pred,
#             y_true,
#             self.PDFs, 
#             self.SqT_integral, 
#             self.QM_integral
#         )

# Load all trained models
try:
    model_files = [f for f in os.listdir(models_folder) if f.endswith('.h5')]
    models_list = []
    
    for f in model_files:
        try:
            model_path = os.path.join(models_folder, f)
            # Try loading the model with compile=False first
            print(f"Attempting to load {f}...")
            model = tf.keras.models.load_model(model_path, compile=False)
            # Recompile with our custom loss wrapper
            model.compile(
                optimizer='adam',
                loss=CustomLossWrapper()
            )
            models_list.append(model)
            print(f"Successfully loaded and recompiled {f}")
        except Exception as e:
            print(f"Error loading model {f}: {str(e)}")
    
    print(f"Loaded {len(models_list)} models from '{models_folder}'.")
except FileNotFoundError:
    print(f"Directory '{models_folder}' not found. Please check the path.")
except Exception as e:
    print(f"Error loading models: {str(e)}")



# Compute B(QM)
B_QM = A_true_values / (hc_factor * factor * PDFs * Sk_contribution * QM_integral)


### Cross-Section #####

# Compute A Predictions
def compute_A(x1, x2, qT, QM):
    # Get Predictions from All Models
    fDNN_contributions = np.array([model.predict(QM, verbose=0).flatten() for model in models_list])
    fDNN_mean = np.mean(fDNN_contributions, axis=0)
    fDNN_std = np.std(fDNN_contributions, axis=0)

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
    A_std = fDNN_std * factor_temp * PDFs_temp * Sk_temp * hc_factor * QM_integral_temp

    return A_pred, A_std



def prep_data_for_plots(df):
    temp_df = {'x1': [],
        'x2': [],
        'qT': [],
        'QM': [],
        'A_true': [],
        'A_true_err': [],
        'A_pred': [],
        'A_pred_err': []}

    qT = df['qT'].values
    QM = df['QM'].values
    x1 = df['x1'].values
    x2 = df['x2'].values
    A_true = df['A'].values
    A_true_err = df['dA'].values
    A_pred, A_std = compute_A(x1, x2, qT, QM)

    temp_df['x1'] = x1
    temp_df['x2'] = x2
    temp_df['qT'] = qT
    temp_df['QM'] = QM
    temp_df['A_true'] = A_true
    temp_df['A_true_err'] = A_true_err
    temp_df['A_pred'] = A_pred
    temp_df['A_pred_err'] = A_std

    return pd.DataFrame(temp_df)


from matplotlib.backends.backend_pdf import PdfPages

def generate_subplots(df):
    prepared_df = prep_data_for_plots(df)
    unique_QM = np.unique(prepared_df['QM'])
    
    # Create a PDF to store subplots
    with PdfPages(f"{results_folder}/Comparison_Subplots.pdf") as pdf:
        fig, axes = plt.subplots(nrows=len(unique_QM) // 2 + len(unique_QM) % 2, ncols=2, figsize=(12, 6 * (len(unique_QM) // 2)))
        axes = axes.flatten()
        
        for i, QM_val in enumerate(unique_QM):
            mask = prepared_df['QM'] == QM_val
            # axes[i].scatter(prepared_df['qT'][mask], prepared_df['A_true'][mask], color='b', label=f'QM = {QM_val:.2f} (True)')
            # axes[i].scatter(prepared_df['qT'][mask], prepared_df['A_pred'][mask], color='r', marker='x', label=f'QM = {QM_val:.2f} (Pred)')
            axes[i].errorbar(prepared_df['qT'][mask], prepared_df['A_true'][mask], yerr=prepared_df['A_true_err'][mask], fmt='bo', label=f'QM = {QM_val:.2f} (True)', capsize=3)
            axes[i].errorbar(prepared_df['qT'][mask], prepared_df['A_pred'][mask], yerr=prepared_df['A_pred_err'][mask], fmt='rx', label=f'QM = {QM_val:.2f} (Pred)', capsize=3)
            axes[i].set_xlabel(r'$q_T$')
            axes[i].set_ylabel(r'$A(q_T)$')
            axes[i].legend()
            axes[i].grid(True)
            
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        print("Subplots saved successfully in Subplots.pdf")

generate_subplots(data)




######### B(QM) #################

# Analytical Function for fDNNQ
# def fDNNQ(QM, b=0.5):
#     return np.exp(-b * QM)

def fDNNQ(QM, b=0.5):
    return b * QM

# Generate QM Range for Comparison
QM_values = np.linspace(data['QM'].min(), data['QM'].max(), 200)
fDNNQ_values = fDNNQ(QM_values)


# Get Model Predictions
dnnQ_contributions = np.array([model.predict(QM_values, verbose=0).flatten() for model in models_list])
dnnQ_mean = np.mean(dnnQ_contributions, axis=0)
dnnQ_std = np.std(dnnQ_contributions, axis=0)



#fDNNQ_values = fDNNQ(QM_values)

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