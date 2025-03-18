import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import lhapdf
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
                     'PDFs':[],
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
        ReplicaA = np.random.normal(loc=tempA, scale=1.0*tempAerr)
        if np.all(ReplicaA > 0):
            break
    pseudodata_df['A_replica'] = ReplicaA

    pseudodata_df['A_ratio'] = ReplicaA/tempA


    # Extract Values
    x1_values = tf.constant(df['x1'].values, dtype=tf.float32)
    x2_values = tf.constant(df['x2'].values, dtype=tf.float32)
    qT_values = tf.constant(df['qT'].values, dtype=tf.float32)
    QM_values = tf.constant(df['QM'].values, dtype=tf.float32)

    # Constants
    alpha = tf.constant(1/137, dtype=tf.float32)
    hc_factor = 3.89 * 10**8
    factor = ((4 * np.pi * alpha) ** 2) / (9 * 2 * np.pi)

    pseudodata_df['factor'] = factor*hc_factor

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
    B_QM = ReplicaA / (factor * PDFs * Sk_contribution * QM_integral)

    pseudodata_df['B_calc'] = B_QM
    pseudodata_df['B_true'] = tempBtrue

    pseudodata_df['B_ratio'] = B_QM/tempBtrue

    return pd.DataFrame(pseudodata_df)


class CrossSectionLayer(tf.keras.layers.Layer):
    def __init__(self, hc_factor, **kwargs):
        super(CrossSectionLayer, self).__init__(**kwargs)
        self.hc_factor = hc_factor
        
    def call(self, inputs):
        # inputs is a list: [B_pred, factor, PDFs, SqT, QM_int]
        B_pred, factor, PDFs, SqT, QM_int = inputs
        # Calculate cross-section: B_pred * hc_factor * factor * PDFs * SqT * QM_int
        return B_pred * self.hc_factor * factor * PDFs * SqT * QM_int
    
    def get_config(self):
        config = super(CrossSectionLayer, self).get_config()
        config.update({'hc_factor': self.hc_factor})
        return config


def DNNB_A():
    # Input for QM
    qm_input = tf.keras.Input(shape=(1,), name='qm_input')
    
    # Additional inputs for conversion factors
    factor_input = tf.keras.Input(shape=(1,), name='factor_input')
    pdfs_input = tf.keras.Input(shape=(1,), name='pdfs_input')
    sqt_input = tf.keras.Input(shape=(1,), name='sqt_input')
    qm_int_input = tf.keras.Input(shape=(1,), name='qm_int_input')
    
    # B prediction layers
    x = tf.keras.layers.Dense(100, activation='relu')(qm_input)
    x = tf.keras.layers.Dense(100, activation='relu')(x)
    b_pred = tf.keras.layers.Dense(1, activation='linear')(x)
    
    # Cross-section calculation layer
    a_pred = CrossSectionLayer(1.0)([b_pred, factor_input, pdfs_input, sqt_input, qm_int_input])
    
    # Create model with QM input and A output
    model = tf.keras.Model(
        inputs=[qm_input, factor_input, pdfs_input, sqt_input, qm_int_input],
        outputs=a_pred
    )
    
    return model


# Train the Model: Using custom model with CrossSectionLayer
def replica_model(i):
    # Generate replica data
    replica_data = GenerateReplicaData(data)
    replica_data.to_csv(f"{replica_data_folder}/replica_data_{i}.csv")

    # After creating replica_data DataFrame, explicitly convert each column:
    replica_data['QM'] = replica_data['QM'].astype('float32')
    replica_data['factor'] = replica_data['factor'].astype('float32')
    replica_data['PDFs'] = replica_data['PDFs'].astype('float32')
    replica_data['SqT'] = replica_data['SqT'].astype('float32')
    replica_data['QM_int'] = replica_data['QM_int'].astype('float32')
    replica_data['A_replica'] = replica_data['A_replica'].astype('float32')

    QM_values = np.array(replica_data['QM'], dtype='float32').reshape(-1, 1)
    factor_values = np.array(replica_data['factor'], dtype='float32').reshape(-1, 1)
    PDFs_values = np.array(replica_data['PDFs'], dtype='float32').reshape(-1, 1)
    SqT_values = np.array(replica_data['SqT'], dtype='float32').reshape(-1, 1)
    QM_int_values = np.array(replica_data['QM_int'], dtype='float32').reshape(-1, 1)
    A_replica_values = np.array(replica_data['A_replica'], dtype='float32').reshape(-1, 1)

    # # Prepare inputs
    # QM_values = np.array(replica_data['QM']).reshape(-1, 1)
    # factor_values = np.array(replica_data['factor']).reshape(-1, 1)
    # PDFs_values = np.array(replica_data['PDFs']).reshape(-1, 1)
    # SqT_values = np.array(replica_data['SqT']).reshape(-1, 1)
    # QM_int_values = np.array(replica_data['QM_int']).reshape(-1, 1)
    
    # Target values: A_replica
    A_replica_values = np.array(replica_data['A_replica']).reshape(-1, 1)
    
    initial_lr = 0.001
    epochs = 500
    batch_size = 200
    
    # Create and compile model
    model = DNNB_A()
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    model.compile(optimizer=optimizer, loss='mse')
    
    # Train model
    history = model.fit(
        x=[QM_values, factor_values, PDFs_values, SqT_values, QM_int_values],
        y=A_replica_values,
        epochs=epochs,
        batch_size=batch_size,
        verbose=2
    )
    
    # Save model
    model_path = os.path.join(models_folder, f'DNNB_A_model_{i}.h5')
    model.save(model_path)
    print(f"Model {i} saved successfully at {model_path}!")
    
    # Plot loss
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
    
    return model, replica_data


# Train replicas
for i in range(3):
    replica_model(i)