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


    x1_values = tf.constant(df['x1'].values, dtype=tf.float32)
    x2_values = tf.constant(df['x2'].values, dtype=tf.float32)
    qT_values = tf.constant(df['qT'].values, dtype=tf.float32)
    QM_values = tf.constant(df['QM'].values, dtype=tf.float32)

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


######################################## 
### Generating check plots

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


def gen_B_plots(model, df,replica_id):
    # Generate QM Range for Comparison
    QM_test = np.linspace(df['QM'].min(), df['QM'].max(), 200)
    fDNNQ_values = fDNNQ(QM_test)

    # Get Model Predictions for B(QM)
    dnnQ_contributions = model.predict(QM_test, verbose=0).flatten()
    
    # Plot Analytical vs. Model Predictions
    plt.figure(figsize=(10, 6))
    plt.plot(QM_test, fDNNQ_values, label=r'Analytical $\mathcal{B}(Q_M)$', linestyle='--', color='blue')
    plt.plot(QM_test, dnnQ_contributions, label='DNNQ Model Mean', linestyle='-', color='red')
    plt.xlabel(r'$Q_M$', fontsize=14)
    plt.ylabel(r'$f_{DNNQ}(Q_M)$', fontsize=14)
    plt.title('Comparison of Analytical $\mathcal{B}(Q_M)$ and DNNQ Model', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig(f"{replica_data_folder}/QM_comparison_plot_{replica_id}.pdf")
    plt.close()



##########################################
##########################################


def DNNB():
    return tf.keras.Sequential([
        tf.keras.Input(shape=(1,)), 
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])



def split_data(X,y,split=0.1):
  temp =np.random.choice(list(range(len(y))), size=int(len(y)*split), replace = False)

  test_X = pd.DataFrame.from_dict({k: v[temp] for k,v in X.items()})
  train_X = pd.DataFrame.from_dict({k: v.drop(temp) for k,v in X.items()})

  test_y = y[temp]
  train_y = y.drop(temp)

  return train_X, test_X, train_y, test_y



def replica_model(i):
    # Generate replica data
    replica_data = GenerateReplicaData(data)
    replica_data.to_csv(f"{replica_data_folder}/replica_data_{i}.csv")

    generate_replica_A_subplots(replica_data, i)
    generate_replica_B_subplots(replica_data, i)
    generate_ratio_subplots(replica_data, i)

    # prep_features, prep_A, prep_dA = makeData_for_split(replica_data)
    prep_A = replica_data['A_replica']
    prep_features = replica_data.drop(['A_replica'], axis=1)

    delta = 0.1
    # # Condition to cut large fluctuations/deviations of B(QM) with respect to qT
    # replica_data = replica_data[(replica_data['A_ratio'] <= 1 + delta) & (replica_data['A_ratio'] >= 1 - delta) | 
    #                         (replica_data['B_ratio'] <= 1 + delta) & (replica_data['B_ratio'] >= 1 - delta)]

    train_X, test_X, train_A, test_A = split_data(prep_features, prep_A)

    # Training data
    train_QM_values = np.array(train_X['QM'])
    train_B_QM = np.array(train_X['B_calc'])
    train_A_replica = np.array(train_A)
    
    train_factor_values = np.array(train_X['factor'])
    train_PDFs_values = np.array(train_X['PDFs'])
    train_SqT_values = np.array(train_X['SqT'])
    train_QM_int_values = np.array(train_X['QM_int'])

    # Validation data
    test_QM_values = np.array(test_X['QM'])
    test_B_QM = np.array(test_X['B_calc'])
    test_A_replica = np.array(test_A)
    
    test_factor_values = np.array(test_X['factor'])
    test_PDFs_values = np.array(test_X['PDFs'])
    test_SqT_values = np.array(test_X['SqT'])
    test_QM_int_values = np.array(test_X['QM_int'])

    initial_lr = 0.001  
    epochs = 1000  
    batch_size = 8

    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=initial_lr,
    #     decay_steps=50,
    #     decay_rate=0.96,
    #     staircase=True
    # )
    
    dnnB = DNNB()
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    
    # Define custom loss function for both training and validation
    def custom_loss(y_true, y_pred, factors, pdfs, sqt, qm_int):
        a_pred = y_pred * hc_factor * factors * pdfs * sqt * qm_int
        return tf.reduce_mean(tf.square(a_pred - y_true))

    # Create a custom training loop to track both training and validation loss
    train_losses = []
    val_losses = []
    
    @tf.function
    def train_step(x, y_true, factors, pdfs, sqt, qm_int):
        with tf.GradientTape() as tape:
            y_pred = dnnB(x, training=True)
            loss = custom_loss(y_true, y_pred, factors, pdfs, sqt, qm_int)
        
        gradients = tape.gradient(loss, dnnB.trainable_variables)
        optimizer.apply_gradients(zip(gradients, dnnB.trainable_variables))
        return loss
    
    @tf.function
    def val_step(x, y_true, factors, pdfs, sqt, qm_int):
        y_pred = dnnB(x, training=False)
        loss = custom_loss(y_true, y_pred, factors, pdfs, sqt, qm_int)
        return loss
    
    # Prepare data for training
    train_QM_tensor = tf.convert_to_tensor(train_QM_values, dtype=tf.float32)
    train_A_tensor = tf.convert_to_tensor(train_A_replica, dtype=tf.float32)
    train_factor_tensor = tf.convert_to_tensor(train_factor_values, dtype=tf.float32)
    train_PDFs_tensor = tf.convert_to_tensor(train_PDFs_values, dtype=tf.float32)
    train_SqT_tensor = tf.convert_to_tensor(train_SqT_values, dtype=tf.float32)
    train_QM_int_tensor = tf.convert_to_tensor(train_QM_int_values, dtype=tf.float32)
    
    # Prepare data for validation
    test_QM_tensor = tf.convert_to_tensor(test_QM_values, dtype=tf.float32)
    test_A_tensor = tf.convert_to_tensor(test_A_replica, dtype=tf.float32)
    test_factor_tensor = tf.convert_to_tensor(test_factor_values, dtype=tf.float32)
    test_PDFs_tensor = tf.convert_to_tensor(test_PDFs_values, dtype=tf.float32)
    test_SqT_tensor = tf.convert_to_tensor(test_SqT_values, dtype=tf.float32)
    test_QM_int_tensor = tf.convert_to_tensor(test_QM_int_values, dtype=tf.float32)
    
    # Create dataset for batching
    train_dataset = tf.data.Dataset.from_tensor_slices((
        train_QM_tensor, train_A_tensor, train_factor_tensor, 
        train_PDFs_tensor, train_SqT_tensor, train_QM_int_tensor
    ))
    train_dataset = train_dataset.batch(batch_size)
    
    # Training loop
    for epoch in range(epochs):
        # Training
        epoch_train_loss = tf.keras.metrics.Mean()
        for x, y, factors, pdfs, sqt, qm_int in train_dataset:
            batch_loss = train_step(x, y, factors, pdfs, sqt, qm_int)
            epoch_train_loss.update_state(batch_loss)
        
        # Validation
        val_loss = val_step(
            test_QM_tensor, test_A_tensor, test_factor_tensor,
            test_PDFs_tensor, test_SqT_tensor, test_QM_int_tensor
        )
        
        # Store losses
        train_losses.append(epoch_train_loss.result().numpy())
        val_losses.append(val_loss.numpy())
        
        # Print progress every 100 epochs
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")
    
    # Save Model
    model_path = os.path.join(models_folder, f'DNNB_model_{i}.h5')
    dnnB.save(model_path)
    print(f"Model {i} saved successfully at {model_path}!")
    generate_subplots(dnnB, replica_data, i)
    gen_B_plots(dnnB, replica_data, i)
    
    # Plot Loss
    plt.figure(1, figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='b')
    plt.plot(val_losses, label='Validation Loss', color='r')
    plt.title(f'Model {i} Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(loss_plot_folder, f'loss_plot_model_{i}.pdf')
    plt.savefig(loss_plot_path)
    print(f"Loss plot for Model {i} saved successfully at {loss_plot_path}!")
    plt.close()
    
    return dnnB, replica_data

# Train replicas
for i in range(10):
    replica_model(i)