import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import lhapdf
from functions_and_constants import *
from matplotlib.backends.backend_pdf import PdfPages
import random
from pathlib import Path

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# Define the Progressive DNN Model
def build_progressive_model(input_shape=(1,), depth=4, width=256, 
                           L1_reg=1e-12, initializer_range=0.1,
                           use_residual=False, activation='relu', 
                           output_activation='linear', name=None):
    """
    Build a model with `depth` hidden layers of size `width`.
    Residual connections and L1 regularization can be enabled.
    """
    initializer = tf.keras.initializers.RandomUniform(minval=-initializer_range,
                                                     maxval=initializer_range)
    regularizer = tf.keras.regularizers.L1(L1_reg)
    inp = tf.keras.Input(shape=input_shape, name="input")
    x = tf.keras.layers.Dense(width, activation=activation,
                             kernel_initializer=initializer,
                             kernel_regularizer=regularizer)(inp)
    hidden_layers = [x]
    for i in range(1, depth):
        dense = tf.keras.layers.Dense(width, activation=activation,
                                     kernel_initializer=initializer,
                                     kernel_regularizer=regularizer,
                                     activity_regularizer=regularizer,
                                     name=f"dense_{i}_{np.random.randint(10000)}")
        h = dense(hidden_layers[-1])
        if use_residual:
            x = tf.keras.layers.Add()([hidden_layers[-1], h])
        else:
            x = h
        hidden_layers.append(x)
    out = tf.keras.layers.Dense(1, activation=output_activation,
                               kernel_initializer=initializer)(x)
    model = tf.keras.Model(inputs=inp, outputs=out, name=name)
    return model, hidden_layers

# Define Progressive DNNB
def DNNB(name):
    model, _ = build_progressive_model(
        input_shape=(1,), 
        depth=3,  
        width=100,  
        use_residual=False,
        activation='relu',
        output_activation='linear',
        name=name
    )
    return model

# Define Progressive DNNS
def DNNS(name):
    model, _ = build_progressive_model(
        input_shape=(1,), 
        depth=3,  
        width=100,  
        use_residual=False,
        initializer_range=0.1,
        activation='relu', 
        output_activation='softplus', 
        name=name
    )
    return model

def SB_model():
    qT = tf.keras.Input(shape=(1,), name='qT')
    QM = tf.keras.Input(shape=(1,), name='QM')

    SModel = DNNS('SqT')
    BModel = DNNB('BQM')

    Sq = SModel(qT)
    BQM = BModel(QM)

    SB = tf.keras.layers.Multiply()([Sq, BQM])
    return tf.keras.Model([qT, QM], SB)

initial_lr = 0.002
epochs = 1000
batch_size = 8

modify_LR = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',factor=0.9,patience=100,mode='auto')

# Load PDF Set
NNPDF4_nlo = lhapdf.mkPDF('NNPDF40_nlo_as_01180')
alpha = 1/137
hc_factor = 3.89 * 10**8

def pdf(pdfset, flavor, x, QQ):
    return pdfset.xfxQ(flavor, x, QQ)

# Create necessary folders using Path for better cross-platform compatibility
def create_folders(folder_name):
    folder_path = Path(folder_name)
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"Folder '{folder_name}' created successfully!")

models_folder = Path('Models')
loss_plot_folder = Path('Loss_Plots')
replica_data_folder = Path('Replica_Data')
data_folder = Path('Data_with_weights')

for folder in [models_folder, loss_plot_folder, replica_data_folder, data_folder]:
    create_folders(folder)

# Function to handle data paths and loading
def load_data(base_path, file_names):
    data_frames = {}
    for file_name in file_names:
        file_path = Path(base_path) / file_name
        try:
            data_frames[file_name] = pd.read_csv(file_path)
            print(f"Successfully loaded {file_name}")
        except FileNotFoundError:
            print(f"Warning: File {file_path} not found")
    return data_frames


# Load Data
E288_200_initial = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated/E288_200.csv")
E288_300_initial = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated/E288_300.csv")
E288_400_initial = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated/E288_400.csv")

# def add_weight_column(dataframe, beamenergy, user_choice):
#     w_200_45 = 10
#     w_200_55 = 10
#     w_200_65 = 10
#     w_400_55 = 1000
#     w_400_65 = 1000
#     w_400_75 = 1000
#     if user_choice.lower() == 'y':
#         dataframe['weight'] = np.where(dataframe['QM'] < threshold_QM, weight_value, 1)
#     elif user_choice.lower() == 'n':
#         dataframe['weight'] = 1
#     else:
#         print("Invalid choice. Please enter 'y' or 'n'.")
#         return dataframe
#     return dataframe

def add_weight_column(dataframe, beamenergy, user_choice):

    w_200_45 = 100
    w_200_55 = 100
    w_200_65 = 100
    w_400_55 = 1000
    w_400_65 = 1000
    w_400_75 = 1000
    
    dataframe['weight'] = 1
    
    if user_choice.lower() == 'y':
        if beamenergy == 200:
            dataframe.loc[dataframe['QM'] == 4.5, 'weight'] = w_200_45
            dataframe.loc[dataframe['QM'] == 5.5, 'weight'] = w_200_55
            dataframe.loc[dataframe['QM'] == 6.5, 'weight'] = w_200_65
        elif beamenergy == 400:
            dataframe.loc[dataframe['QM'] == 5.5, 'weight'] = w_400_55
            dataframe.loc[dataframe['QM'] == 6.5, 'weight'] = w_400_65
            dataframe.loc[dataframe['QM'] == 7.5, 'weight'] = w_400_75
    
    elif user_choice.lower() == 'n':
        pass
    else:
        print("Invalid choice. Please enter 'y' or 'n'.")
        return dataframe
    
    return dataframe

# E288_200 = add_weight_column(E288_200_initial, 1000, 7.0, 'y')
# E288_300 = add_weight_column(E288_300_initial, 1, 4.0, 'n')
# E288_400 = add_weight_column(E288_400_initial, 1000, 7.7, 'y')


E288_200 = add_weight_column(E288_200_initial, 200,'y')
E288_300 = add_weight_column(E288_300_initial, 300,'n')
E288_400 = add_weight_column(E288_400_initial, 400,'y')


data_folder = 'Data_with_weights'
create_folders(data_folder)
E288_200.to_csv(str(data_folder)+'/'+'E288_200_with_weights.csv')
E288_300.to_csv(str(data_folder)+'/'+'E288_300_with_weights.csv')
E288_400.to_csv(str(data_folder)+'/'+'E288_400_with_weights.csv')



# Combine datasets
data = pd.concat([E288_200, E288_300, E288_400], ignore_index=True)

def QM_int(QM):
    return (-1) / (2 * QM**2)

def compute_QM_integrals(QM_array):
    QM_integrated = QM_int(QM_array + 0.5) - QM_int(QM_array - 0.5)
    return QM_integrated

def fDNNQ(QM):
    return 1 + 0*QM

def GenerateReplicaData(df):
    # df=df[df['qT'] < 0.2 * df['QM']]
    df = df[(9.0 > df['QM']) | (df['QM'] > 11.0)]
    pseudodata_df = {'x1': [],
                     'x2': [],
                     'qT': [],
                     'QM': [],
                     'SqrtS': [],
                     'weight': [],
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
    pseudodata_df['SqrtS'] = df['SqrtS']
    pseudodata_df['weight'] = df['weight']
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

    PDFs = (eu2*np.array(f_u_x1) * np.array(f_ubar_x2) + 
            eu2*np.array(f_u_x2) * np.array(f_ubar_x1) + 
            ed2*np.array(f_d_x1) * np.array(f_dbar_x2) + 
            ed2*np.array(f_d_x2) * np.array(f_dbar_x1) + 
            es2*np.array(f_s_x1) * np.array(f_sbar_x2) + 
            es2*np.array(f_s_x2) * np.array(f_sbar_x1))

    pseudodata_df['PDFs'] = PDFs

    QM_integral = compute_QM_integrals(tempQM)
    pseudodata_df['QM_int'] = QM_integral
    # B_QM = ReplicaA / (hc_factor * factor * PDFs * Sk_contribution * QM_integral)

    SB = ReplicaA / (hc_factor * factor * PDFs * QM_integral)

    pseudodata_df['SB_calc'] = SB

    return pd.DataFrame(pseudodata_df)

# Compute A Predictions
def compute_A(model, x1, x2, qT, QM):
    # Get Predictions from All Models
    SB_DNN_mean = model.predict([qT,QM], verbose=0).flatten()

    factor_temp = factor

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

    ux1ubarx2_temp = eu2*np.array(f_u_x1)*np.array(f_ubar_x2)
    ubarx1ux2_temp = eu2*np.array(f_u_x2)*np.array(f_ubar_x1)
    dx1dbarx2_temp = ed2*np.array(f_d_x1)*np.array(f_dbar_x2)
    dbarx1dx2_temp = ed2*np.array(f_d_x2)*np.array(f_dbar_x1)
    sx1sbarx2_temp = es2*np.array(f_s_x1)*np.array(f_sbar_x2)
    sbarx1sx2_temp = es2*np.array(f_s_x2)*np.array(f_sbar_x1)
    PDFs_temp = ux1ubarx2_temp + ubarx1ux2_temp + dx1dbarx2_temp + dbarx1dx2_temp + sx1sbarx2_temp + sbarx1sx2_temp

    QM_integral_temp = compute_QM_integrals(QM)
    A_pred = SB_DNN_mean * factor_temp * PDFs_temp * hc_factor * QM_integral_temp
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
    pdf_path = replica_data_folder / f"Replica_{rep_num}_Result.pdf"
    with PdfPages(pdf_path) as pdf:
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
        print(f"Subplots saved successfully in {pdf_path}")


def gen_SB_plots(model, df,replica_id):
    # Generate QM Range for Comparison
    QM_test = np.linspace(df['QM'].min(), df['QM'].max(), 200)
    qT_test = np.linspace(df['qT'].min(), df['qT'].max(), 200)

    dnnQ_contributions = model.predict([qT_test,QM_test], verbose=0).flatten()
    
    # Plot Analytical vs. Model Predictions
    plt.figure(figsize=(10, 6))
    #plt.plot(QM_test, fDNNQ_values, label=r'Analytical $\mathcal{B}(Q_M)$', linestyle='--', color='blue')
    plt.plot(QM_test, dnnQ_contributions, label='DNNQ Model Mean', linestyle='-', color='red')
    plt.xlabel(r'$Q_M$', fontsize=14)
    plt.ylabel(r'$f_{DNNQ}(Q_M)$', fontsize=14)
    plt.title('Comparison of Analytical $\mathcal{B}(Q_M)$ and DNNQ Model', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig(f"{replica_data_folder}/QM_comparison_plot_{replica_id}.pdf")
    plt.close()



def split_data(X,y,split=0.1):
  temp =np.random.choice(list(range(len(y))), size=int(len(y)*split), replace = False)

  test_X = pd.DataFrame.from_dict({k: v[temp] for k,v in X.items()})
  train_X = pd.DataFrame.from_dict({k: v.drop(temp) for k,v in X.items()})

  test_y = y[temp]
  train_y = y.drop(temp)

  return train_X, test_X, train_y, test_y


def custom_weighted_loss(y_true, y_pred, w):

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    w = tf.cast(w, tf.float32)
    
    mean_w = tf.reduce_mean(w)
    # weights = w / mean_w
    weights = w

    squared_error = tf.square(y_pred - y_true)
    
    weighted_squared_error = squared_error * weights
    
    return tf.reduce_mean(weighted_squared_error)


class WeightedValidationLoss(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, w_test):
        super(WeightedValidationLoss, self).__init__()
        self.validation_data = validation_data
        self.w_test = w_test
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        x_test, y_test = self.validation_data
        
        y_pred = self.model.predict(x_test, verbose=0)
        
        val_loss = custom_weighted_loss(y_test, y_pred, self.w_test)
        
        logs['val_loss'] = val_loss
        print(f"\nEpoch {epoch+1}: weighted_val_loss = {val_loss:.4f}")



def train_progressively(model, X_train, y_train, X_val, y_val, w_train, w_val,
                      freeze_prev=True, epochs_per_stage=100,
                      batch_size=8, learning_rate=0.002):
    """
    Progressive training with weighted loss, using Keras' built-in sample_weight.
    """
    # Extract the trainable layers from the model
    trainable_layers = [layer for layer in model.layers if len(layer.trainable_weights) > 0]
    depth = len(trainable_layers)
    
    all_history = []
    for stage in range(1, depth + 1):
        print(f"\n--- Training Stage {stage}/{depth} ---")
        
        # Configure layer trainability
        if freeze_prev:
            # Freeze all layers before current stage
            for i, layer in enumerate(trainable_layers):
                layer.trainable = (i >= stage - 1)
            print(f"Trainable layers: {[layer.name for layer in model.layers if layer.trainable]}")
        else:
            # Make all layers trainable
            for layer in trainable_layers:
                layer.trainable = True
                
        # Configure optimizer with current learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Define a weighted loss function for model compilation
        def train_weighted_loss(y_true, y_pred):
            return custom_weighted_loss(y_true, y_pred, w_train)
        
        # Compile with our custom loss function
        model.compile(optimizer=optimizer, loss=train_weighted_loss)
        
        # Setup callbacks
        callbacks = [modify_LR]  # Use your existing ReduceLROnPlateau callback
        
        # Train for this stage, using sample_weight parameter
        history = model.fit(
            x=[X_train[0], X_train[1]],  # List of input arrays
            y=y_train,                    # Target values
            validation_data=([X_val[0], X_val[1]], y_val, w_val),  # Include val weights
            epochs=epochs_per_stage,
            batch_size=batch_size,
            sample_weight=w_train,  # This is key - weights passed to Keras
            callbacks=callbacks,
            verbose=1
        )
        
        all_history.append(history)
    
    return model, all_history


# Enhanced replica model training function
def replica_model(i):
    print(f"\n=== Training Replica Model {i} ===")
    
    # Generate replica data
    E288_200_Replica = GenerateReplicaData(E288_200)
    E288_300_Replica = GenerateReplicaData(E288_300)
    E288_400_Replica = GenerateReplicaData(E288_400)

    replica_data = pd.concat([E288_200_Replica, E288_300_Replica, E288_400_Replica], ignore_index=True)
    replica_data.to_csv(replica_data_folder / f"replica_data_{i}.csv")

    # Extract features and target
    prep_A = replica_data['A_replica']
    prep_features = replica_data.drop(['A_replica'], axis=1)
    
    # Split data into training and testing sets
    train_X, test_X, train_A, test_A = split_data(prep_features, prep_A)

    # Extract features and weights
    qT_train = np.array(train_X['qT']).reshape(-1, 1)
    QM_train = np.array(train_X['QM']).reshape(-1, 1)
    SB_train = np.array(train_X['SB_calc'])
    weights_train = np.array(train_X['weight'])

    qT_test = np.array(test_X['qT']).reshape(-1, 1)
    QM_test = np.array(test_X['QM']).reshape(-1, 1)
    SB_test = np.array(test_X['SB_calc'])
    weights_test = np.array(test_X['weight'])

    print(f"Training set: {len(SB_train)} samples")
    print(f"Testing set: {len(SB_test)} samples")
    print(f"Weight range: {np.min(weights_train)} - {np.max(weights_train)}")

    # Build model
    dnnSB = SB_model()
    dnnSB.summary()
    
    # Train model with weighted loss
    trained_model, all_history = train_progressively(
        dnnSB,
        X_train=[qT_train, QM_train],
        y_train=SB_train,
        X_val=[qT_test, QM_test],
        y_val=SB_test,
        w_train=weights_train,
        w_val=weights_test,
        freeze_prev=True,
        epochs_per_stage=epochs // 3,
        batch_size=batch_size,
        learning_rate=initial_lr
    )
    
    # Save model
    model_path = models_folder / f'DNNB_progressive_weighted_model_{i}.h5'
    trained_model.save(model_path)
    print(f"Progressive Model {i} saved successfully at {model_path}!")

    # Generate plots
    generate_subplots(trained_model, replica_data, i)
    gen_SB_plots(trained_model, replica_data, i)
    
    # Plot combined loss from all stages
    plt.figure(figsize=(10, 6))
    for stage, history in enumerate(all_history):
        plt.plot(range(stage*len(history.history['loss']), (stage+1)*len(history.history['loss'])), 
                 history.history['loss'], label=f'Stage {stage+1} Training Loss')
        plt.plot(range(stage*len(history.history['val_loss']), (stage+1)*len(history.history['val_loss'])), 
                 history.history['val_loss'], label=f'Stage {stage+1} Validation Loss', linestyle='--')
    
    plt.title(f'Progressive Model {i} Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(loss_plot_folder, f'progressive_loss_plot_model_{i}.pdf')
    plt.savefig(loss_plot_path)
    print(f"Loss plot for Progressive Model {i} saved successfully at {loss_plot_path}!")

# Train multiple replicas
for i in range(Num_Replicas):
    replica_model(i)