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
        depth=3,  # Equivalent to 2 hidden layers in original model
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
        depth=3,  # Equivalent to 3 hidden layers in original model
        width=100,  
        use_residual=False,
        initializer_range=0.1,
        activation='relu', 
        output_activation='softplus',  # Original uses softplus for output
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

# Modified to use relative paths or environment variables
data_path = os.environ.get('DATA_PATH', '/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated')
files = ['E288_200.csv', 'E288_300.csv', 'E288_400.csv']

try:
    data_frames = load_data(data_path, files)
    E288_200_initial = data_frames['E288_200.csv']
    E288_300_initial = data_frames['E288_300.csv']
    E288_400_initial = data_frames['E288_400.csv']
except KeyError as e:
    print(f"Error loading data: {e}")
    # Fallback to sample data or exit gracefully
    raise SystemExit("Critical data files missing. Please check data paths.")

def add_weight_column(dataframe, weight_value, threshold_QM, user_choice):
    """
    Add weight column to dataframe based on QM threshold and user choice.
    Enhanced with validation and error handling.
    """
    if dataframe is None or dataframe.empty:
        print("Warning: Empty dataframe provided")
        return dataframe
    
    if user_choice.lower() == 'y':
        if 'QM' not in dataframe.columns:
            print("Error: QM column not found in dataframe")
            return dataframe
        dataframe['weight'] = np.where(dataframe['QM'] < threshold_QM, weight_value, 1)
    elif user_choice.lower() == 'n':
        dataframe['weight'] = 1
    else:
        print("Invalid choice. Please enter 'y' or 'n'.")
    return dataframe

# Apply weights
E288_200 = add_weight_column(E288_200_initial, 100, 7.0, 'y')
E288_300 = add_weight_column(E288_300_initial, 1, 4.0, 'n')
E288_400 = add_weight_column(E288_400_initial, 100, 7.7, 'y')

# Save weighted data
for name, df in zip(['E288_200', 'E288_300', 'E288_400'], [E288_200, E288_300, E288_400]):
    output_path = data_folder / f'{name}_with_weights.csv'
    df.to_csv(output_path)
    print(f"Saved weighted data to {output_path}")

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

def gen_SB_plots(model, df, replica_id):
    # Generate QM Range for Comparison
    QM_test = np.linspace(df['QM'].min(), df['QM'].max(), 200)
    qT_test = np.linspace(df['qT'].min(), df['qT'].max(), 200)
    
    # Reshape for model input
    QM_test_reshaped = QM_test.reshape(-1, 1)
    qT_test_reshaped = qT_test.reshape(-1, 1)
    
    # Create meshgrid for 2D plotting
    QM_mesh, qT_mesh = np.meshgrid(QM_test, qT_test)
    QM_flat = QM_mesh.flatten().reshape(-1, 1)
    qT_flat = qT_mesh.flatten().reshape(-1, 1)
    
    # Get model predictions
    dnnQ_contributions = model.predict([qT_test_reshaped, QM_test_reshaped], verbose=0).flatten()
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(QM_test, dnnQ_contributions, label='DNNQ Model Mean', linestyle='-', color='red')
    plt.xlabel(r'$Q_M$', fontsize=14)
    plt.ylabel(r'$f_{DNNQ}(Q_M)$', fontsize=14)
    plt.title('Model Predictions across QM Range', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    plot_path = replica_data_folder / f"QM_comparison_plot_{replica_id}.pdf"
    plt.savefig(plot_path)
    plt.close()
    
    # Create 2D surface plot of SB values
    sb_values = model.predict([qT_flat, QM_flat], verbose=0).reshape(qT_mesh.shape)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(qT_mesh, QM_mesh, sb_values, cmap='viridis', alpha=0.8)
    ax.set_xlabel(r'$q_T$', fontsize=14)
    ax.set_ylabel(r'$Q_M$', fontsize=14)
    ax.set_zlabel(r'$S \times B$', fontsize=14)
    ax.set_title(f'SB Function for Replica {replica_id}', fontsize=16)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    surface_plot_path = replica_data_folder / f"SB_surface_plot_{replica_id}.pdf"
    plt.savefig(surface_plot_path)
    plt.close()
    
    print(f"SB plots saved at {plot_path} and {surface_plot_path}")

# Enhanced weighted loss function with better error handling
def custom_weighted_loss(y_true, y_pred, w):
    """
    Custom weighted loss function with proper type casting and reshaping.
    Handles edge cases gracefully.
    """
    # Cast to float32 for better compatibility
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    w = tf.cast(w, tf.float32)
    
    # Ensure weights are properly shaped
    if len(w.shape) < len(y_true.shape):
        w = tf.reshape(w, [-1, 1])
    
    # Calculate weighted squared error
    squared_error = tf.square(y_pred - y_true)
    weighted_squared_error = squared_error * w
    
    # Return mean, handling empty tensors
    return tf.reduce_mean(weighted_squared_error)

# Improved callback for weighted validation loss
class WeightedValidationLoss(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, w_test):
        super(WeightedValidationLoss, self).__init__()
        self.validation_data = validation_data
        self.w_test = w_test
        self.val_losses = []  # Track validation losses
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        x_test, y_test = self.validation_data
        
        # Get predictions and calculate loss
        y_pred = self.model.predict(x_test, verbose=0)
        val_loss = custom_weighted_loss(y_test, y_pred, self.w_test)
        
        # Store and log the loss
        self.val_losses.append(float(val_loss))
        logs['val_loss'] = float(val_loss)
        
        # Print progress update every 10 epochs to reduce output verbosity
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch+1}: weighted_val_loss = {val_loss:.4f}")

# Function to split data with better handling
def split_data(X, y, split=0.1):
    """
    Split data into training and testing sets with improved handling
    for weights and different input formats.
    """
    if isinstance(X, pd.DataFrame):
        indices = np.random.choice(X.index, size=int(len(X)*split), replace=False)
        test_X = X.loc[indices]
        train_X = X.drop(indices)
    else:  # Assume it's a dict
        indices = np.random.choice(len(y), size=int(len(y)*split), replace=False)
        test_X = {k: v[indices] for k, v in X.items()}
        train_X = {k: np.delete(v, indices, axis=0) for k, v in X.items()}
    
    if isinstance(y, pd.Series):
        test_y = y.loc[indices]
        train_y = y.drop(indices)
    else:  # Assume it's a numpy array
        test_y = y[indices]
        train_y = np.delete(y, indices, axis=0)
    
    return train_X, test_X, train_y, test_y

# Enhanced progressive training with weighted loss
def train_progressively(model, X_train, y_train, X_val, y_val, w_train, w_val,
                      freeze_prev=True, epochs_per_stage=100,
                      batch_size=8, learning_rate=0.002):
    """
    Enhanced progressive training with properly integrated weighted loss
    and more robust layer freezing.
    """
    # Create a custom weighted loss function that uses the training weights
    def train_weighted_loss(y_true, y_pred):
        return custom_weighted_loss(y_true, y_pred, w_train)
    
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
        model.compile(optimizer=optimizer, loss=train_weighted_loss)
        
        # Setup validation callback
        validation_callback = WeightedValidationLoss(
            validation_data=(X_val, y_val),
            w_test=w_val
        )
        
        # Train for this stage
        history = model.fit(
            x=X_train,
            y=y_train,
            validation_data=(X_val, y_val),
            epochs=epochs_per_stage,
            batch_size=batch_size,
            callbacks=[modify_LR, validation_callback],
            verbose=1
        )
        
        # Store history with custom validation loss
        history.history['val_loss'] = validation_callback.val_losses
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