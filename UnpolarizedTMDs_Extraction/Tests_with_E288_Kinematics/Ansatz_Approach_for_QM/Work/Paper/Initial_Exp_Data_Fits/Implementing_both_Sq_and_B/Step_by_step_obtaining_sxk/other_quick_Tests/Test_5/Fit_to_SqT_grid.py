import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

k_lower = 0
k_upper = 10
kBins = 10
phiBins = 10

NUM_REPLICAS = 3

def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")
        
Models_folder = 'DNNmodels'
create_folders('DNNmodels')
create_folders('Losses_Plots')

########### Import pseudodata file 
df = pd.read_csv('results.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')

####### Here we define a function that can sample cross-section within errA ###
def GenerateReplicaData(df):
    pseudodata_df = {'x1': [],
                     'x2': [],
                     'qT': [],
                     'SqT': [],
                     'SqT_err':[]}
    
    pseudodata_df['x1'] = df['x1']
    pseudodata_df['x2'] = df['x2']
    pseudodata_df['qT'] = df['qT']
    pseudodata_df['SqT_err'] = df['SqT_err']
    tempA = df['SqT']
    tempAerr = np.abs(np.array(df['SqT_err'])) 
    pseudodata_df['SqT'] = np.random.normal(loc=tempA, scale=tempAerr)
    return pd.DataFrame(pseudodata_df)


################ Defining the DNN model ####################
Hidden_Layers = 3  # Reduced for simplicity
Nodes_per_HL = 100  # Reduced for simplicity
Learning_Rate = 0.00001
L1_reg = 10**(-12)
EPOCHS = 1000
BATCH = 64

def create_nn_model(name):
    # Input shape is 2: either (x1, k) or (x2, kB)
    inp = tf.keras.Input(shape=(2,))
    initializer = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=42)
    
    # First layer
    x = tf.keras.layers.Dense(Nodes_per_HL, activation='relu6', 
                             kernel_initializer=initializer, 
                             kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(inp)
    
    # Hidden layers
    for _ in range(Hidden_Layers-1):
        x = tf.keras.layers.Dense(Nodes_per_HL, activation='relu6', 
                                 kernel_initializer=initializer, 
                                 kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x)
    
    # Output layer
    nnout = tf.keras.layers.Dense(1, activation='relu6', 
                                 kernel_initializer=initializer, 
                                 kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x)
    
    mod = tf.keras.Model(inp, nnout, name=name)
    return mod

# Complete refactor using a different approach for integration
class TMDIntegrationLayer(tf.keras.layers.Layer):
    def __init__(self, k_bins=10, phi_bins=10, k_lower=0.0, k_upper=10.0, **kwargs):
        super(TMDIntegrationLayer, self).__init__(**kwargs)
        self.k_bins = k_bins
        self.phi_bins = phi_bins
        self.k_lower = float(k_lower)
        self.k_upper = float(k_upper)
        
        # Pre-compute k and phi values
        self.k_values = np.linspace(self.k_lower, self.k_upper, self.k_bins, dtype=np.float32)
        self.phi_values = np.linspace(0.0, np.pi, self.phi_bins, dtype=np.float32)
        
        # Calculate step sizes
        self.dk = float(self.k_upper - self.k_lower) / float(self.k_bins - 1)
        self.dphi = float(np.pi) / float(self.phi_bins - 1)
        
        # Create the neural networks
        self.modn1 = create_nn_model('n1')
        self.modn2 = create_nn_model('n2')
    
    def build(self, input_shape):
        # Nothing to build specifically
        super(TMDIntegrationLayer, self).build(input_shape)
    
    def call(self, inputs):
        qT, x1, x2 = inputs
        
        # Initialize the output tensor with zeros
        batch_size = tf.shape(qT)[0]
        result = tf.zeros((batch_size, 1), dtype=tf.float32)
        
        # Loop through k and phi values
        for k_idx in range(self.k_bins):
            k_val = self.k_values[k_idx]
            k_factor = k_val  # For Jacobian if needed
            
            for phi_idx in range(self.phi_bins):
                phi_val = self.phi_values[phi_idx]
                
                # Calculate kB
                kb_tensor = tf.sqrt(qT**2 + k_val**2 - 2*qT*k_val*tf.cos(phi_val))
                
                # Create input tensors for the neural networks
                n1_input = tf.concat([x1, tf.ones_like(x1) * k_val], axis=1)
                n2_input = tf.concat([x2, kb_tensor], axis=1)
                
                # Get neural network outputs
                nn1_out = self.modn1(n1_input)
                nn2_out = self.modn2(n2_input)
                
                # Multiply, apply integration weights, and add to result
                contribution = nn1_out * nn2_out * k_factor * self.dk * self.dphi
                result += contribution
        
        return result
    

def createModel_SqT():
    # Define inputs
    qT = tf.keras.Input(shape=(1,), name='qT')
    x1 = tf.keras.Input(shape=(1,), name='x1')
    x2 = tf.keras.Input(shape=(1,), name='x2')
    
    # Use the custom integration layer
    output = TMDIntegrationLayer(k_bins=kBins, phi_bins=phiBins, k_lower=k_lower, k_upper=k_upper)([qT, x1, x2])
    
    # Create and return the model
    return tf.keras.Model(inputs=[qT, x1, x2], outputs=output)

# Create the model
model = createModel_SqT()

def mse_loss(y_true, y_pred):
    """Mean squared error loss function."""
    return tf.reduce_mean(tf.square(y_true - y_pred))

def split_data(X, y, yerr, split=0.1):
    """Split data into training and testing sets."""
    indices = list(range(len(y)))
    test_indices = np.random.choice(indices, size=int(len(y)*split), replace=False)
    test_indices_set = set(test_indices)
    
    # Create test and train dictionaries
    test_X = {}
    train_X = {}
    
    for col in X.columns:
        test_X[col] = X[col].iloc[test_indices].values
        train_X[col] = X[col].iloc[[i for i in indices if i not in test_indices_set]].values
    
    # Handle y and yerr
    test_y = y.iloc[test_indices].values
    train_y = y.iloc[[i for i in indices if i not in test_indices_set]].values
    
    test_yerr = yerr.iloc[test_indices].values
    train_yerr = yerr.iloc[[i for i in indices if i not in test_indices_set]].values
    
    return train_X, test_X, train_y, test_y, train_yerr, test_yerr

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=Learning_Rate), loss=mse_loss)

def run_replica(i):
    replica_number = i
    print(f"Starting replica {replica_number}")
    
    # Generate replica data
    tempdf = GenerateReplicaData(df)
    
    # Split data into training and testing sets
    trainKin, testKin, trainA, testA, trainAerr, testAerr = split_data(
        tempdf[['qT', 'x1', 'x2']], tempdf['SqT'], tempdf['SqT_err'], split=0.1)
    
    # Fit the model
    history = model.fit(
        [trainKin['qT'], trainKin['x1'], trainKin['x2']],
        trainA,
        validation_data=([testKin['qT'], testKin['x1'], testKin['x2']], testA),
        epochs=EPOCHS,
        batch_size=BATCH,
        verbose=2
    )
    
    # Save the model
    model.save(f"{Models_folder}/model{replica_number}.h5", save_format='h5')
    
    # Plot and save training history
    plt.figure(figsize=(12, 5))
    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Val. loss')
    plt.title(f'Losses for Replica {replica_number}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'Losses_Plots/loss_plots{replica_number}.pdf')
    plt.close()
    
    return history

import datetime

# Run the replicas
for i in range(NUM_REPLICAS):
    starttime = datetime.datetime.now().replace(microsecond=0)
    history = run_replica(i)
    finishtime = datetime.datetime.now().replace(microsecond=0)
    print('#################################')
    print(f'Completed replica {i}')
    print('##################')
    print('Duration for this replica:')
    print(finishtime - starttime)