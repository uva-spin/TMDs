import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

# Constants for the model
k_lower = 0
k_upper = 10
kBins = 10
phiBins = 10

Learning_Rate = 0.00001

NUM_REPLICAS = 1
EPOCHS = 1000
BATCH = 64

# Constants for data generation
NUM_SAMPLES = 100
# QT_FIXED = 1.0
QT_MIN = 0.01
QT_MAX = 4
X_MIN = 0.1
X_MAX = 0.3

def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")

Models_folder = 'DNNmodels'
create_folders('DNNmodels')
create_folders('Losses_Plots')

# Model architecture parameters
Hidden_Layers = 3
Nodes_per_HL = 100
L1_reg = 10**(-12)

def custom_weighted_loss(y_true, y_pred, w=None):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    if w is not None:
        w = tf.cast(w, tf.float32)
        mean_w = tf.reduce_mean(w)
        weights = w / mean_w
        squared_error = tf.square(y_pred - y_true)
        weighted_squared_error = squared_error * weights
        return tf.reduce_mean(weighted_squared_error)
    else:
        return tf.reduce_mean(tf.square(y_pred - y_true))

class CustomWeightedLoss(tf.keras.losses.Loss):
    def __init__(self, name="custom_weighted_loss"):
        super().__init__(name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        return custom_weighted_loss(y_true, y_pred)

CSmodel = tf.keras.models.load_model('averaged_model.h5', 
    custom_objects={
        'custom_weighted_loss': custom_weighted_loss,
        'CustomWeightedLoss': CustomWeightedLoss,
        'train_weighted_loss': custom_weighted_loss
    })

SqT_model = CSmodel.get_layer('SqT')
print("Loaded the averaged model")

def create_nn_model(name):
    inp = tf.keras.Input(shape=(2,))
    initializer = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=42)
    x = tf.keras.layers.Dense(Nodes_per_HL, activation='relu6', 
                              kernel_initializer=initializer, 
                              kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(inp)
    for _ in range(Hidden_Layers - 1):
        x = tf.keras.layers.Dense(Nodes_per_HL, activation='relu6', 
                                  kernel_initializer=initializer, 
                                  kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x)
    nnout = tf.keras.layers.Dense(1, activation='relu6', 
                                  kernel_initializer=initializer, 
                                  kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x)
    return tf.keras.Model(inp, nnout, name=name)

class TMDIntegrationLayer(tf.keras.layers.Layer):
    def __init__(self, k_bins=10, phi_bins=10, k_lower=0.0, k_upper=10.0, **kwargs):
        super().__init__(**kwargs)
        self.k_bins = k_bins
        self.phi_bins = phi_bins
        self.k_lower = float(k_lower)
        self.k_upper = float(k_upper)
        self.k_values = np.linspace(self.k_lower, self.k_upper, self.k_bins, dtype=np.float32)
        self.phi_values = np.linspace(0.0, np.pi, self.phi_bins, dtype=np.float32)
        self.dk = (self.k_upper - self.k_lower) / (self.k_bins - 1)
        self.dphi = np.pi / (self.phi_bins - 1)
        self.modn1 = create_nn_model('n1')
        self.modn2 = create_nn_model('n2')

    def call(self, inputs):
        qT, x1, x2 = inputs
        batch_size = tf.shape(qT)[0]
        result = tf.zeros((batch_size, 1), dtype=tf.float32)

        for k_val in self.k_values:
            k_factor = k_val
            for phi_val in self.phi_values:
                kb_tensor = tf.sqrt(qT**2 + k_val**2 - 2*qT*k_val*tf.cos(phi_val))
                n1_input = tf.concat([x1, tf.ones_like(x1) * k_val], axis=1)
                n2_input = tf.concat([x2, kb_tensor], axis=1)
                nn1_out = self.modn1(n1_input)
                nn2_out = self.modn2(n2_input)
                contribution = nn1_out * nn2_out * k_factor * self.dk * self.dphi
                result += contribution
        return result

def createModel_SqT():
    qT = tf.keras.Input(shape=(1,), name='qT')
    x1 = tf.keras.Input(shape=(1,), name='x1')
    x2 = tf.keras.Input(shape=(1,), name='x2')
    tmd_layer = TMDIntegrationLayer(k_bins=kBins, phi_bins=phiBins, k_lower=k_lower, k_upper=k_upper)
    output = tmd_layer([qT, x1, x2])
    model = tf.keras.Model(inputs=[qT, x1, x2], outputs=output, name='SqT_model')
    model.tmd_layer = tmd_layer
    model.n1 = tmd_layer.modn1
    model.n2 = tmd_layer.modn2
    return model

def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def generate_data(num_samples=NUM_SAMPLES):
    qT = np.random.uniform(QT_MIN, QT_MAX, num_samples).astype(np.float32)
    x1 = np.random.uniform(X_MIN, X_MAX, num_samples).astype(np.float32)
    x2 = np.random.uniform(X_MIN, X_MAX, num_samples).astype(np.float32)
    SqT_input = np.column_stack([qT, x1, x2])
    SqT_pred = SqT_model.predict(SqT_input, verbose=0)
    data = {
        'qT': qT.reshape(-1, 1),
        'x1': x1.reshape(-1, 1),
        'x2': x2.reshape(-1, 1),
        'y': SqT_pred
    }
    return data

def split_data(data, test_ratio=0.2):
    num_samples = len(data['qT'])
    indices = np.random.permutation(num_samples)
    test_size = int(num_samples * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    train_data = {key: val[train_indices] for key, val in data.items()}
    test_data = {key: val[test_indices] for key, val in data.items()}
    return train_data, test_data

def run_replica(i):
    print(f"Starting replica {i}")
    model = createModel_SqT()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=Learning_Rate), loss=mse_loss)
    data = generate_data()
    train_data, test_data = split_data(data)
    history = model.fit(
        [train_data['qT'], train_data['x1'], train_data['x2']],
        train_data['y'],
        validation_data=([test_data['qT'], test_data['x1'], test_data['x2']], test_data['y']),
        epochs=EPOCHS,
        batch_size=BATCH,
        verbose=2
    )
    model.save(f"{Models_folder}/model{i}.h5")
    plt.figure(figsize=(12, 5))
    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Val. loss')
    plt.title(f'Losses for Replica {i}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'Losses_Plots/loss_plots{i}.pdf')
    plt.close()
    return model, history


# Run training for each replica
for i in range(NUM_REPLICAS):
    starttime = datetime.datetime.now().replace(microsecond=0)
    model, history = run_replica(i)
    finishtime = datetime.datetime.now().replace(microsecond=0)
    print('#################################')
    print(f'Completed replica {i}')
    print('##################')
    print('Duration for this replica:')
    print(finishtime - starttime)