import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the neural network for S(k)
def build_model():
    model = keras.Sequential([
        layers.Input(shape=(1,)),  # Input: k
        layers.Dense(64, activation='tanh'),
        layers.Dense(64, activation='tanh'),
        layers.Dense(1, activation='linear')  # Output: S(k)
    ])
    return model

# Define Monte Carlo integration for A(qT)
def compute_A(model, qT, N_k=10, N_phi=10, k_max=2.0):
    k_samples = np.random.uniform(0, k_max, N_k)
    phi_samples = np.random.uniform(0, 2 * np.pi, N_phi)
    
    A_values = []
    for q in qT:
        integral_sum = 0
        for k in k_samples:
            S_k = model.predict(np.array([[k]]), verbose=0)[0, 0]
            for phi in phi_samples:
                k_prime = np.sqrt(q**2 + k**2 - 2 * q * k * np.cos(phi))
                S_kprime = model.predict(np.array([[k_prime]]), verbose=0)[0, 0]
                integral_sum += S_k * S_kprime
        
        A_qT = (2 * np.pi / N_phi) * (k_max**2 / N_k) * integral_sum
        A_values.append(A_qT)
    
    return np.array(A_values)

# Generate synthetic training data
def generate_training_data(n_samples=10):
    qT_values = np.random.uniform(0, 2.0, n_samples)
    A_values = np.sin(qT_values)  # Example function for A(qT)
    return qT_values, A_values

# Training procedure
model = build_model()
optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss_fn = keras.losses.MeanSquaredError()

qT_train, A_train = generate_training_data()

batch_size = 32
epochs = 100
for epoch in range(epochs):
    indices = np.random.permutation(len(qT_train))
    qT_train, A_train = qT_train[indices], A_train[indices]
    
    for i in range(0, len(qT_train), batch_size):
        qT_batch = qT_train[i:i+batch_size]
        A_batch = A_train[i:i+batch_size]
        
        with tf.GradientTape() as tape:
            A_pred = compute_A(model, qT_batch)
            loss = loss_fn(A_batch, A_pred)
        
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy()}")
