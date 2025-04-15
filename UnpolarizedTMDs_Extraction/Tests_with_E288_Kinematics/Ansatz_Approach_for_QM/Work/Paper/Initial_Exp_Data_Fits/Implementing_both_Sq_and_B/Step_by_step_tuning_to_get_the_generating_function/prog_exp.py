import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# --- Synthetic Regression Data (example) ---
def generate_data(n=1000):
    X = np.random.uniform(-2, 2, size=(n, 1))
    y = np.sin(3 * X) + 0.3 * np.random.randn(n, 1)
    return X, y

# --- Progressive DNN Model Definition ---
def ProgressiveDNN(name="ProgressiveDNN", input_dim=1, depth=4, width=500, 
                   initializer_range=0.1, L1_reg=1e-12, use_residual=False):
    initializer = tf.keras.initializers.RandomUniform(minval=-initializer_range, maxval=initializer_range)
    regularizer = tf.keras.regularizers.L1(L1_reg)

    inp = tf.keras.Input(shape=(input_dim,), name='Input')
    x = tf.keras.layers.Dense(width, activation='relu', kernel_initializer=initializer)(inp)
    
    for i in range(depth):
        layer = tf.keras.layers.Dense(width, activation='relu',
                                      kernel_initializer=initializer,
                                      kernel_regularizer=regularizer,
                                      activity_regularizer=regularizer)
        h = layer(x)
        x = x + h if use_residual else h  # Optional residual connection

    out = tf.keras.layers.Dense(1, activation='linear', kernel_initializer=initializer)(x)
    model = tf.keras.Model(inputs=inp, outputs=out, name=name)
    return model

# --- Training and Visualization ---
def train_and_evaluate():
    X, y = generate_data(2000)
    X_train, y_train = X[:1600], y[:1600]
    X_val, y_val = X[1600:], y[1600:]

    model = ProgressiveDNN(depth=5, width=300, use_residual=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='mse', metrics=['mae'])

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=100,
                        batch_size=64,
                        callbacks=[
                            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
                        ])

    # Evaluation
    X_test = np.linspace(-2, 2, 1000).reshape(-1, 1)
    y_pred = model.predict(X_test)

    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, alpha=0.2, label="Training Data")
    plt.plot(X_test, y_pred, color='r', label="Model Prediction", linewidth=2)
    plt.legend()
    plt.title("Progressive DNN Regression")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return model

if __name__ == "__main__":
    model = train_and_evaluate()
