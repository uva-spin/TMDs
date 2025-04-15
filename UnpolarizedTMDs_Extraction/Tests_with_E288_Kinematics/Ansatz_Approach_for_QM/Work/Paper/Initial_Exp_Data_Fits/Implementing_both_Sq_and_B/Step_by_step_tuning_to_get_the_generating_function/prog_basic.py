import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# === SET RANDOM SEED FOR REPRODUCIBILITY ===
def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
set_seed(42)


# === GENERATE SYNTHETIC PHYSICS-LIKE DATA ===
def generate_data(n_samples=2000):
    qT = np.random.uniform(0.05, 2.0, size=(n_samples, 1))
    QM = np.random.uniform(3.0, 9.0, size=(n_samples, 1))
    # Target function is designed to resemble a physical response
    SB = np.exp(-qT / QM) * np.sin(QM) + 0.1 * np.random.randn(n_samples, 1)
    X = np.hstack([qT, QM])
    y = SB
    return X, y


# === BUILD PROGRESSIVE MODEL ===
def build_progressive_model(input_shape=(2,), depth=4, width=256,
                            L1_reg=1e-12, initializer_range=0.1,
                            use_residual=False):
    """
    Build a model with `depth` hidden layers of size `width`.
    Residual connections and L1 regularization can be enabled.
    """
    initializer = tf.keras.initializers.RandomUniform(minval=-initializer_range,
                                                      maxval=initializer_range)
    regularizer = tf.keras.regularizers.L1(L1_reg)

    inp = tf.keras.Input(shape=input_shape, name="input")
    x = tf.keras.layers.Dense(width, activation='relu',
                              kernel_initializer=initializer,
                              kernel_regularizer=regularizer)(inp)

    hidden_layers = [x]

    for i in range(1, depth):
        dense = tf.keras.layers.Dense(width, activation='relu',
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

    out = tf.keras.layers.Dense(1, activation='linear',
                                kernel_initializer=initializer)(x)
    model = tf.keras.Model(inputs=inp, outputs=out)
    return model, hidden_layers


# === STAGE-WISE PROGRESSIVE TRAINING ===
def train_progressively(X_train, y_train, X_val, y_val,
                        depth=5, width=256, use_residual=True,
                        freeze_prev=True, epochs_per_stage=30,
                        batch_size=64, learning_rate=1e-3):
    """
    Trains a model stage-by-stage, freezing previous layers.
    """
    input_shape = X_train.shape[1:]
    model, hidden_layers = build_progressive_model(input_shape=input_shape,
                                                   depth=depth,
                                                   width=width,
                                                   use_residual=use_residual)

    all_history = []

    for stage in range(1, depth + 1):
        print(f"\n--- Training Stage {stage}/{depth} ---")

        if freeze_prev:
            for layer in hidden_layers[:stage-1]:
                layer.trainable = False
            for layer in hidden_layers[stage-1:]:
                layer.trainable = True
        else:
            for layer in hidden_layers:
                layer.trainable = True

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='mse',
                      metrics=['mae'])

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs_per_stage,
            batch_size=batch_size,
            verbose=2
        )
        all_history.append(history)

    return model, all_history


# === PLOTTING RESULTS ===
def plot_predictions(model, X_test, y_test, title="Model Prediction"):
    y_pred = model.predict(X_test)
    plt.figure(figsize=(10, 5))
    plt.scatter(X_test[:, 1], y_test, alpha=0.3, label="True", s=10)
    plt.scatter(X_test[:, 1], y_pred, alpha=0.3, label="Predicted", s=10)
    plt.xlabel("QM")
    plt.ylabel("SB")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_loss_curves(history_list):
    plt.figure(figsize=(8, 4))
    for idx, h in enumerate(history_list):
        plt.plot(h.history['loss'], label=f'Train Loss Stage {idx+1}')
        plt.plot(h.history['val_loss'], label=f'Val Loss Stage {idx+1}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Loss per Stage')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# === MAIN RUN ===
if __name__ == "__main__":
    # === TUNABLE PARAMETERS ===
    depth = 5                # Number of hidden layers
    width = 256              # Neurons per hidden layer
    use_residual = True      # Enable/disable residual connections
    freeze_layers = True     # Whether to freeze earlier layers during training
    epochs_per_stage = 25    # Epochs per stage (per added layer)
    learning_rate = 1e-3     # Adam optimizer learning rate
    batch_size = 64

    # === Generate data ===
    X, y = generate_data(2000)
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    # === Train model progressively ===
    model, histories = train_progressively(
        X_train, y_train, X_val, y_val,
        depth=depth,
        width=width,
        use_residual=use_residual,
        freeze_prev=freeze_layers,
        epochs_per_stage=epochs_per_stage,
        batch_size=batch_size,
        learning_rate=learning_rate
    )

    # === Plot results ===
    plot_predictions(model, X_val, y_val, title="Stage-Wise Progressive DNN Regression")
    plot_loss_curves(histories)
