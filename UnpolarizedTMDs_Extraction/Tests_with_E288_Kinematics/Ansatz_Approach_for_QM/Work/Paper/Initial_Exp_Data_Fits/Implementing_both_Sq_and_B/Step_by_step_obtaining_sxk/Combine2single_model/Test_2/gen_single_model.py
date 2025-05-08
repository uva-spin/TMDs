import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


### Model Architecture ###

# Define the Progressive DNN Model
def build_progressive_model(input_shape=(1,), depth=4, width=256,
                           L1_reg=1e-12, initializer_range=0.1,
                           use_residual=False, activations=None,
                           output_activation='linear', name=None):
    # Default activations if none provided
    if activations is None:
        activations = ['relu'] * depth
    elif isinstance(activations, str):
        activations = [activations] * depth
    elif len(activations) < depth:
        # Pad with the last activation if list isn't long enough
        activations.extend([activations[-1]] * (depth - len(activations)))
    
    initializer = tf.keras.initializers.RandomUniform(minval=-initializer_range,
                                                     maxval=initializer_range)
    regularizer = tf.keras.regularizers.L1(L1_reg)
    inp = tf.keras.Input(shape=input_shape, name="input")
    x = tf.keras.layers.Dense(width, activation=activations[0],
                             kernel_initializer=initializer,
                             kernel_regularizer=regularizer)(inp)
    hidden_layers = [x]
    for i in range(1, depth):
        dense = tf.keras.layers.Dense(width, activation=activations[i],
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
    # Using different activations for each layer
    model, _ = build_progressive_model(
        input_shape=(1,),
        depth=3,
        width=100,
        use_residual=False,
        activations=['relu', 'tanh', 'relu'],  # Different activation for each layer
        output_activation='linear',
        name=name
    )
    return model

# Define Progressive DNNS
def DNNS(name):
    # Using different activations for each layer
    model, _ = build_progressive_model(
        input_shape=(3,),
        depth=3,
        width=100,
        use_residual=False,
        initializer_range=0.1,
        activations=['relu', 'tanh', 'relu'],  # Different activation for each layer
        output_activation='softplus',
        name=name
    )
    return model

def SB_model():
    qT = tf.keras.Input(shape=(1,), name='qT')
    QM = tf.keras.Input(shape=(1,), name='QM')
    x1 = tf.keras.Input(shape=(1,), name='x1')
    x2 = tf.keras.Input(shape=(1,), name='x2')
    pdfs_x1x2 = tf.keras.Input(shape=(1,), name='pdfs_x1x2')
    pdfs_x2x1 = tf.keras.Input(shape=(1,), name='pdfs_x2x1') 

    SModel = DNNS('SqT')
    BModel = DNNB('BQM')
    
    concatenatedx1x2 = tf.keras.layers.Concatenate()([qT, x1, x2])
    concatenatedx2x1 = tf.keras.layers.Concatenate()([qT, x2, x1])
    
    Sqx1x2 = SModel(concatenatedx1x2)
    Sqx2x1 = SModel(concatenatedx2x1)
    BQM = BModel(QM)

    pdfs_sqT_x1x2 = tf.keras.layers.Multiply()([pdfs_x1x2, Sqx1x2])
    pdfs_sqT_x2x1 = tf.keras.layers.Multiply()([pdfs_x2x1, Sqx2x1])
    
    # Calculate combined S contribution
    PDFs_S_combined = tf.keras.layers.Add()([pdfs_sqT_x1x2, pdfs_sqT_x2x1])
    
    # Multiply with pre-calculated PDFs
    SB_PDF = tf.keras.layers.Multiply()([PDFs_S_combined, BQM])
    
    return tf.keras.Model([qT, QM, x1, x2, pdfs_x1x2, pdfs_x2x1], SB_PDF)

########################################################################


models_folder = 'Models'
if not os.path.exists(models_folder):
    models_folder = '../../../Step_by_step_tuning_to_get_sqT/Test_68/Models'

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
        return custom_weighted_loss(y_true, y_pred, sample_weight)


model_files = [f for f in os.listdir(models_folder) if f.endswith('.h5')]
models_list = [tf.keras.models.load_model(
    os.path.join(models_folder, f), 
    custom_objects={
        'custom_weighted_loss': custom_weighted_loss,
        'CustomWeightedLoss': CustomWeightedLoss,
        'train_weighted_loss': custom_weighted_loss}) for f in model_files]


def create_averaged_model_simple(models_list):
    if not models_list:
        raise ValueError("No models provided")
    
    fresh_model = SB_model()
    model_weights = models_list[0].get_weights()
    
    fresh_model.set_weights(model_weights)
    
    return fresh_model

# Create the model with averaged weights
averaged_model = create_averaged_model_simple(models_list)


# Save model
avg_model_path = 'averaged_model.h5'
averaged_model.save(avg_model_path)
