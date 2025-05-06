import tensorflow as tf


################ Defining the DNN model ####################
Hidden_Layers = 3  # Reduced for simplicity
Nodes_per_HL = 100  # Reduced for simplicity
L1_reg = 10**(-12)


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