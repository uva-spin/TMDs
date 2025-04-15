import tensorflow as tf


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


def SB_model():
    qT= tf.keras.Input(shape=(1,), name='qT')
    QM = tf.keras.Input(shape=(1,), name='QM')

    SModel = ProgressiveDNN('SqT')
    BModel = ProgressiveDNN('BQM')

    Sq = SModel(qT)
    BQM = BModel(QM)

    SB = tf.keras.layers.Multiply()([Sq, BQM])
    return tf.keras.Model([qT,QM],SB)


# # Define the DNN Model
# def DNNB(name):
#     return tf.keras.Sequential([
#         tf.keras.Input(shape=(1,)), 
#         tf.keras.layers.Dense(100, activation='relu'),
#         tf.keras.layers.Dense(100, activation='relu'),
#         tf.keras.layers.Dense(1, activation='linear')
#     ],name=name)


# # def DNNS(name):
# #     return tf.keras.Sequential([
# #         tf.keras.Input(shape=(1,)), 
# #         tf.keras.layers.Dense(100, activation='relu6'),
# #         tf.keras.layers.Dense(100, activation='relu6'),
# #         tf.keras.layers.Dense(1, activation='softplus')  # Ensures non-negative output
# #     ], name=name)


# # def DNNS(name):
# #     inp = tf.keras.Input(shape=(1,))
# #     initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=None)
# #     x1 = tf.keras.layers.Dense(100, activation='relu', kernel_initializer = initializer)(inp)
# #     x2 = tf.keras.layers.Dense(100, activation='relu', kernel_initializer = initializer)(x1)
# #     out = tf.keras.layers.Dense(1, activation='softplus', kernel_initializer = initializer)(x2)
# #     return tf.keras.Model(inp, out, name=name)


# # def DNNS(name):
# #     inp = tf.keras.Input(shape=(1,))
# #     initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=None)
# #     x1 = tf.keras.layers.Dense(100, activation='relu6', kernel_initializer = initializer)(inp)
# #     x2 = tf.keras.layers.Dense(100, activation='tanh', kernel_initializer = initializer)(x1)
# #     out = tf.keras.layers.Dense(1, activation='softplus', kernel_initializer = initializer)(x2)
# #     return tf.keras.Model(inp, out, name=name)

# def DNNS(name):
#     inp = tf.keras.Input(shape=(1,))
#     initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=None)
#     x1 = tf.keras.layers.Dense(100, activation='relu', kernel_initializer = initializer)(inp)
#     x2 = tf.keras.layers.Dense(100, activation='tanh', kernel_initializer = initializer)(x1)
#     x3 = tf.keras.layers.Dense(100, activation='relu', kernel_initializer = initializer)(x2)
#     out = tf.keras.layers.Dense(1, activation='softplus', kernel_initializer = initializer)(x3)
#     return tf.keras.Model(inp, out, name=name)


# def SB_model():
#     qT= tf.keras.Input(shape=(1,), name='qT')
#     QM = tf.keras.Input(shape=(1,), name='QM')

#     SModel = DNNS('SqT')
#     BModel = DNNB('BQM')

#     Sq = SModel(qT)
#     BQM = BModel(QM)

#     SB = tf.keras.layers.Multiply()([Sq, BQM])
#     return tf.keras.Model([qT,QM],SB)


initial_lr = 0.002
# initial_lr = 0.016 #0.001  

epochs = 1000 # 1000 #500  

batch_size = 8

modify_LR = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',factor=0.9,patience=100,mode='auto')
