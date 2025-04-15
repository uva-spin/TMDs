import tensorflow as tf



# Define the DNN Model
def DNNB(name):
    return tf.keras.Sequential([
        tf.keras.Input(shape=(1,)), 
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ],name=name)


# def DNNS(name):
#     return tf.keras.Sequential([
#         tf.keras.Input(shape=(1,)), 
#         tf.keras.layers.Dense(100, activation='relu6'),
#         tf.keras.layers.Dense(100, activation='relu6'),
#         tf.keras.layers.Dense(1, activation='softplus')  # Ensures non-negative output
#     ], name=name)


# def DNNS(name):
#     inp = tf.keras.Input(shape=(1,))
#     initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=None)
#     x1 = tf.keras.layers.Dense(100, activation='relu', kernel_initializer = initializer)(inp)
#     x2 = tf.keras.layers.Dense(100, activation='relu', kernel_initializer = initializer)(x1)
#     out = tf.keras.layers.Dense(1, activation='softplus', kernel_initializer = initializer)(x2)
#     return tf.keras.Model(inp, out, name=name)


# def DNNS(name):
#     inp = tf.keras.Input(shape=(1,))
#     initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=None)
#     x1 = tf.keras.layers.Dense(100, activation='relu6', kernel_initializer = initializer)(inp)
#     x2 = tf.keras.layers.Dense(100, activation='tanh', kernel_initializer = initializer)(x1)
#     out = tf.keras.layers.Dense(1, activation='softplus', kernel_initializer = initializer)(x2)
#     return tf.keras.Model(inp, out, name=name)

def DNNS(name):
    inp = tf.keras.Input(shape=(1,))
    initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=None)
    x1 = tf.keras.layers.Dense(100, activation='relu', kernel_initializer = initializer)(inp)
    x2 = tf.keras.layers.Dense(100, activation='tanh', kernel_initializer = initializer)(x1)
    x3 = tf.keras.layers.Dense(100, activation='relu', kernel_initializer = initializer)(x2)
    out = tf.keras.layers.Dense(1, activation='softplus', kernel_initializer = initializer)(x3)
    return tf.keras.Model(inp, out, name=name)


def SB_model():
    qT= tf.keras.Input(shape=(1,), name='qT')
    QM = tf.keras.Input(shape=(1,), name='QM')

    SModel = DNNS('SqT')
    BModel = DNNB('BQM')

    Sq = SModel(qT)
    BQM = BModel(QM)

    SB = tf.keras.layers.Multiply()([Sq, BQM])
    return tf.keras.Model([qT,QM],SB)


initial_lr = 0.002
# initial_lr = 0.016 #0.001  

epochs = 1000 # 1000 #500  

batch_size = 8