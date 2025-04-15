import tensorflow as tf

initial_lr = 0.001
epochs = 1000 
batch_size = 8

def DNNB(name):
    L1_reg = 10**(-12)
    return tf.keras.Sequential([
        tf.keras.Input(shape=(1,)), 
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ],name=name)


def DNNS(name):
    L1_reg = 10**(-12)
    inp = tf.keras.Input(shape=(1,))
    initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=None)
    x = tf.keras.layers.Dense(1, activation='relu', kernel_initializer = initializer)(inp)
    for i in range(3):
        x = tf.keras.layers.Dense(100, activation='relu', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg), activity_regularizer=tf.keras.regularizers.L1(L1_reg))(x)
    out = tf.keras.layers.Dense(1, activation='linear', kernel_initializer = initializer)(x)
    return tf.keras.Model(inp, out, name=name)


# def DNNS(name):
#     inp = tf.keras.Input(shape=(1,))
#     initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=None)
#     x1 = tf.keras.layers.Dense(100, activation='relu', kernel_initializer = initializer)(inp)
#     x2 = tf.keras.layers.Dense(100, activation='relu', kernel_initializer = initializer)(x1)
#     x3 = tf.keras.layers.Dense(100, activation='relu', kernel_initializer = initializer)(x2)
#     x3 = tf.keras.layers.Dense(100, activation='relu', kernel_initializer = initializer)(x2)
#     x3 = tf.keras.layers.Dense(100, activation='relu', kernel_initializer = initializer)(x2)
#     x3 = tf.keras.layers.Dense(100, activation='relu', kernel_initializer = initializer)(x2)
#     out = tf.keras.layers.Dense(1, activation='softplus', kernel_initializer = initializer)(x3)
#     return tf.keras.Model(inp, out, name=name)



def SB_model():
    qT= tf.keras.Input(shape=(1,), name='qT')
    QM = tf.keras.Input(shape=(1,), name='QM')

    SModel = DNNS('SqT')
    BModel = DNNB('BQM')

    Sq = SModel(qT)
    BQM = BModel(QM)

    SB = tf.keras.layers.Multiply()([Sq, BQM])
    # SB = Sq
    return tf.keras.Model([qT,QM],SB)


modify_LR = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',factor=0.9,patience=100,mode='auto')

