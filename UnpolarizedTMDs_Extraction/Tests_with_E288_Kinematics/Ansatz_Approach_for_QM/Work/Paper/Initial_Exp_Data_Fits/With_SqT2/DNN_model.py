import tensorflow as tf


def DNNB():
    return tf.keras.Sequential([
        tf.keras.Input(shape=(1,)), 
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

initial_lr = 0.016 #0.001  

epochs = 300 # 1000 #500  

batch_size = 8