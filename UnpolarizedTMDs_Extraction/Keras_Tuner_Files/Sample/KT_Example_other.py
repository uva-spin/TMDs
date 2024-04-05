import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from kerastuner.tuners import BayesianOptimization
from sklearn.model_selection import train_test_split


np.random.seed(42)
x = np.linspace(-1,1,400)
y = 0.5*x**3 - 0.2*x**2 + 0.1*x - 0.2 + np.random.normal(scale=0.05, size=x.shape)


def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(1,)))

    for i in range(hp.Int('num_layers',1,5)):
        model.add(layers.Dense(units=hp.Int('units_'+str(i), min_value=32, max_value=512, step=32),activation=hp.Choice('activation_'+str(i),['relu','tanh','sigmoid'])))
        model.add(layers.Dense(1))
        model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate',[1e-2,1e-3,1e-4])), loss='mean_squared_error', metrics=['mean_squared_error'])
    
    return model

tuner = BayesianOptimization(build_model, objective='val_mean_squared_error', max_trials=10, executions_per_trial=1, directory='my_dir', project_name='keras_tuner_demo')

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

tuner.search(x_train, y_train, epochs=20, validation_data(x_val,y_val))

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

model = tuner.hypermodel.build(best_hps)
history = model.fit(x_train,y_train,epochs=50, validation_data=(x_val,y_val))

