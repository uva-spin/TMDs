import sys
import ROOT
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from tensorflow_addons.activations import tanhshrink
sys.path.append('../../../')
from Formulation.BHDVCS_tf_modified import TotalFLayer
tf.keras.utils.get_custom_objects().update({'tanhshrink': tanhshrink})

class MyHyperModel(kt.HyperModel):

    def build(self, hp):
        initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=0)  
        inputs = tf.keras.Input(shape=(5)) # k, QQ, x_b, t, phi
        k, QQ, xB, t, phi = tf.split(inputs, num_or_size_splits=5, axis=1)
        kinematics = tf.keras.layers.concatenate([QQ, xB, t], axis=1)
        # Tune the number of units (neurons (nodes)) and the number of dense hidden layer.Activation is going to be a list of choices.
        for i in range(hp.Int('layers', 1, 6)):
            kinematics = tf.keras.layers.Dense(units=hp.Int('units_' + str(i), 20, 200, step=10),
                                               activation=hp.Choice('act_' + str(i), ['relu', 'tanh', 'sigmoid', 'tanhshrink']), 
                                               kernel_initializer=initializer)(kinematics)      
        outputs = tf.keras.layers.Dense(4, activation="linear", kernel_initializer=initializer)(kinematics)
        #### k, QQ, xB, t, phi, ReH, ReE, ReHt, dvcs ####
        total_FInputs = tf.keras.layers.concatenate([inputs, outputs], axis=1)
        TotalF = TotalFLayer()(total_FInputs) # get rid of f1 and f2
        tfModel = tf.keras.Model(inputs=inputs, outputs = TotalF)
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])   
        tfModel.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss = tf.keras.losses.MeanSquaredError()
        )
        return tfModel
     
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Int('batch_size', 1, 10, step=1),
            **kwargs,
        )
    def evaluate(self, model, x, y):
        return model.evaluate(x,y)
    def predict(self, model, x):
        return model.predict(x)   
    
def get_data():
    df = ROOT.RDF.FromCSV('/media/lily/Data/GPDs/DVCS/Pseudodata/JLabKinematics/withRealErrors/pseudo_'+GPD_MODEL+'_BKM10_Jlab_all_t2.csv')
    npset = df.Filter("set<=5").AsNumpy() 
    return npset

GPD_MODEL = 'basic'

# Instantiate the tuner to perform the hypertuning. The Keras Tuner has four tuners available - RandomSearch, Hyperband, BayesianOptimization, and Sklearn.
tuner = kt.RandomSearch(
    MyHyperModel(),
    objective=kt.Objective("val_loss", "min"),
    max_trials=10,
    executions_per_trial = 3,
    overwrite=True,
    directory="tuner_tmp",
    project_name="tune_lmi",
)

tuner.search_space_summary()

pseudo = get_data()   
kin = np.dstack((pseudo['k'], pseudo['QQ'],pseudo['xB'], pseudo['t'], pseudo['phi']))
kin = kin.reshape(kin.shape[1:])
# ---- model fit ---- 
kin_train, kin_test, F_train, F_test = train_test_split(kin, pseudo['F'], test_size=0.1, random_state=42)

# create callbacks
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./tuner_tmp/tb', histogram_freq=1, embeddings_freq=1, write_graph=True, update_freq='batch')
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')

tuner.search(kin_train, F_train, epochs=200, validation_data=(kin_test, F_test), callbacks=[tensorboard])

# Print best model results
tuner.results_summary()
best_models = tuner.get_best_models(num_models=10)
print(best_models[0].summary())
