import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from kerastuner.tuners import BayesianOptimization
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.integrate import simps
import os
#import lhapdf
np.random.seed(42)  # Seed for reproducibility

k_upper = 2
L1_reg = 10**(-12)
EPOCHS = 500
BATCH = 64

########### Import pseudodata file 
df = pd.read_csv('E288_pseudo_data.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')


def build_model(hp):
    model = keras.Sequential()
    inp = tf.keras.Input(shape=(3,))
    model.add(inp)
    initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=42)  
    for i in range(hp.Int('num_layers', 1, 10)):
        model.add(layers.Dense(
            units=hp.Int('units_' + str(i), min_value=4, max_value=240, step=4),
            # activation=hp.Choice('activation_' + str(i), ['tanhshrink','relu'])
            activation=hp.Choice('activation_' + str(i), ['relu', 'relu6', 'tanh','selu', 'sigmoid'])
        , kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg)))
    model.add(layers.Dense(1))  
    return model



####### Here we define a function that can sample cross-section within errA ###
def GenerateReplicaData(df):
    pseudodata_df = {'x1': [],
                     'x2': [],
                     'pT': [],
                     'QM': [],
                     'A': [],
                     'errA':[]}
    #pseudodata_df = pd.DataFrame(pseudodata_df)
    pseudodata_df['x1'] = df['x1']
    pseudodata_df['x2'] = df['x2']
    pseudodata_df['pT'] = df['pT']
    pseudodata_df['QM'] = df['QM']
    pseudodata_df['A'] = df['A']
    tempA = df['A']
    tempAerr = np.abs(np.array(df['errA'])) 
    pseudodata_df['errA'] = np.random.normal(loc=tempA, scale=tempAerr)
    pseudodata_df['fsbar_xA']=df['fsbar_xA']
    pseudodata_df['fubar_xA']=df['fubar_xA']
    pseudodata_df['fdbar_xA']=df['fdbar_xA']
    pseudodata_df['fd_xA']=df['fd_xA']
    pseudodata_df['fu_xA']=df['fu_xA']
    pseudodata_df['fs_xA']=df['fs_xA']
    pseudodata_df['fsbar_xB']=df['fsbar_xB']
    pseudodata_df['fubar_xB']=df['fubar_xB']
    pseudodata_df['fdbar_xB']=df['fdbar_xB']
    pseudodata_df['fd_xB']=df['fd_xB']
    pseudodata_df['fu_xB']=df['fu_xB']
    pseudodata_df['fs_xB']=df['fs_xB']
    return pd.DataFrame(pseudodata_df)



def createModel_DY(hp):
    x1 = tf.keras.Input(shape=(1), name='x1')
    x2 = tf.keras.Input(shape=(1), name='x2')
    qT = tf.keras.Input(shape=(1), name='qT')
    qM = tf.keras.Input(shape=(1), name='QM')


    modnnu = build_model(hp)
    modnnubar = build_model(hp)

    pdf_k_val = 0    

    k_values = tf.linspace(0.0, k_upper, 100)
    dk = k_values[1] - k_values[0]

    f_xA, fbar_xB, f_xB, fbar_xA = [], [], [], []
    product_list = []  # List to store TMD values for each k value
    for k_val in k_values:     
        nnu_input = tf.keras.layers.Concatenate()([x1, qT*0 + k_val, qM])
        nnubar_input = tf.keras.layers.Concatenate()([x2, qT - k_val, qM])

        nnu_x1 = modnnu(nnu_input)
        nnubar_x2 = modnnubar(nnubar_input)
        nnu_x1_rev = modnnu(nnubar_input)
        nnubar_x2_rev = modnnubar(nnu_input)

        product_1 = tf.multiply(nnu_x1, nnubar_x2)
        product_2 = tf.multiply(nnu_x1_rev, nnubar_x2_rev)

        result = tf.add(product_1,product_2)
        product_list.append(result)
    
        xA_k__input = tf.keras.layers.Concatenate()([x1, qT*0 + k_val, qM])
        xB_k__input = tf.keras.layers.Concatenate()([x2, qT*0 + k_val, qM])
        f_xA_k = modnnu(xA_k__input)
        fbar_xB_k = modnnubar(xB_k__input)
        f_xB_k = modnnu(xB_k__input)
        fbar_xA_k = modnnubar(xA_k__input)
        f_xA.append(f_xA_k)
        fbar_xB.append(fbar_xB_k)
        f_xB.append(f_xB_k)
        fbar_xA.append(fbar_xA_k)


    # Summing over all k values using tf.reduce_sum
    tmd_product_sum = tf.reduce_sum(product_list, axis=0) * dk 
    #### k_perp Integrals ###
    f_xA_sum = tf.reduce_sum(f_xA, axis=0) * dk 
    fbar_xB_sum = tf.reduce_sum(fbar_xB, axis=0) * dk
    f_xB_sum = tf.reduce_sum(f_xB, axis=0) * dk 
    fbar_xA_sum = tf.reduce_sum(fbar_xA, axis=0) * dk 

    model = tf.keras.Model([x1, x2, qT, qM], [tmd_product_sum, f_xA_sum, fbar_xB_sum, f_xB_sum, fbar_xA_sum]) 

    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', [1e-3, 1e-4, 1e-5])),
        loss='mean_squared_error',
        metrics=['mean_squared_error']
    )
    return model


tuner = BayesianOptimization(
    createModel_DY,
    objective='mean_squared_error',
    # objective='val_mean_squared_error',
    max_trials=50,  # The total number of trials (model configurations) to test
    executions_per_trial=1,  # The number of models that should be built and fit for each trial
    directory='my_dir',
    project_name='keras_tuner_demo'
)


def split_data(X,y,yerr, fq, fqbar, fq_rev, fqbar_rev, split=0.1):
  temp =np.random.choice(list(range(len(y))), size=int(len(y)*split), replace = False)

  test_X = pd.DataFrame.from_dict({k: v[temp] for k,v in X.items()})
  train_X = pd.DataFrame.from_dict({k: v.drop(temp) for k,v in X.items()})

  test_y = y[temp]
  train_y = y.drop(temp)

  test_yerr = yerr[temp]
  train_yerr = yerr.drop(temp)

  test_fq = fq[temp]
  train_fq = fq.drop(temp)

  test_fqbar = fqbar[temp]
  train_fqbar = fqbar.drop(temp)

  test_fq_rev = fq_rev[temp]
  train_fq_rev = fq_rev.drop(temp)

  test_fqbar_rev = fqbar_rev[temp]
  train_fqbar_rev = fqbar_rev.drop(temp)

  return train_X, test_X, train_y, test_y, train_yerr, test_yerr, train_fq, test_fq, train_fqbar, test_fqbar, train_fq_rev, test_fq_rev, train_fqbar_rev, test_fqbar_rev


tempdf = df
trainKin, testKin, trainA, testA, trainAerr, testAerr, trainfq, testfq, trainfqbar, testfqbar, trainfq_rev, testfq_rev, trainfqbar_rev, testfqbar_rev = split_data(tempdf[['x1', 'x2', 'pT', 'QM']],
                                                                    tempdf['A'], tempdf['errA'], tempdf['fu_xA'], tempdf['fubar_xB'], tempdf['fu_xB'], tempdf['fubar_xA'], split=0.1)

tuner.search([trainKin['x1'],trainKin['x2'],trainKin['pT'],trainKin['QM']], [trainA, trainfq, trainfqbar, trainfq_rev, trainfqbar_rev],  validation_data=([testKin['x1'],testKin['x2'],testKin['pT'],testKin['QM']], [testA, testfq, testfqbar, testfq_rev, testfqbar_rev]), epochs=EPOCHS)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

hyp_par_dict = best_hps.values
hyp_df = pd.DataFrame(hyp_par_dict.items(), columns=['Hyperparameter','Value'])
hyp_df.to_csv('best_hyperparameters.csv', index=False)


model = tuner.hypermodel.build(best_hps)
history = model.fit([trainKin['x1'],trainKin['x2'],trainKin['pT'],trainKin['QM']], [trainA, trainfq, trainfqbar, trainfq_rev, trainfqbar_rev],  validation_data=([testKin['x1'],testKin['x2'],testKin['pT'],testKin['QM']], [testA, testfq, testfqbar, testfq_rev, testfqbar_rev]), epochs=EPOCHS, batch_size=BATCH, verbose=2)


# Optional: plotting the loss curve
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('Loss_vs_Epochs_bestmodel.pdf')
plt.legend()
plt.savefig('Loss_Plot_from_best_hp_model.pdf')