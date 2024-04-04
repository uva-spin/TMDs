######################################################################


import tensorflow as tf
import pandas as pd
import numpy as np
from kerastuner.tuners import RandomSearch
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from scipy.integrate import simps

########### Generating Pseudodata #################



lhapdf_df = pd.read_csv('NNPDF4_nlo.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')



def Skq(k):
    return np.exp(-4*k**2/(4*k**2 + 4))

def Skqbar(k):
    return np.exp(-4*k**2/(4*k**2 + 1))


fu = np.array(lhapdf_df['fu'])
fubar = np.array(lhapdf_df['fubar'])

# fu = np.array(np.zeros_like(fu)+1)
# fubar = np.array(np.zeros_like(fubar)+1)


def f_map(x,f_array):
    mapping = dict(zip(x,f_array))
    return mapping


#x1vals = np.linspace(0.1, 0.3, 10)
x1vals = np.array(lhapdf_df['x'])
x2vals = np.array(lhapdf_df['x'])
pTvals = np.linspace(0.1,4,len(x1vals))
Kvals = np.linspace(0.1,2,len(x1vals))
pT_k_vals = pTvals - Kvals
kk_values_loss = np.array(np.linspace(0,0,len(x1vals)))


def fu_val(x):
    map = f_map(x1vals,fu)
    return map.get(x,None)

def fubar_val(x):
    map = f_map(x2vals,fubar)
    return map.get(x,None)

#print(fubar_val(x1vals[0]))

def fx1kx2k(x1,x2,pT,k):
    return fu_val(x1)*Skq(k)*fubar_val(x2)*Skqbar(pT-k)


def Apseudo(x1,x2,pT):
    tempx1, tempx2, temppT, tempA = [], [], [], []
    kk = np.linspace(0.0,2.0,len(x1vals))
    for i in range(len(x1)):
        tempx1.append(x1[i])
        tempx2.append(x2[i])
        temppT.append(pT[i])
        tempfx1kfx2k = simps(fx1kx2k(x1[i],x2[i],pT[i],kk), dx=(kk[1]-kk[0]))
        tempA.append(tempfx1kfx2k)
    return np.array(tempx1), np.array(tempx2), np.array(temppT), np.array(tempA)

x1Vals, x2Vals, pTVals, Avals = Apseudo(x1vals,x2vals,pTvals)

df = pd.DataFrame({'x1': x1Vals, 'x2': x2Vals, 'pT': pTVals, 'A': Avals})

# Define a function to build the model using Keras Tuner
def create_nn_model(hp):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(2,)))
    
    # Tune the number of hidden layers
    for i in range(hp.Int('num_layers', min_value=1, max_value=7, step=1)):
        # Tune the number of units in each hidden layer
        model.add(layers.Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),
                               activation='relu'))
    
    # Output layer
    model.add(layers.Dense(1, activation='relu'))
    
    model.compile(optimizer='adam', loss='mse')
    return model



def createModel_DY():
    x1 = tf.keras.Input(shape=(1), name='x1')
    x2 = tf.keras.Input(shape=(1), name='x2')
    qT = tf.keras.Input(shape=(1), name='qT')

    # Implementing f(x,k) --> f(x) when k-->0
    #nnu_pdf = modnnu(tf.keras.layers.Concatenate()([x1, x1*0 ]))
    #nnubar_pdf = modnnu(tf.keras.layers.Concatenate()([x2, x2*0 ]))

    modnnu = create_nn_model('nnu')
    modnnubar = create_nn_model('nnubar')

    pdf_k_val = 0    
    nnu_pdf_input = tf.keras.layers.Concatenate()([x1, qT*0 + pdf_k_val])
    nnubar_pdf_input = tf.keras.layers.Concatenate()([x2, qT*0 - pdf_k_val])

    modnnu_pdf_eval = modnnu(nnu_pdf_input)
    modnnubar_pdf_eval = modnnubar(nnubar_pdf_input)

    k_values = tf.linspace(0.1, 2.0, 100)
    dk = k_values[1] - k_values[0]

    tmd1_list, tmd2_list = [], []
    product_list = []  # List to store TMD values for each k value
    for k_val in k_values:      
        nnu_input = tf.keras.layers.Concatenate()([x1, qT*0 + k_val])
        nnubar_input = tf.keras.layers.Concatenate()([x2, qT - k_val])

        nnu_x1 = modnnu(nnu_input)
        nnubar_x2 = modnnubar(nnubar_input)

        nnu_x1_rev = modnnu(nnubar_input)
        nnubar_x2_rev = modnnubar(nnu_input)

        product_1 = tf.multiply(nnu_x1, nnubar_x2)
        product_2 = tf.multiply(nnu_x1_rev, nnubar_x2_rev)

        result = tf.add(product_1,product_2)
        product_list.append(result)

    # Summing over all k values using tf.reduce_sum
    tmd_product_sum = tf.reduce_sum(product_list, axis=0) * dk 

    return tf.keras.Model([x1, x2, qT], [tmd_product_sum, modnnu_pdf_eval, modnnubar_pdf_eval])

model = createModel_DY()


# Split the dataset into training and validation sets
def train_test_split(df, test_size=0.2):
    train_indices = np.random.rand(len(df)) < (1 - test_size)
    train_data = df[train_indices]
    test_data = df[~train_indices]
    return train_data, test_data

df_train, df_val = train_test_split(df, test_size=0.2)

x_train = df_train[['x1', 'x2', 'pT']]
y_train = df_train['A']
x_val = df_val[['x1', 'x2', 'pT']]
y_val = df_val['A']

# Instantiate the Keras Tuner RandomSearch tuner
tuner = RandomSearch(
    create_nn_model,
    objective='val_loss',
    max_trials=5,  # Adjust this value based on your computational resources
    directory='keras_tuner_dir',
    project_name='pseudodata_model'
)

# Search for the best hyperparameter configuration
tuner.search(x_train, y_train, epochs=200, validation_data=(x_val, y_val))

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Summary of the best model
best_model.summary()
