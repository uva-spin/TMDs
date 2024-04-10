# In this example I have included the two DNNs multiplying to give the cross-section 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from kerastuner.tuners import BayesianOptimization
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.integrate import simps

# Step 1: Generating Synthetic Data
# This generates data based on a polynomial function with some noise
np.random.seed(42)  # Seed for reproducibility
# x = np.linspace(-1, 1, 400)
# y = 0.5 * x**3 - 0.2 * x**2 + 0.1 * x - 0.2 + np.random.normal(scale=0.05, size=x.shape)

x1 = np.linspace(-1, 1, 100)
x2 = np.linspace(-1, 1, 100)
pT = np.linspace(1, 3, 100)
y = (0.5 * x1**3 - 0.2 * x1**2 + 0.1 * x1 - 0.2 + 
     0.7 * x2**3 - 0.4 * x2**2 + 0.05 * x2 - 0.2 + 0.2 * pT**2 + np.random.normal(scale=0.05, size=x1.shape))

def func(x1,x2,pT,kk):
    temp = (0.5 * x1**3 - 0.2 * x1**2 + 0.1 * x1 - 0.2 + 
     0.7 * x2**3 - 0.4 * x2**2 + 0.05 * x2 - 0.2 + 0.2 * pT**2 + 2*kk + np.random.normal(scale=0.05, size=x1.shape))
    return temp


def Apseudo(x1,x2,pT):
    tempx1, tempx2, temppT, tempA = [], [], [], []
    kk = np.linspace(0.0,2.0,len(x1))
    for i in range(len(x1)):
        tempx1.append(x1[i])
        tempx2.append(x2[i])
        temppT.append(pT[i])
        tempfx1kfx2k = simps(func(x1[i],x2[i],pT[i],kk), dx=(kk[1]-kk[0]))
        tempA.append(tempfx1kfx2k)
    return np.array(tempx1), np.array(tempx2), np.array(temppT), np.array(tempA)

x1Vals, x2Vals, pTVals, Avals = Apseudo(x1,x2,pT)

df = pd.DataFrame({'x1': x1Vals, 'x2': x2Vals, 'pT': pTVals, 'A': Avals})

# Step 2: Building the Model and Optimizing it with Keras Tuner
def build_model(hp):
    model = keras.Sequential()
    inp = tf.keras.Input(shape=(2,))
    model.add(inp)  # Input layer
    
    # Tuning the number of layers and their configurations
    # Keras Tuner will choose an optimal number of layers, units in each layer, and their activation functions
    for i in range(hp.Int('num_layers', 1, 5)):
        model.add(layers.Dense(
            units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),
            activation=hp.Choice('activation_' + str(i), ['relu', 'tanh', 'sigmoid'])
        ))
        
    model.add(layers.Dense(1))  # Output layer
    #model = tf.keras.Model(inp, model, name=name)
    
    # Compilation of the model with a tunable learning rate
    # model.compile(
    #     optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
    #     loss='mean_squared_error',
    #     metrics=['mean_squared_error']
    # )
    return model


def createModel_DY(hp):
    x1 = tf.keras.Input(shape=(1), name='x1')
    x2 = tf.keras.Input(shape=(1), name='x2')
    qT = tf.keras.Input(shape=(1), name='qT')

    modnnu = build_model(hp)
    modnnubar = build_model(hp)

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

    model = tf.keras.Model([x1, x2, qT], tmd_product_sum)

    # Compilation of the model with a tunable learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='mean_squared_error',
        metrics=['mean_squared_error']
    )

    return model


# Initiating the tuner
# Here, BayesianOptimization is used to find the best hyperparameter values by building and evaluating different models
tuner = BayesianOptimization(
    createModel_DY,
    objective='val_mean_squared_error',
    max_trials=10,  # The total number of trials (model configurations) to test
    executions_per_trial=1,  # The number of models that should be built and fit for each trial
    directory='my_dir',
    project_name='keras_tuner_demo'
)


def split_data(X,y,split=0.1):
  temp =np.random.choice(list(range(len(y))), size=int(len(y)*split), replace = False)

  test_X = pd.DataFrame.from_dict({k: v[temp] for k,v in X.items()})
  #train_X = {k: np.delete(v,temp) for k,v in X.items()}
  train_X = pd.DataFrame.from_dict({k: v.drop(temp) for k,v in X.items()})

  test_y = y[temp]
  #train_y =np.delete(y, temp)
  train_y = y.drop(temp)

  return train_X, test_X, train_y, test_y


#dd=data.drop(['A','x'], axis = 1)
y = df['A']
X = df.drop(['A'], axis = 1)

# Splitting data into training and validation sets
#x = tf.keras.layers.Concatenate()([x1,x2])
# x_train, x_val, y_train, y_val = train_test_split([x1,x2], y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = split_data(X, y)

# print(x_train)
#print(x_train['x1'].to_numpy())
#print(x_train.to_numpy()[:,0])

x1_tr = x_train.to_numpy()[:,0]
x2_tr = x_train.to_numpy()[:,1]
pT_tr = x_train.to_numpy()[:,2]

x1_val = x_val.to_numpy()[:,0]
x2_val = x_val.to_numpy()[:,1]
pT_val = x_val.to_numpy()[:,2]

# Start the search for the best hyperparameter configuration
# The tuner explores different configurations, evaluating them on the validation set
# tuner.search(x_train, y_train, epochs=20, validation_data=(x_val, y_val))
tuner.search([x1_tr,x2_tr,pT_tr], y_train, epochs=20, validation_data=([x1_val,x2_val,pT_val], y_val))

# After the search, retrieve the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
#print(best_hps)

# Step 3: Training the Best Model Found by Keras Tuner
# The model is built using the best hyperparameters and then trained on the full dataset
model = tuner.hypermodel.build(best_hps)
# history = model.fit(x_train, y_train, epochs=50, validation_data=(x_val, y_val))
history = model.fit([x1_tr,x2_tr,pT_tr], y_train, epochs=50, validation_data=([x1_val,x2_val,pT_val], y_val))

# Optional: plotting the loss curve
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

