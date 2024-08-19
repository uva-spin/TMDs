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
from tensorflow_addons.activations import tanhshrink
tf.keras.utils.get_custom_objects().update({'tanhshrink': tanhshrink})


np.random.seed(42)  # Seed for reproducibility


########### Import pseudodata file 
df = pd.read_csv('E288_pseudo_data.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')

L1_reg = 10**(-12)


def build_model(hp):
    model = keras.Sequential()
    inp = tf.keras.Input(shape=(3,))
    model.add(inp)
    initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=42)  
    for i in range(hp.Int('num_layers', 1, 5)):
        model.add(layers.Dense(
            units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),
            # activation=hp.Choice('activation_' + str(i), ['tanhshrink','relu'])
            activation=hp.Choice('activation_' + str(i), ['relu', 'relu6', 'tanh', 'tanhshrink','selu', 'sigmoid', 'softmax'])
        , kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg)))
    model.add(layers.Dense(1))  
    return model



# def build_model(hp):
#     model = keras.Sequential()
#     inp = tf.keras.Input(shape=(3,))
#     model.add(inp)
#     initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=42) 
#     num_layers = hp.Int('num_layers', min_value=1, max_value=7, default=3) 
#     for i in range(num_layers):
#         model.add(layers.Dense(
#             units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),
#             # activation=hp.Choice('activation_' + str(i), ['tanhshrink','relu'])
#             activation=hp.Choice('activation_' + str(i), ['relu', 'relu6', 'tanh', 'tanhshrink','selu', 'sigmoid', 'softmax'])
#         , kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg)))
#     model.add(layers.Dense(1))  
#     return model


# def build_model(hp):
#     model = keras.Sequential()
#     inp = tf.keras.Input(shape=(3,))
#     x, k, QM = tf.split(inp, num_or_size_splits=3, axis=1)
#     kinematics = tf.keras.layers.concatenate([x, k, QM], axis=1)
#     initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=42)  
#     for i in range(hp.Int('num_layers', 1, 7)):
#         kinematics = tf.keras.layers.Dense(
#             units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),
#             activation=hp.Choice('activation_' + str(i), ['relu', 'relu6', 'tanh', 'tanhshrink','selu', 'sigmoid', 'softmax'])
#         , kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(kinematics)
#     output = tf.keras.layers.Dense(1, activation="relu", kernel_initializer=initializer)(kinematics)
#     return output

# def build_model(hp):
#     model = keras.Sequential()
#     inp = tf.keras.Input(shape=(3,))
#     model.add(inp)
#     initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=42)  
#     for i in range(hp.Int('num_layers', 1, 7)):
#         model = tf.keras.layers.Dense(
#             units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),
#             activation=hp.Choice('activation_' + str(i), ['relu', 'relu6', 'tanh', 'tanhshrink','selu', 'sigmoid', 'softmax'])
#         , kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))
#     model= tf.keras.layers.Dense(1)
#     return model

# def build_model(hp):
#     model = keras.Sequential()
#     inp = tf.keras.Input(shape=(3,))
#     model.add(inp)
#     initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=42)  
#     for i in range(hp.Int('num_layers', 1, 7)):
#         model.add(layers.Dense(
#             units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),
#             # activation=hp.Choice('activation_' + str(i), ['tanhshrink','relu'])
#             activation=hp.Choice('activation_' + str(i), ['relu', 'relu6', 'tanh', 'tanhshrink','selu', 'sigmoid', 'softmax'])
#         , kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg)))
#     model.add(layers.Dense(1))  
#     return model

# def build_model(hp):
#     model = keras.Sequential()
#     kinematics = tf.keras.Input(shape=(3,))
#     initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=42)  
#     for i in range(hp.Int('num_layers', 1, 7)):
#         kinematics = tf.keras.layers.Dense(
#             units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),
#             activation=hp.Choice('activation_' + str(i), ['relu', 'relu6', 'tanh', 'tanhshrink','selu', 'sigmoid', 'softmax'])
#         , kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(kinematics)
#     output = tf.keras.layers.Dense(1, activation="relu", kernel_initializer=initializer)(kinematics)
#     return output



def createModel_DY(hp):
    x1 = tf.keras.Input(shape=(1), name='x1')
    x2 = tf.keras.Input(shape=(1), name='x2')
    qT = tf.keras.Input(shape=(1), name='qT')
    qM = tf.keras.Input(shape=(1), name='QM')

    modnnu = build_model(hp)
    modnnubar = build_model(hp)

    pdf_k_val = 0    
    nnu_pdf_input = tf.keras.layers.Concatenate()([x1, qT*0 + pdf_k_val, qM])
    nnubar_pdf_input = tf.keras.layers.Concatenate()([x2, qT*0 - pdf_k_val, qM])

    modnnu_pdf_eval = modnnu(nnu_pdf_input)
    modnnubar_pdf_eval = modnnubar(nnubar_pdf_input)


    k_values = tf.linspace(0.1, 2.0, 100)
    dk = k_values[1] - k_values[0]

    tmd1_list, tmd2_list = [], []
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

    # Summing over all k values using tf.reduce_sum
    tmd_product_sum = tf.reduce_sum(product_list, axis=0) * dk 

    #model = tf.keras.Model([x1, x2, qT], [tmd_product_sum, modnnu_pdf_eval, modnnubar_pdf_eval])
    model = tf.keras.Model([x1, x2, qT, qM], tmd_product_sum)

    # Compilation of the model with a tunable learning rate
    # model.compile(
    #     optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
    #     loss='mean_squared_error',
    #     metrics=['mean_squared_error']
    # )
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4, 1e-5])),
        loss='mean_squared_error',
        metrics=['mean_squared_error']
    )
    return model


# Here, BayesianOptimization is used to find the best hyperparameter values by building and evaluating different models
tuner = BayesianOptimization(
    createModel_DY,
    objective='mean_squared_error',
    # objective='val_mean_squared_error',
    max_trials=50,  # The total number of trials (model configurations) to test
    executions_per_trial=1,  # The number of models that should be built and fit for each trial
    directory='my_dir',
    project_name='keras_tuner_demo'
)


# def split_data(X,y,split=0.1):
#   temp =np.random.choice(list(range(len(y))), size=int(len(y)*split), replace = False)

#   test_X = pd.DataFrame.from_dict({k: v[temp] for k,v in X.items()})
#   #train_X = {k: np.delete(v,temp) for k,v in X.items()}
#   train_X = pd.DataFrame.from_dict({k: v.drop(temp) for k,v in X.items()})

#   test_y = pd.DataFrame.from_dict({k: v[temp] for k,v in y.items()})
#   #test_y = y[temp]
#   #train_y =np.delete(y, temp)
#   train_y = y.drop(temp)

#   return train_X, test_X, train_y, test_y


def split_data(X,y,yerr, fq, fqbar, split=0.1):
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

  return train_X, test_X, train_y, test_y, train_yerr, test_yerr, train_fq, test_fq, train_fqbar, test_fqbar


# #dd=data.drop(['A','x'], axis = 1)
# y = df[['A','fu','fubar']]
# #X = df.drop(['A'], axis = 1)
# X = df[['x1','x2','pT', 'QM']]

# # Splitting data into training and validation sets
# x_train, x_val, y_train, y_val = split_data(X, y)


trainKin, testKin, trainA, testA, trainAerr, testAerr, trainfq, testfq, trainfqbar, testfqbar = split_data(df[['x1', 'x2', 'pT', 'QM']],
                                                                       df['A'], df['errA'], df['fu_xA'], df['fubar_xB'], split=0.1)


x1_tr = trainKin.to_numpy()[:,0]
x2_tr = trainKin.to_numpy()[:,1]
pT_tr = trainKin.to_numpy()[:,2]
qM_tr = trainKin.to_numpy()[:,3]

x1_val = testKin.to_numpy()[:,0]
x2_val = testKin.to_numpy()[:,1]
pT_val = testKin.to_numpy()[:,2]
qM_val = testKin.to_numpy()[:,3]

A_tr = trainA.to_numpy()
pdfu_tr = trainfq.to_numpy()
pdfubar_tr = trainfqbar.to_numpy()

A_val = testA.to_numpy()
pdfu_val = testfq.to_numpy()
pdffubar_val = testfqbar.to_numpy()

#print(x1_val)

# Start the search for the best hyperparameter configuration
# The tuner explores different configurations, evaluating them on the validation set
# tuner.search(x_train, y_train, epochs=20, validation_data=(x_val, y_val))
# tuner.search([x1_tr,x2_tr,pT_tr], [A_tr, pdfu_tr, pdfubar_tr], epochs=20, validation_data=([x1_val,x2_val,pT_val], [A_val,pdfu_val,pdffubar_val]))
# tuner.search([x1_tr,x2_tr,pT_tr,qM_tr], [A_tr, pdfu_tr, pdfubar_tr], epochs=500, validation_data=([x1_val,x2_val,pT_val,qM_val], A_val))
tuner.search([x1_tr,x2_tr,pT_tr,qM_tr], A_tr, epochs=500, validation_data=([x1_val,x2_val,pT_val,qM_val], A_val))


# After the search, retrieve the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

hyp_par_dict = best_hps.values
hyp_df = pd.DataFrame(hyp_par_dict.items(), columns=['Hyperparameter','Value'])
hyp_df.to_csv('best_hyperparameters.csv', index=False)
#print(best_hps)

# Step 3: Training the Best Model Found by Keras Tuner
# The model is built using the best hyperparameters and then trained on the full dataset
model = tuner.hypermodel.build(best_hps)
# history = model.fit(x_train, y_train, epochs=50, validation_data=(x_val, y_val))
#history = model.fit([x1_tr,x2_tr,pT_tr], [A_tr, pdfu_tr, pdfubar_tr], epochs=50, validation_data=([x1_val,x2_val,pT_val], [A_val,pdfu_val,pdffubar_val]))
# history = model.fit([x1_tr,x2_tr,pT_tr,qM_tr], [A_tr, pdfu_tr, pdfubar_tr], epochs=50, validation_data=([x1_val,x2_val,pT_val,qM_val], A_val))
history = model.fit([x1_tr,x2_tr,pT_tr,qM_tr], A_tr, epochs=100, validation_data=([x1_val,x2_val,pT_val,qM_val], A_val))


# Optional: plotting the loss curve
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('Loss_vs_Epochs_bestmodel.pdf')
plt.legend()
plt.show()

