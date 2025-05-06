import tensorflow as tf
import pandas as pd
import numpy as np
#import lhapdf
import matplotlib.pyplot as plt
import os

k_lower = 0
k_upper = 10
kBins = 10
phiBins = 10

def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")
        
Models_folder = 'DNNmodels'
create_folders('DNNmodels')
create_folders('Losses_Plots')

########### Import pseudodata file 
df = pd.read_csv('results.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')


####### Here we define a function that can sample cross-section within errA ###
def GenerateReplicaData(df):
    pseudodata_df = {'x1': [],
                     'x2': [],
                     'qT': [],
                     'SqT': [],
                     'SqT_err':[]}
    #pseudodata_df = pd.DataFrame(pseudodata_df)
    pseudodata_df['x1'] = df['x1']
    pseudodata_df['x2'] = df['x2']
    pseudodata_df['qT'] = df['qT']
    pseudodata_df['SqT_err'] = df['SqT_err']
    tempA = df['SqT']
    tempAerr = np.abs(np.array(df['SqT_err'])) 
    pseudodata_df['SqT'] = np.random.normal(loc=tempA, scale=tempAerr)
    return pd.DataFrame(pseudodata_df)


NUM_REPLICAS = 3


################ Defining the DNN model ####################
# Hidden_Layers=7
# Nodes_per_HL=500
# Learning_Rate = 0.00001
# L1_reg = 10**(-12)
# EPOCHS = 2
# BATCH = 64

Hidden_Layers=7
Nodes_per_HL=100
Learning_Rate = 0.00001
L1_reg = 10**(-12)
EPOCHS = 1000
BATCH = 64




# def create_nn_model(name):
#     inp = tf.keras.Input(shape=(2))
#     initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=42)
#     x = tf.keras.layers.Dense(240, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(inp)
#     x1 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x)
#     x2 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x1)
#     x3 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x2)
#     x4 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x3)
#     x5 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x4)
#     x6 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x5)
#     x7 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x6)
#     x8 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x7)
#     x9 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x8)
#     x10 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x9)
#     nnout = tf.keras.layers.Dense(1, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x10)
#     mod = tf.keras.Model(inp, nnout, name=name)
#     return mod

def create_nn_model(name):
    inp = tf.keras.Input(shape=(3,))
    initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=42)
    x = tf.keras.layers.Dense(240, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(inp)
    x1 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x)
    x2 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x1)
    nnout = tf.keras.layers.Dense(1, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x2)
    mod = tf.keras.Model(inp, nnout, name=name)
    return mod


def kB(qT, k, phi):
    k = tf.convert_to_tensor(k, dtype=qT.dtype)
    phi = tf.convert_to_tensor(phi, dtype=qT.dtype)
    return tf.sqrt(qT**2 + k**2 - 2*qT*k*tf.cos(phi))


def createModel_SqT():
    qT = tf.keras.Input(shape=(1,), name='qT')
    x1 = tf.keras.Input(shape=(1,), name='x1')
    x2 = tf.keras.Input(shape=(1,), name='x2')

    modn1 = create_nn_model('n1')
    modn2 = create_nn_model('n2')

    k_values = tf.linspace(k_lower, k_upper, kBins)
    dk = k_values[1] - k_values[0]

    phi_values = tf.linspace(0, np.pi, phiBins)
    dphi = phi_values[1] - phi_values[0]

    k_product_list = []
    phi_product_list = []

    for k_val in k_values:
        for phi_val in phi_values:
            tempkB = kB(qT, k_val, phi_val)

            # Inputs 
            n1_input = tf.keras.layers.Concatenate()([x1,k_val])
            n2_input = tf.keras.layers.Concatenate()([x2, tempkB])

            nn1 = modn1(n1_input)
            nn2 = modn2(n2_input)

            Skphi_product = tf.multiply(nn1, nn2)
            phi_product_list.append(Skphi_product)
        phi_integral = tf.reduce_sum(phi_product_list) * dphi

        k_product_list.append(phi_integral)
    # Integrate over k
    total_integral = tf.reduce_sum(k_product_list) * dk

    # Create and return the model
    return tf.keras.Model(inputs=[qT,x1, x2], outputs=[total_integral])


    

model = createModel_SqT()

def mse_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    return mse_loss


def split_data(X,y,yerr, split=0.1):
  temp =np.random.choice(list(range(len(y))), size=int(len(y)*split), replace = False)

  test_X = pd.DataFrame.from_dict({k: v[temp] for k,v in X.items()})
  train_X = pd.DataFrame.from_dict({k: v.drop(temp) for k,v in X.items()})

  test_y = y[temp]
  train_y = y.drop(temp)

  test_yerr = yerr[temp]
  train_yerr = yerr.drop(temp)

  return train_X, test_X, train_y, test_y, train_yerr, test_yerr




model.compile(optimizer='adam', loss=mse_loss)


def run_replica(i):
    #replica_number = sys.argv[1]   # If you want to use this scrip for job submission, then uncomment this line,
    #  then comment the following line, and then delete the 'i' in the parenthesis of run_replica(i) function's definition
    replica_number = i
    #replica_number = 99
    tempdf = GenerateReplicaData(df)
    trainKin, testKin, trainA, testA, trainAerr, testAerr = split_data(tempdf[['qT','x1', 'x2']],tempdf['SqT'], tempdf['SqT_err'], split=0.1)
  
    # Model fitting with the added array of ones
    history = model.fit(
        [trainKin['qT'],trainKin['x1'], trainKin['x2']],
        [trainA], validation_data=( [testKin['qT'],testKin['x1'], testKin['x2']],
            [testA]), epochs=EPOCHS, batch_size=BATCH, verbose=2)
    model.save(str(Models_folder) + '/' + 'model' + str(replica_number) + '.h5', save_format='h5')

    # Create subplots for loss plots
    plt.figure(1,figsize=(12, 5))
    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Val. loss')
    plt.title('Losses')
    #plt.ylim(0,0.01)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Losses_Plots/' + 'loss_plots' + str(replica_number) + '.pdf')
    plt.close()


import datetime
###### Running Jobs on Rivanna: Comment the following lines and uncomment the run_replica(), uncomment replica_number = sys.argv[1] and comment replica_number = i in the 'def run_replica()'  
for i in range(0,NUM_REPLICAS):
    starttime = datetime.datetime.now().replace(microsecond=0)
    run_replica(i)
    finistime = datetime.datetime.now().replace(microsecond=0)
    print('#################################')
    print('Completed replica')
    print(i)
    print('##################')
    print('The duration for this replica')
    print(finistime - starttime)
