import tensorflow as tf
import pandas as pd
import numpy as np
#import lhapdf
import matplotlib.pyplot as plt
import os
import time
from tensorflow_addons.activations import tanhshrink
tf.keras.utils.get_custom_objects().update({'tanhshrink': tanhshrink})


np.random.seed(42)  # Seed for reproducibility

def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")
        

create_folders('DNNmodels')
create_folders('Losses_Plots')

########### Import pseudodata file 
df = pd.read_csv('E288_pseudo_data.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')


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
    #pseudodata_df['A'] = df['A']
    tempA = df['A']
    tempAerr = np.abs(np.array(df['errA']))
    pseudodata_df['A'] = np.random.normal(loc=tempA, scale=tempAerr) 
    pseudodata_df['errA'] = tempAerr
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


# testdf = GenerateReplicaData(df)
# print(testdf)



NUM_REPLICAS = 50


################ Defining the DNN model ####################
#Hidden_Layers=7
#Nodes_per_HL=500
# Learning_Rate = 0.0001
Learning_Rate = 0.001
L1_reg = 10**(-12)
EPOCHS = 200
BATCH = 64


# def create_nn_model(name):
#     inp = tf.keras.Input(shape=(3))
#     initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=42)
#     x = tf.keras.layers.Dense(240, activation='tanh', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(inp)
#     x1 = tf.keras.layers.Dense(32, activation='relu', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x)
#     x2 = tf.keras.layers.Dense(32, activation='relu', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x1)
#     x3 = tf.keras.layers.Dense(32, activation='relu', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x2)
#     x4 = tf.keras.layers.Dense(32, activation='relu', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x3)
#     x5 = tf.keras.layers.Dense(32, activation='relu', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x4)
#     x6 = tf.keras.layers.Dense(32, activation='relu', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x5)
#     nnout = tf.keras.layers.Dense(1, activation='relu', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x6)
#     mod = tf.keras.Model(inp, nnout, name=name)
#     return mod


def create_nn_model(name):
    inp = tf.keras.Input(shape=(3))
    initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=None)
    x = tf.keras.layers.Dense(256, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(inp)
    x1 = tf.keras.layers.Dense(192, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x)
    x2 = tf.keras.layers.Dense(128, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x1)
    x3 = tf.keras.layers.Dense(64, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x2)
    x4 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x3)
    x5 = tf.keras.layers.Dense(16, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x4)
    x6 = tf.keras.layers.Dense(8, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x5)
    nnout = tf.keras.layers.Dense(1, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x6)
    mod = tf.keras.Model(inp, nnout, name=name)
    return mod



def createModel_DY():
    x1 = tf.keras.Input(shape=(1), name='x1')
    x2 = tf.keras.Input(shape=(1), name='x2')
    qT = tf.keras.Input(shape=(1), name='qT')
    qM = tf.keras.Input(shape=(1), name='QM')

    modnnu = create_nn_model('nnu')
    modnnubar = create_nn_model('nnubar')

    modnnd = create_nn_model('nnd')
    modnndbar = create_nn_model('nndbar')

    modnns = create_nn_model('nns')
    modnnsbar = create_nn_model('nnsbar')

    pdf_k_val = 0    
    nnq_pdf_input = tf.keras.layers.Concatenate()([x1, qT*0 + pdf_k_val, qM])
    nnqbar_pdf_input = tf.keras.layers.Concatenate()([x2, qT*0 - pdf_k_val, qM])

    modnnu_pdf_eval = modnnu(nnq_pdf_input)
    modnnubar_pdf_eval = modnnubar(nnqbar_pdf_input)

    modnnd_pdf_eval = modnnd(nnq_pdf_input)
    modnndbar_pdf_eval = modnndbar(nnqbar_pdf_input)

    modnns_pdf_eval = modnns(nnq_pdf_input)
    modnnsbar_pdf_eval = modnnsbar(nnqbar_pdf_input)


    k_values = tf.linspace(0.1, 2.0, 100)
    dk = k_values[1] - k_values[0]

    tmd1_list, tmd2_list = [], []
    product_list = []  # List to store TMD values for each k value
    for k_val in k_values:      
        nnq_input = tf.keras.layers.Concatenate()([x1, qT*0 + k_val, qM])
        nnqbar_input = tf.keras.layers.Concatenate()([x2, qT - k_val, qM])

        nnu_x1 = modnnu(nnq_input)
        nnubar_x2 = modnnubar(nnqbar_input)
        nnu_x1_rev = modnnu(nnqbar_input)
        nnubar_x2_rev = modnnubar(nnq_input)

        nnd_x1 = modnnd(nnq_input)
        nndbar_x2 = modnndbar(nnqbar_input)
        nnd_x1_rev = modnnd(nnqbar_input)
        nndbar_x2_rev = modnndbar(nnq_input)

        nns_x1 = modnns(nnq_input)
        nnsbar_x2 = modnnsbar(nnqbar_input)
        nns_x1_rev = modnns(nnqbar_input)
        nnsbar_x2_rev = modnnsbar(nnq_input)

        product_1u = tf.multiply(nnu_x1, nnubar_x2)
        product_2u = tf.multiply(nnu_x1_rev, nnubar_x2_rev)

        product_1d = tf.multiply(nnd_x1, nndbar_x2)
        product_2d = tf.multiply(nnd_x1_rev, nndbar_x2_rev)

        product_1s = tf.multiply(nns_x1, nnsbar_x2)
        product_2s = tf.multiply(nns_x1_rev, nnsbar_x2_rev)

        result1 = tf.add(product_1u, product_2u)
        result2 = tf.add(product_1d, product_2d)
        result3 = tf.add(product_1s, product_2s)
        result = result1+result2+result3
        product_list.append(result)

    # Summing over all k values using tf.reduce_sum
    tmd_product_sum = tf.reduce_sum(product_list, axis=0) * dk 

    return tf.keras.Model([x1, x2, qT, qM], [tmd_product_sum, modnnu_pdf_eval, modnnubar_pdf_eval, modnnd_pdf_eval, modnndbar_pdf_eval, modnns_pdf_eval, modnnsbar_pdf_eval])

model = createModel_DY()

def mse_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    return mse_loss


# def custom_loss(y_true, y_pred):
#     return mse_loss(y_true, y_pred) 


def split_data(X,y,yerr, fu, fubar, fd, fdbar, fs, fsbar, split=0.1):
  temp =np.random.choice(list(range(len(y))), size=int(len(y)*split), replace = False)

  test_X = pd.DataFrame.from_dict({k: v[temp] for k,v in X.items()})
  train_X = pd.DataFrame.from_dict({k: v.drop(temp) for k,v in X.items()})

  test_y = y[temp]
  train_y = y.drop(temp)

  test_yerr = yerr[temp]
  train_yerr = yerr.drop(temp)

  test_fu = fu[temp]
  train_fu = fu.drop(temp)
  test_fubar = fubar[temp]
  train_fubar = fubar.drop(temp)

  test_fd = fd[temp]
  train_fd = fd.drop(temp)
  test_fdbar = fdbar[temp]
  train_fdbar = fdbar.drop(temp)

  test_fs = fs[temp]
  train_fs = fs.drop(temp)
  test_fsbar = fsbar[temp]
  train_fsbar = fsbar.drop(temp)

  return train_X, test_X, train_y, test_y, train_yerr, test_yerr, train_fu, test_fu, train_fubar, test_fubar, train_fd, test_fd, train_fdbar, test_fdbar, train_fs, test_fs, train_fsbar, test_fsbar



# model.compile(optimizer='adam', loss=mse_loss)
# # history = model.fit([df['x1'],df['x2'],df['pT']], df['A'], epochs=EPOCHS, batch_size=32, verbose=2)
# history = model.fit([df['x1'],df['x2'],df['pT'],df['QM']], [df['A'], fu, fubar], epochs=EPOCHS, batch_size=32, verbose=2)
# model.save('model.h5', save_format='h5')


# predictions = model.predict([df['x1'],df['x2'],df['pT'],df['QM']])[0]

# # 3D scatter plot
# fig = plt.figure(2)
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x1Vals, pTVals, df['A'], c='r', marker='o', label='Actual')
# ax.scatter(x1Vals, pTVals, predictions, c='b', marker='^', label='Predicted')
# ax.set_xlabel('x1')
# ax.set_ylabel('pT')
# ax.set_zlabel('A')
# ax.set_title('Actual vs Predicted')
# ax.legend()
# #plt.show()
# plt.savefig('Actual_vs_Predicted_Integral.pdf')


model.compile(optimizer='adam', loss=mse_loss)

def run_replica(i):
    #replica_number = sys.argv[1]   # If you want to use this scrip for job submission, then uncomment this line, 
    #  then comment the following line, and then delete the 'i' in the parenthesis of run_replica(i) function's definition
    replica_number = i
    tempdf = GenerateReplicaData(df)
    # x1Vals = np.array(tempdf['x1'])
    # x2Vals = np.array(tempdf['x2'])
    # pTVals = np.array(tempdf['pT'])
    # QMvals = np.array(tempdf['QM'])
    # Kvals = np.linspace(0.1,2,len(x1Vals))
    # pT_k_vals = pTVals - Kvals
    # fu = tempdf['fu_xA']
    # fubar = tempdf['fubar_xB']


    trainKin, testKin, trainA, testA, trainAerr, testAerr, trainfu, testfu, trainfubar, testfubar, trainfd, testfd, trainfdbar, testfdbar, trainfs, testfs, trainfsbar, testfsbar = split_data(tempdf[['x1', 'x2', 'pT', 'QM']],
                                                                       tempdf['A'], tempdf['errA'], tempdf['fu_xA'], tempdf['fubar_xB'], tempdf['fd_xA'], tempdf['fdbar_xB'], tempdf['fs_xA'], tempdf['fsbar_xB'], split=0.1)

    # model.compile(optimizer='adam', loss=mse_loss)
    # history = model.fit([tempdf['x1'],tempdf['x2'],tempdf['pT'],tempdf['QM']], [tempdf['A'], tempdf['fu_xA'], tempdf['fubar_xB']], epochs=EPOCHS, batch_size=32, verbose=2)
    history = model.fit([trainKin['x1'],trainKin['x2'],trainKin['pT'],trainKin['QM']], [trainA, trainfu, trainfubar, trainfd, trainfdbar, trainfs, trainfsbar],  validation_data=([testKin['x1'],testKin['x2'],testKin['pT'],testKin['QM']], [testA, testfu, testfubar, testfd, testfdbar, testfs, testfsbar]), epochs=EPOCHS, batch_size=BATCH, verbose=2)
    model.save('DNNmodels/' + 'model' + str(replica_number) + '.h5', save_format='h5')

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
    time.sleep(20)
    print('Completed replica: '+str(i))
    print('The duration for this replica: '+str(finistime - starttime))
    tf.keras.backend.clear_session()
