import tensorflow as tf
import pandas as pd
import numpy as np
#import lhapdf
import matplotlib.pyplot as plt
import os
import sys

k_upper = 2

def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")
        
models_path = '/scratch/cee9hc/Unpolarized_TMD/E288/flavor_1/Arc_06_batch_128'
create_folders(str(models_path))
Models_folder = str(models_path)+'/'+'DNNmodels'
create_folders(str(Models_folder))
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


# testdf = GenerateReplicaData(df)
# print(testdf)



NUM_REPLICAS = 3


################ Defining the DNN model ####################
Hidden_Layers=7
Nodes_per_HL=500
Learning_Rate = 0.00001
L1_reg = 10**(-12)
EPOCHS = 1000
BATCH = 128




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



#def create_nn_model(name):
#    inp = tf.keras.Input(shape=(3))
#    initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=42)
#    x = tf.keras.layers.Dense(240, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(inp)
#    x1 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x)
#    x2 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x1)
#    x3 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x2)
#    x4 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x3)
#    x5 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x4)
#    x6 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x5)
#    nnout = tf.keras.layers.Dense(1, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x6)
#    mod = tf.keras.Model(inp, nnout, name=name)
#    return mod


## Loss function were not good
#def create_nn_model(name):
#    inp = tf.keras.Input(shape=(3))
#    initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=42)
#    x = tf.keras.layers.Dense(240, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(inp)
#    x1 = tf.keras.layers.Dense(200, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x)
#    x2 = tf.keras.layers.Dense(160, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x1)
#    x3 = tf.keras.layers.Dense(120, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x2)
#    x4 = tf.keras.layers.Dense(80, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x3)
#    x5 = tf.keras.layers.Dense(40, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x4)
#    x6 = tf.keras.layers.Dense(10, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x5)
#    nnout = tf.keras.layers.Dense(1, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x6)
#    mod = tf.keras.Model(inp, nnout, name=name)
#    return mod


## Was good
#def create_nn_model(name):
#    inp = tf.keras.Input(shape=(3))
#    initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=42)
#    x = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(inp)
#    x1 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x)
#    x2 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x1)
#    x3 = tf.keras.layers.Dense(240, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x2)
#    x4 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x3)
#    x5 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x4)
#    x6 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x5)
#    nnout = tf.keras.layers.Dense(1, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x6)
#    mod = tf.keras.Model(inp, nnout, name=name)
#    return mod


#def create_nn_model(name):
#    inp = tf.keras.Input(shape=(3))
#    initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=42)
#    x = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(inp)
#    x1 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x)
#    x2 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x1)
#    x3 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x2)
#    x4 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x3)
#    x5 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x4)
#    x6 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x5)
#    nnout = tf.keras.layers.Dense(1, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x6)
#    mod = tf.keras.Model(inp, nnout, name=name)
#    return mod


def create_nn_model(name):
    inp = tf.keras.Input(shape=(3))
    initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=42)
    x = tf.keras.layers.Dense(16, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(inp)
    x1 = tf.keras.layers.Dense(16, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x)
    x2 = tf.keras.layers.Dense(16, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x1)
    x3 = tf.keras.layers.Dense(16, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x2)
    x4 = tf.keras.layers.Dense(16, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x3)
    x5 = tf.keras.layers.Dense(16, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x4)
    x6 = tf.keras.layers.Dense(16, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x5)
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

    pdf_k_val = 0    
    # nnu_pdf_input = tf.keras.layers.Concatenate()([x1, qT*0 + pdf_k_val, qM])
    # nnubar_pdf_input = tf.keras.layers.Concatenate()([x2, qT*0 - pdf_k_val, qM])

    # modnnu_pdf_eval = modnnu(nnu_pdf_input)
    # modnnubar_pdf_eval = modnnubar(nnubar_pdf_input)

    # modnnu_pdf_eval_rev = modnnu(nnubar_pdf_input)
    # modnnubar_pdf_eval_rev = modnnubar(nnu_pdf_input)

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


    return tf.keras.Model([x1, x2, qT, qM], [tmd_product_sum, f_xA_sum, fbar_xB_sum, f_xB_sum, fbar_xA_sum])

model = createModel_DY()

def mse_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    return mse_loss


# def custom_loss(y_true, y_pred):
#     return mse_loss(y_true, y_pred) 


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

def run_replica():
    replica_number = sys.argv[1]   # If you want to use this scrip for job submission, then uncomment this line, 
    #  then comment the following line, and then delete the 'i' in the parenthesis of run_replica(i) function's definition
    # replica_number = i
    tempdf = GenerateReplicaData(df)
    trainKin, testKin, trainA, testA, trainAerr, testAerr, trainfq, testfq, trainfqbar, testfqbar, trainfq_rev, testfq_rev, trainfqbar_rev, testfqbar_rev = split_data(tempdf[['x1', 'x2', 'pT', 'QM']],
                                                                       tempdf['A'], tempdf['errA'], tempdf['fu_xA'], tempdf['fubar_xB'], tempdf['fu_xB'], tempdf['fubar_xA'], split=0.1)

    #model.compile(optimizer='adam', loss=mse_loss)
    # history = model.fit([tempdf['x1'],tempdf['x2'],tempdf['pT'],tempdf['QM']], [tempdf['A'], tempdf['fu_xA'], tempdf['fubar_xB']], epochs=EPOCHS, batch_size=32, verbose=2)
    history = model.fit([trainKin['x1'],trainKin['x2'],trainKin['pT'],trainKin['QM']], [trainA, trainfq, trainfqbar, trainfq_rev, trainfqbar_rev],  validation_data=([testKin['x1'],testKin['x2'],testKin['pT'],testKin['QM']], [testA, testfq, testfqbar, testfq_rev, testfqbar_rev]), epochs=EPOCHS, batch_size=BATCH, verbose=2)
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
    

run_replica()