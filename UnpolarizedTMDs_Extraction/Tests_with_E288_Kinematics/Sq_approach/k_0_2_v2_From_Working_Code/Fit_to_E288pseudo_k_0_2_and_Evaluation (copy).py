import tensorflow as tf
import pandas as pd
import numpy as np
#import lhapdf
import matplotlib.pyplot as plt
import os

k_upper = 2

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



NUM_REPLICAS = 10


################ Defining the DNN model ####################
Hidden_Layers=7
Nodes_per_HL=500
# Learning_Rate = 0.0001
Learning_Rate = 0.00001
L1_reg = 10**(-12)
# EPOCHS = 500
EPOCHS = 1000
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



# def create_nn_model(name):
#     inp = tf.keras.Input(shape=(3))
#     initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=42)
#     x = tf.keras.layers.Dense(240, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(inp)
#     x1 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x)
#     x2 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x1)
#     x3 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x2)
#     x4 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x3)
#     x5 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x4)
#     x6 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x5)
#     nnout = tf.keras.layers.Dense(1, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x6)
#     mod = tf.keras.Model(inp, nnout, name=name)
#     return mod



def create_nn_model(name):
    inp = tf.keras.Input(shape=(3))
    initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=42)
    x = tf.keras.layers.Dense(240, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(inp)
    x1 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x)
    x2 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x1)
    x3 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x2)
    x4 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x3)
    x5 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x4)
    x6 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x5)
    x7 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x6)
    x8 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x7)
    x9 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x8)
    x10 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x9)
    nnout = tf.keras.layers.Dense(1, activation='relu6', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x10)
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




model.compile(optimizer='adam', loss=mse_loss)

def run_replica(i):
    #replica_number = sys.argv[1]   # If you want to use this scrip for job submission, then uncomment this line, 
    #  then comment the following line, and then delete the 'i' in the parenthesis of run_replica(i) function's definition
    replica_number = i
    tempdf = GenerateReplicaData(df)
    trainKin, testKin, trainA, testA, trainAerr, testAerr, trainfq, testfq, trainfqbar, testfqbar, trainfq_rev, testfq_rev, trainfqbar_rev, testfqbar_rev = split_data(tempdf[['x1', 'x2', 'pT', 'QM']],
                                                                       tempdf['A'], tempdf['errA'], tempdf['fu_xA'], tempdf['fubar_xB'], tempdf['fu_xB'], tempdf['fubar_xA'], split=0.1)

    #model.compile(optimizer='adam', loss=mse_loss)
    # history = model.fit([tempdf['x1'],tempdf['x2'],tempdf['pT'],tempdf['QM']], [tempdf['A'], tempdf['fu_xA'], tempdf['fubar_xB']], epochs=EPOCHS, batch_size=32, verbose=2)
    history = model.fit([trainKin['x1'],trainKin['x2'],trainKin['pT'],trainKin['QM']], [trainA, trainfq, trainfqbar, trainfq_rev, trainfqbar_rev],  validation_data=([testKin['x1'],testKin['x2'],testKin['pT'],testKin['QM']], [testA, testfq, testfqbar, testfq_rev, testfqbar_rev]), epochs=EPOCHS, batch_size=BATCH, verbose=2)
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
    print('#################################')
    print('Completed replica')
    print(i)
    print('##################')
    print('The duration for this replica')
    print(finistime - starttime)



####################################################

def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")


create_folders('Evaluations_TMDs_with_kval')


Models_folder = 'DNNmodels'
folders_array=os.listdir(Models_folder)
numreplicas=len(folders_array)
# numreplicas=3

def mse_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    return mse_loss


modelsArray = []
for i in range(numreplicas):
    testmodel = tf.keras.models.load_model(str(Models_folder)+'/' + str(folders_array[i]),custom_objects={'mse_loss': mse_loss})
    modelsArray.append(testmodel)

modelsArray = np.array(modelsArray)

# model = tf.keras.models.load_model('model.h5',custom_objects={'mse_loss': mse_loss})
# modnnu = model.get_layer('nnu')
# modnnubar = model.get_layer('nnubar')


## Load the pdf grids
pdfs0 = pd.read_csv('Eval_PDFs/Eval_pdfs_0.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# pdfs1 = pd.read_csv('Eval_PDFs/Eval_pdfs_1.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# pdfs2 = pd.read_csv('Eval_PDFs/Eval_pdfs_2.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# pdfs3 = pd.read_csv('Eval_PDFs/Eval_pdfs_3.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# pdfs4 = pd.read_csv('Eval_PDFs/Eval_pdfs_4.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# pdfs5 = pd.read_csv('Eval_PDFs/Eval_pdfs_5.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# pdfs6 = pd.read_csv('Eval_PDFs/Eval_pdfs_6.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# pdfs7 = pd.read_csv('Eval_PDFs/Eval_pdfs_7.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# pdfs8 = pd.read_csv('Eval_PDFs/Eval_pdfs_8.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# pdfs9 = pd.read_csv('Eval_PDFs/Eval_pdfs_9.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')

#### S(k) used for pseudo-data ####

def Skq(k):
    return np.exp(-4*k**2/(4*k**2 + 4))

def Skqbar(k):
    return np.exp(-4*k**2/(4*k**2 + 1))



def nnq(model, flavor, kins):
    # For example: flavor = {'nnu', 'nnubar'}
    mod_out = tf.keras.backend.function(model.get_layer(flavor).input,
                                       model.get_layer(flavor).output)
    return mod_out(kins)

def Generate_Comparison_Plots_for_Kvalue(df, num_replicas, kvalue, output_name):
    x1vals = df['xA']
    x2vals = df['xB']
    pTvals = df['PT']
    QMvals = df['QM']

    fu_xA = df['fu_xA']
    fubar_xB = df['fubar_xB']

    fu_xB = df['fu_xB']
    fubar_xA = df['fubar_xA']

    #Kvals = np.linspace(0.1,2,len(x1vals))
    #Kvals = np.linspace(0,0,len(x1vals))
    Kvals = np.array([i*0 + kvalue for i in range(len(x1vals))])
    #pTvals = np.array([i*0 + 5 for i in range(len(pTvals))])
    #pT_k_vals = pTvals - Kvals
    pT_k_vals = Kvals

    true_values_1 = fu_xA * Skq(Kvals)  
    true_values_2 = fubar_xB * Skqbar(pT_k_vals) 

    # true_values_1 = fu_xA 
    # true_values_2 = fubar_xB

    true_values_1_rev = fu_xB * Skq(Kvals)
    true_values_2_rev = fubar_xA * Skqbar(pT_k_vals) 

    concatenated_inputs_1 = np.column_stack((x1vals,Kvals,QMvals))
    concatenated_inputs_2 = np.column_stack((x2vals,pT_k_vals,QMvals))


    tempfu = []
    tempfubar = []

    tempfu_rev = []
    tempfubar_rev = []

    for i in range(num_replicas):
        t = modelsArray[i]
        # modnnu = t.get_layer('nnu')
        # modnnubar = t.get_layer('nnubar')
        # temp_pred_fu = modnnu.predict(concatenated_inputs_1)
        # temp_pred_fubar = modnnubar.predict(concatenated_inputs_2)
        temp_pred_fu = nnq(t, 'nnu', concatenated_inputs_1)
        temp_pred_fubar = nnq(t, 'nnubar', concatenated_inputs_2)
        tempfu.append(list(temp_pred_fu))
        tempfubar.append(list(temp_pred_fubar))

        temp_pred_fu_rev = nnq(t, 'nnu', concatenated_inputs_2)
        temp_pred_fubar_rev = nnq(t, 'nnubar', concatenated_inputs_1)
        tempfu_rev.append(list(temp_pred_fu_rev))
        tempfubar_rev.append(list(temp_pred_fubar_rev))
    


    tempfu = np.array(tempfu)
    tempfu_mean = np.array(tempfu.mean(axis=0))
    tempfu_err = np.array(tempfu.std(axis=0))

    tempfubar = np.array(tempfubar)
    tempfubar_mean = np.array(tempfubar.mean(axis=0))
    tempfubar_err = np.array(tempfubar.std(axis=0))

    tempfu_rev = np.array(tempfu_rev)
    tempfu_mean_rev = np.array(tempfu_rev.mean(axis=0))
    tempfu_err_rev = np.array(tempfu_rev.std(axis=0))

    tempfubar_rev = np.array(tempfubar_rev)
    tempfubar_mean_rev = np.array(tempfubar_rev.mean(axis=0))
    tempfubar_err_rev = np.array(tempfubar_rev.std(axis=0))



    plt.figure(1, figsize=(20, 5))
    #################
    plt.subplot(1,4,1)
    plt.plot(x1vals, true_values_1, label='True nnu', linestyle='--')
    plt.plot(x1vals, tempfu_mean, 'r', label='Predicted nnu')
    plt.fill_between(x1vals, (tempfu_mean-tempfu_err).flatten(), (tempfu_mean+tempfu_err).flatten(), facecolor='r', alpha=0.3)
    plt.title(f'f(xA, k ={kvalue})')
    plt.xlabel('xA')
    plt.ylabel('nnu')
    plt.legend()
    plt.grid(True)
    #################
    plt.subplot(1,4,2)
    plt.plot(x2vals, true_values_2, label='True nnubar', linestyle='--')
    plt.plot(x2vals, tempfubar_mean, 'r', label='Predicted nnubar')
    plt.fill_between(x2vals, (tempfubar_mean-tempfubar_err).flatten(), (tempfubar_mean+tempfubar_err).flatten(), facecolor='r', alpha=0.3)
    #plt.title('Comparison of True and Predicted nnubar PDF Values')
    plt.xlabel('xB')
    plt.ylabel('nnubar')
    plt.legend()
    plt.title(f'fbar(xB, k ={kvalue})')
    plt.grid(True)
    ################
    plt.subplot(1,4,3)
    plt.plot(x2vals, true_values_1_rev, label='True nnu', linestyle='--')
    plt.plot(x2vals, tempfu_mean_rev, 'r', label='Predicted nnu')
    plt.fill_between(x2vals, (tempfu_mean_rev-tempfu_err_rev).flatten(), (tempfu_mean_rev+tempfu_err_rev).flatten(), facecolor='r', alpha=0.3)
    plt.title(f'f(xB, k ={kvalue})')
    plt.xlabel('xB')
    plt.ylabel('nnu')
    plt.legend()
    plt.grid(True)
    ################
    plt.subplot(1,4,4)
    plt.plot(x1vals, true_values_2_rev, label='True nnubar', linestyle='--')
    plt.plot(x1vals, tempfubar_mean_rev, 'r', label='Predicted nnubar')
    plt.fill_between(x1vals, (tempfubar_mean_rev-tempfubar_err_rev).flatten(), (tempfubar_mean_rev+tempfubar_err_rev).flatten(), facecolor='r', alpha=0.3)
    plt.title(f'fbar(xA, k ={kvalue})')
    plt.xlabel('xA')
    plt.ylabel('nnubar')
    plt.legend()
    plt.grid(True) 
    plt.savefig('Evaluations_TMDs_with_kval/True_vs_Pred_fxk_'+str(output_name)+'.pdf')
    plt.close()


Generate_Comparison_Plots_for_Kvalue(pdfs0,numreplicas, 0, 'set_0_k=0')
Generate_Comparison_Plots_for_Kvalue(pdfs0,numreplicas, 0.5, 'set_0_k=0.5')
Generate_Comparison_Plots_for_Kvalue(pdfs0,numreplicas, 1, 'set_0_k=1')
Generate_Comparison_Plots_for_Kvalue(pdfs0,numreplicas, 1.5, 'set_0_k=1.5')
Generate_Comparison_Plots_for_Kvalue(pdfs0,numreplicas, 2, 'set_0_k=2')
