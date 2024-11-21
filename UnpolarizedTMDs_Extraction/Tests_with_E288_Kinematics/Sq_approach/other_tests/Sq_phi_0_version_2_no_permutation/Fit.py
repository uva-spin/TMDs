import tensorflow as tf
import pandas as pd
import numpy as np
#import lhapdf
import matplotlib.pyplot as plt
import os
import sys

k_lower = 0.001
k_upper = 6
kBins = 100
#phiBins = 10
NUM_REPLICAS = 3

def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")
        
#models_path = '/scratch/cee9hc/Unpolarized_TMD/E288/flavor_1/Phase_2/Test_05_k_0-001_6_phi_0'
#create_folders(str(models_path))
Models_folder = 'DNNmodels'
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

################ Defining the DNN model ####################
#Hidden_Layers=7
#Nodes_per_HL=500
#Learning_Rate = 0.00001
#L1_reg = 10**(-12)
#EPOCHS = 2000
#BATCH = 64

Hidden_Layers=7
Nodes_per_HL=500
Learning_Rate = 0.01
L1_reg = 10**(-12)
EPOCHS = 20
BATCH = 64


def create_nn_model(name):
    inp = tf.keras.Input(shape=(2))
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


#def kB(qT,k,phi):
#    return np.sqrt(qT**2 + k**2 - 2*qT*k*np.cos(phi))

def kB(qT, k, phi):
    k = tf.convert_to_tensor(k, dtype=qT.dtype)
    phi = tf.convert_to_tensor(phi, dtype=qT.dtype)
    return tf.sqrt(qT**2 + k**2 - 2*qT*k*tf.cos(phi))



def createModel_DY():
    x1 = tf.keras.Input(shape=(1,), name='x1')
    x2 = tf.keras.Input(shape=(1,), name='x2')
    qT = tf.keras.Input(shape=(1,), name='qT')
    qM = tf.keras.Input(shape=(1,), name='qM')
    fuxA = tf.keras.Input(shape=(1,), name='fu_xA')
    fubarxA = tf.keras.Input(shape=(1,), name='fubar_xA')
    fuxB = tf.keras.Input(shape=(1,), name='fu_xB')
    fubarxB = tf.keras.Input(shape=(1,), name='fubar_xB')

    modnnu = create_nn_model('nnu')
    modnnubar = create_nn_model('nnubar')

    k_values = tf.linspace(k_lower, k_upper, kBins)
    dk = k_values[1] - k_values[0]

    k_product_list = []
    Sq_int, Sqbar_int, Sq_int_rev, Sqbar_int_rev = [], [], [], []

    for k_val in k_values:
        
        tempkB = kB(qT, k_val, 0)

        # Inputs 
        nnu_input = tf.keras.layers.Concatenate()([qT * 0 + k_val, qM])
        nnubar_input = tf.keras.layers.Concatenate()([qT * 0 + tempkB, qM])

        nnu_x1 = modnnu(nnu_input)
        nnubar_x2 = modnnubar(nnubar_input)
        nnu_x1_rev = modnnu(nnubar_input)
        nnubar_x2_rev = modnnubar(nnu_input)

        product_1 = tf.multiply(nnu_x1, nnubar_x2)*fuxA*fubarxB
        product_2 = tf.multiply(nnu_x1_rev, nnubar_x2_rev)*fuxB*fubarxA
        result = tf.add(product_1, product_2)
        result = tf.multiply(k_val,result)

        #phi_product_list.append(result)

        # Sum over all phi
        #tmd_phi_product_sum = tf.reduce_sum(phi_product_list, axis=0) * dphi

        # Append the sum to the k product list
        #k_product_list.append(tmd_phi_product_sum)
        k_product_list.append(result)

        # Compute f_xA, fbar_xB, f_xB, fbar_xA for the current k_val
        Sq_input = tf.keras.layers.Concatenate()([qT * 0 + k_val, qM])
        Sqbar_input = tf.keras.layers.Concatenate()([qT * 0 + k_val, qM])
        Sq_int.append(modnnu(Sq_input))
        Sqbar_int.append(modnnubar(Sqbar_input))
        Sq_int_rev.append(modnnu(Sqbar_input))
        Sqbar_int_rev.append(modnnubar(Sq_input))

    # Summing over all k values 
    tmd_product_sum = tf.reduce_sum(k_product_list, axis=0) * dk

    alpha = 1/137
    mult = ((4*np.pi*alpha)**2)/(9*qM**3)
    CrossSection = mult*qT*tmd_product_sum


    # k_perp Integrals
    Sq_sum = tf.reduce_sum(Sq_int, axis=0) * dk
    Sqbar_sum = tf.reduce_sum(Sqbar_int, axis=0) * dk
    Sq_sum_rev = tf.reduce_sum(Sq_int_rev, axis=0) * dk
    Sqbar_sum_rev = tf.reduce_sum(Sqbar_int_rev, axis=0) * dk

    # Create and return the model
    return tf.keras.Model(inputs=[x1, x2, qT, qM, fuxA, fubarxA, fuxB, fubarxB], outputs=[CrossSection, Sq_sum, Sqbar_sum, Sq_sum_rev, Sqbar_sum_rev])


model = createModel_DY()

def mse_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    return mse_loss


# def custom_loss(y_true, y_pred):
#     return mse_loss(y_true, y_pred) 


def split_data(X,y,yerr, split=0.1):
  temp =np.random.choice(list(range(len(y))), size=int(len(y)*split), replace = False)

  test_X = pd.DataFrame.from_dict({k: v[temp] for k,v in X.items()})
  train_X = pd.DataFrame.from_dict({k: v.drop(temp) for k,v in X.items()})

  test_y = y[temp]
  train_y = y.drop(temp)

  test_yerr = yerr[temp]
  train_yerr = yerr.drop(temp)

#  test_fq = fq[temp]
#  train_fq = fq.drop(temp)
#
#  test_fqbar = fqbar[temp]
#  train_fqbar = fqbar.drop(temp)
#
#  test_fq_rev = fq_rev[temp]
#  train_fq_rev = fq_rev.drop(temp)
#
#  test_fqbar_rev = fqbar_rev[temp]
#  train_fqbar_rev = fqbar_rev.drop(temp)

  return train_X, test_X, train_y, test_y, train_yerr, test_yerr



model.compile(optimizer='adam', loss=mse_loss)

def run_replica(i):
    #replica_number = sys.argv[1]   # If you want to use this scrip for job submission, then uncomment this line,
    #  then comment the following line, and then delete the 'i' in the parenthesis of run_replica(i) function's definition
    replica_number = i
    #replica_number = 99
    tempdf = GenerateReplicaData(df)
    trainKin, testKin, trainA, testA, trainAerr, testAerr = split_data(tempdf[['x1', 'x2', 'pT', 'QM', 'fu_xA','fubar_xA','fu_xB','fubar_xB']],tempdf['A'], tempdf['errA'], split=0.1)

    #model.compile(optimizer='adam', loss=mse_loss)
    # history = model.fit([tempdf['x1'],tempdf['x2'],tempdf['pT'],tempdf['QM']], [tempdf['A'], tempdf['fu_xA'], tempdf['fubar_xB']], epochs=EPOCHS, batch_size=32, verbose=2)
#    history = model.fit([trainKin['x1'],trainKin['x2'],trainKin['pT'],trainKin['QM'],trainKin['fu_xA'],trainKin['fubar_xA'],trainKin['fu_xB'],trainKin['fubar_xB']], [trainA, trainfq, trainfqbar, trainfq_rev, trainfqbar_rev],  validation_data=([testKin['x1'],testKin['x2'],testKin['pT'],testKin['QM'],testKin['fu_xA'],testKin['fubar_xA'],testKin['fu_xB'],testKin['fubar_xB']], [testA, testfq, testfqbar, testfq_rev, testfqbar_rev]), epochs=EPOCHS, batch_size=BATCH, verbose=2)

   # Creating arrays of ones with the length of the training and testing data
    ones_train = (1/(2*np.pi))*np.ones_like(trainKin['x1'])  # Array of ones with the length of training data
    ones_test = (1/(2*np.pi))*np.ones_like(testKin['x1'])    # Array of ones with the length of testing data
    
    # Model fitting with the added array of ones
    history = model.fit(
        [trainKin['x1'], trainKin['x2'], trainKin['pT'], trainKin['QM'], trainKin['fu_xA'], trainKin['fubar_xA'], trainKin['fu_xB'], trainKin['fubar_xB']],
        [trainA, ones_train, ones_train, ones_train, ones_train], validation_data=( [testKin['x1'], testKin['x2'], testKin['pT'], testKin['QM'], testKin['fu_xA'], testKin['fubar_xA'], testKin['fu_xB'], testKin['fubar_xB']],
            [testA, ones_test, ones_test, ones_test, ones_test]), epochs=EPOCHS, batch_size=BATCH, verbose=2)
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
    

#run_replica()
for i in range(0,NUM_REPLICAS):
    run_replica(i)

