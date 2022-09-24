import os
import tensorflow as tf
import pandas as pd
import numpy as np
import lhapdf
import matplotlib.pyplot as plt
import csv
import sys

save_path = 'NN_SIDIS_Models/rep'
Org_Data_path = 'Data/'
HERMES2009 = 'HERMES_p_2009.csv'
HERMES2020 = 'HERMES_p_2020.csv'
COMPASS2009 = 'COMPASS_d_2009.csv'
COMPASS2015 = 'COMPASS_p_2015.csv'
#Replica_Output_File='Replicas.csv'

SIDIS_Repl_Folder = 'SIDIS_Replica_Data'
Losses_Folder = 'SIDIS_Replicas_Losses'

herm09 = pd.read_csv(Org_Data_path + HERMES2009).dropna(axis=0, how='all').dropna(axis=1, how='all')
herm20 = pd.read_csv(Org_Data_path + HERMES2020).dropna(axis=0, how='all').dropna(axis=1, how='all')
comp09 = pd.read_csv(Org_Data_path + COMPASS2009).dropna(axis=0, how='all').dropna(axis=1, how='all')
comp15 = pd.read_csv(Org_Data_path + COMPASS2015).dropna(axis=0, how='all').dropna(axis=1, how='all')

df = pd.concat([herm09, herm20, comp09, comp15])
#df = pd.concat([herm20, comp09, comp15])

Nrep = 2

Hidden_Layers=7
Nodes_per_HL=256
Learning_Rate = 0.0001
EPOCHS = 100000
#EPOCHS = 30
L1_reg = 10**(-12)
modify_LR = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',factor=0.9,patience=400,mode='auto')
EarlyStop = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=1000)
uncert_frac = 1

def chisquare(y, yhat, err):
    return np.sum(((y - yhat)/err)**2)

        
class A0(tf.keras.layers.Layer):
    def __init__(self, kperp2avg=.57, pperp2avg=.12, **kwargs):
        super(A0, self).__init__(name='a0')
        self.m1 = tf.Variable(1., name='m1')
        #self.m1 = tf.constant(1.)
        self.kperp2avg = kperp2avg
        self.pperp2avg = pperp2avg
        self.e = tf.constant(1.)
        
    def get_config(self):
        config = super().get_config().copy()
        
        config.update({
            'kperp2avg': self.kperp2avg,
            'pperp2avg': self.pperp2avg
        })
        
        return config
        
    def call(self, inputs):
        z = inputs[:, 0]
        pht = inputs[:, 1]
        ks2avg = (self.kperp2avg*self.m1**2)/(self.m1**2 + self.kperp2avg)
        topfirst = (z**2 * self.kperp2avg + self.pperp2avg) * ks2avg**2
        bottomfirst = (z**2 * ks2avg + self.pperp2avg)**2 * self.kperp2avg 
        exptop = pht**2 * z**2 * (ks2avg - self.kperp2avg)
        expbottom = (z**2 * ks2avg + self.pperp2avg) * (z**2 * self.kperp2avg + self.pperp2avg)
        last = tf.sqrt(2*self.e) * z * pht / self.m1     
        return (topfirst/bottomfirst) * tf.exp(-exptop/expbottom) * last
    
    
class Quotient(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Quotient, self).__init__()
    
    def call(self, inputs):
        if len(inputs) != 2 or inputs[0].shape[1] != 1:
            raise Exception('must be two tensors of shape (?, 1)')
        return inputs[0]/inputs[1]


def create_nn_model(name, hidden_layers=Hidden_Layers, width=Nodes_per_HL, activation='relu'):
    inp = tf.keras.Input(shape=(1))
    #Norm_inp = tf.keras.layers.BatchNormalization()(inp)
    #initializer = tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.05,seed=None)
    initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=None)
    #initializer = tf.keras.initializers.Zeros() 0.00000000
    x = tf.keras.layers.Dense(width, activation=activation, kernel_initializer = initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(inp)
    x1 = tf.keras.layers.Dense(width, activation=activation, kernel_initializer = initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(x)
    x2 = tf.keras.layers.Dense(width, activation=activation, kernel_initializer = initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(x1)
    x3 = tf.keras.layers.Dense(width, activation=activation, kernel_initializer = initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(x2)
    x4 = tf.keras.layers.Dense(width, activation=activation, kernel_initializer = initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(x3)
    x5 = tf.keras.layers.Dense(width, activation=activation, kernel_initializer = initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(x4)
    x6 = tf.keras.layers.Dense(width, activation=activation, kernel_initializer = initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(x5)
    x7 = tf.keras.layers.Dense(width, activation=activation, kernel_initializer = initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(x6)
    nnout = tf.keras.layers.Dense(1, kernel_initializer = initializer)(x7)
    mod = tf.keras.Model(inp, nnout, name=name)
    return mod

def SIDIS_Model():
    x = tf.keras.Input(shape=(1), name='x')
    z = tf.keras.Input(shape=(1), name='z')
    phT = tf.keras.Input(shape=(1), name='phT')
    uexpr = tf.keras.Input(shape=(1), name='uexpr')
    ubarexpr = tf.keras.Input(shape=(1), name='ubarexpr')
    dexpr = tf.keras.Input(shape=(1), name='dexpr')
    dbarexpr = tf.keras.Input(shape=(1), name='dbarexpr')
    sexpr = tf.keras.Input(shape=(1), name='sexpr')
    sbarexpr = tf.keras.Input(shape=(1), name='sbarexpr')

    # nnuout = create_nn(x, 'nnu')
    # nndout = create_nn(x, 'nnd')
    # nnsout = create_nn(x, 'nns')
    # nnubarout = create_nn(x, 'nnubar')
    # nndbarout = create_nn(x, 'nndbar')
    # nnsbarout = create_nn(x, 'nnsbar')

    NNs=[]
    for i in ['nnu','nnubar','nnd','nndbar','nns','nnsbar']:
        NNs.append(create_nn_model(i)(x))


    nncomb = tf.keras.layers.Concatenate()(NNs)
    denominator = tf.keras.layers.Add()([uexpr, ubarexpr, dexpr, dbarexpr, sexpr, sbarexpr])
    exprcomb = tf.keras.layers.Concatenate()([uexpr, ubarexpr, dexpr, dbarexpr, sexpr, sbarexpr])
    numerator = tf.keras.layers.Dot(axes=1)([nncomb, exprcomb])
    quo = Quotient()([numerator, denominator])
    zpht = tf.keras.layers.Concatenate()([z, phT])
    a0 = A0()(zpht)
    siv = tf.keras.layers.Multiply()([a0, quo])
    return tf.keras.Model([x, z, phT, uexpr, ubarexpr, dexpr, dbarexpr, sexpr, sbarexpr],
                         siv)


       
class DataSIDIS(object):
    def __init__(self, pdfset='cteq61',
                 ff_PIp='NNFF10_PIp_nlo', ff_PIm='NNFF10_PIm_nlo', ff_PIsum='NNFF10_PIsum_nlo',
                 ff_KAp='NNFF10_KAp_nlo', ff_KAm='NNFF10_KAm_nlo'):
        '''
        Get data in proper format for neural network
        '''
        self.pdfData = lhapdf.mkPDF(pdfset)
        self.ffDataPIp = lhapdf.mkPDF(ff_PIp, 0)
        self.ffDataPIm = lhapdf.mkPDF(ff_PIm, 0)
        self.ffDataPIsum = lhapdf.mkPDF(ff_PIsum, 0)
        self.ffDataKAp = lhapdf.mkPDF(ff_KAp, 0)
        self.ffDataKAm = lhapdf.mkPDF(ff_KAm, 0)
        # needs to be extended to generalize for kaons
        self.eu = 2/3
        self.eubar = -2/3
        self.ed = -1/3
        self.edbar = 1/3
        self.es = -1/3
        self.esbar = 1/3

        self.ffDict = {0: self.ffDataPIp,
                       1: self.ffDataPIm,
                       2: self.ffDataPIsum,
                       3: self.ffDataKAp,
                       4: self.ffDataKAm}

    def pdf(self, flavor, x, QQ):
        return np.array([self.pdfData.xfxQ2(flavor, ax, qq) for ax, qq in zip(x, QQ)])

    
    def ff(self, func, flavor, z, QQ):
        return np.array([func.xfxQ2(flavor, az, qq) for az, qq in zip(z, QQ)])

    
    def makeData(self, df, hadrons, dependencies):

        data = {'x': [],
             'z': [],
             'phT': [],
             'uexpr': [],
             'ubarexpr': [],
             'dexpr': [],
             'dbarexpr': [],
             'sexpr': [],
             'sbarexpr': []}
        
        kins = {'hadron':[],
               'Q2': [],
               'x': [],
               'y': [],
               'z': [],
               'phT': [],
               '1D_dependence': []}

        y = []
        err = []
        
        hads = []
        QQs =[]
        yy = []
        deps = []

        df = df.loc[df['hadron'].isin(hadrons), :]
        df = df.loc[df['1D_dependence'].isin(dependencies), :]
        #X = np.array(df[['x', 'z', 'phT', 'Q2', 'hadron']])
        for i, had in enumerate(['pi+', 'pi-', 'pi0', 'k+', 'k-']):
            sliced = df.loc[df['hadron'] == had, :]
            y += list(sliced['Siv'])
            err += list(sliced['tot_err'])

            x = sliced['x']
            z = sliced['z']
            QQ = sliced['Q2']
            hads = sliced['hadron']
            QQs = sliced['Q2']
            yy = sliced['y']
            phTs = sliced['phT']
            deps = sliced['1D_dependence']
            
            data['uexpr'] += list(self.eu**2 * self.pdf(2, x, QQ) * self.ff(self.ffDict[i], 2, z, QQ))
            data['ubarexpr'] += list(self.eubar**2 * self.pdf(-2, x, QQ) * self.ff(self.ffDict[i], -2, z, QQ))
            data['dexpr'] += list(self.ed**2 * self.pdf(1, x, QQ) * self.ff(self.ffDict[i], 1, z, QQ))
            data['dbarexpr'] += list(self.edbar**2 * self.pdf(-1, x, QQ) * self.ff(self.ffDict[i], -1, z, QQ))
            data['sexpr'] += list(self.es**2 * self.pdf(3, x, QQ) * self.ff(self.ffDict[i], 3, z, QQ))
            data['sbarexpr'] += list(self.esbar**2 * self.pdf(-3, x, QQ) * self.ff(self.ffDict[i], -3, z, QQ))
            data['x'] += list(x)
            data['z'] += list(z)
            data['phT'] += list(phTs)
            
            kins['hadron']+= list(hads)
            kins['Q2']+= list(QQs)
            kins['x']+= list(x)
            kins['y']+= list(yy)
            kins['z']+= list(z)
            kins['phT']+= list(phTs)
            kins['1D_dependence']+= list(deps)


        for key in data.keys():
            data[key] = np.array(data[key])
            
        for key in kins.keys():
            kins[key] = np.array(kins[key])

        return kins, data, data[dependencies[0]], np.array(y), np.array(err)
    

SIDISdataann = DataSIDIS()

def GenSIDISReplicaData(datasetdf):
    data_dictionary = {'hadron':[],
                      'Q2': [],
                      'x': [],
                      'y': [],
                      'z': [],
                      'phT': [],
                      'Siv':[],
                      'tot_err': [],
                      '1D_dependence': []}
    temp_hads = pd.unique(datasetdf['hadron'])
    temp_deps = pd.unique(datasetdf['1D_dependence'])
    for i in temp_hads:
        for j in temp_deps:
            T_Kins, T_Xplt, T_DEP, T_yplt, T_errplt = SIDISdataann.makeData(datasetdf, [str(i)], [str(j)])
            Yhat = np.random.normal(T_yplt, uncert_frac*T_errplt)
            Yerr = T_errplt
            data_dictionary['hadron']+= list(T_Kins['hadron'])
            data_dictionary['Q2']+= list(T_Kins['Q2'])
            data_dictionary['x']+= list(T_Kins['x'])
            data_dictionary['y']+= list(T_Kins['y'])
            data_dictionary['z']+= list(T_Kins['z'])
            data_dictionary['phT']+= list(T_Kins['phT'])
            data_dictionary['1D_dependence']+= list(T_Kins['1D_dependence'])
            data_dictionary['Siv']+= list(Yhat)
            data_dictionary['tot_err']+= list(Yerr)
    return pd.DataFrame(data_dictionary)


def trn_tst(X, y, err, split=0.1):
    tstidxs = np.random.choice(list(range(len(y))), size=int(len(y)*split), replace=False)
    
    tst_X = {k: v[tstidxs] for k, v in X.items()}
    trn_X = {k: np.delete(v, tstidxs) for k, v in X.items()}
    
    tst_y = y[tstidxs]
    trn_y = np.delete(y, tstidxs)
    
    tst_err = err[tstidxs]
    trn_err = np.delete(err, tstidxs)
    
    return trn_X, tst_X, trn_y, tst_y, trn_err, tst_err

        
def run_replica(i):
    #replica_number = sys.argv[1]
    replica_number = i
    tempdf=GenSIDISReplicaData(df)
    tempdf.to_csv(str(SIDIS_Repl_Folder)+'/rep'+str(replica_number)+'.csv')
    T_Kins, T_Xplt, T_DEP, T_yplt, T_errplt = SIDISdataann.makeData(tempdf, ['pi+', 'pi-', 'pi0', 'k+', 'k-'], ['x', 'z', 'phT'])
    
    trn_X, tst_X, trn_y, tst_y, trn_err, tst_err = trn_tst(T_Xplt, T_yplt, T_errplt)

    sivModel = SIDIS_Model()
    sivModel.compile(
        optimizer = tf.keras.optimizers.Adam(Learning_Rate),
        loss = tf.keras.losses.MeanSquaredError()
        )
    #, callbacks=[modify_LR,EarlyStop]
    history = sivModel.fit(trn_X, trn_y, sample_weight=(1/trn_err**2), validation_data=(tst_X, tst_y), epochs=EPOCHS, callbacks=[modify_LR], batch_size=300, verbose=2)
    #sivModel.fit(trn_X, trn_y, sample_weight=(1/trn_err**2), validation_data=(tst_X, tst_y), epochs=EPOCHS, verbose=2)
    
    sivModel.save(save_path + str(replica_number) + '.h5', save_format='h5')
    tempdf = pd.DataFrame()
    tempdf["Train_Loss"] = history.history['loss'][-100:]
    tempdf["Val_Loss"] = history.history['val_loss'][-100:]
    tempdf.to_csv(str(Losses_Folder)+'/reploss_'+str(replica_number)+'.csv')
    plt.figure(1)
    plt.plot(history.history['loss'])
    #plt.xlim([10000,15000])
    plt.ylim([0,5])
    plt.savefig(str(Losses_Folder)+'/Plots'+'/train_loss'+str(replica_number)+'.pdf')
    plt.figure(2)
    plt.plot(history.history['val_loss'])
    #plt.xlim([10000,15000])
    #plt.ylim([0,0.005])
    plt.savefig(str(Losses_Folder)+'/Plots'+'/val_loss'+str(replica_number)+'.pdf')

for i in range(0,Nrep):
    run_replica(i)

#run_replica()