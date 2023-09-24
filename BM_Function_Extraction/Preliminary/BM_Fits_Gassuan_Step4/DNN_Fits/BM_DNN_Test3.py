import os
import tensorflow as tf
import pandas as pd
import numpy as np
import lhapdf
import matplotlib.pyplot as plt
import csv
import sys
import keras_tuner as kt

Org_Data_path = 'Pseudodata/'
save_path = 'SIDIS_Models/rep'
HERMES13p= 'HERMES13p_Pseudo.csv'
SIDIS_Repl_Folder = 'SIDIS_Replica_Data'
Losses_Folder = 'SIDIS_Replicas_Losses'
herm13p = pd.read_csv(Org_Data_path + HERMES13p).dropna(axis=0, how='all').dropna(axis=1, how='all')
df = pd.concat([herm13p])

Hidden_Layers=2
Nodes_per_HL=100
Learning_Rate = 0.005
EPOCHS = 500
L1_reg = 10**(-12)
modify_LR = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',factor=0.9,patience=400,mode='auto')
EarlyStop = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=200)
uncert_frac = 1


class Quotient(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Quotient, self).__init__()
    
    def call(self, inputs):
        if len(inputs) != 2 or inputs[0].shape[1] != 1:
            raise Exception('must be two tensors of shape (?, 1)')
        return inputs[0]/inputs[1]
    
    
def chisquare(y, yhat, err):
    return np.sum(((y - yhat)/err)**2)

def pperp2avgVal(a,b,z):
    return a + b*(z**2)

def kBM2Avg(m1,kperp2Avg):
    temp=((m1**2)*kperp2Avg)/((m1**2)+kperp2Avg)
    return temp

def pc2Avg(pperp2Avg,mc):
    temp = ((mc**2)*pperp2Avg)/((mc**2)+pperp2Avg)
    return temp

def phT2Avg(pperp2Avg,kperp2Avg,z):
    temp = pperp2Avg + (kperp2Avg)*z**2
    return temp

def pht2avgBM(pperp2Avg,mc,z,m1,kperp2Avg):
    temp = pc2Avg(pperp2Avg,mc) + (z**2)*kBM2Avg(m1,kperp2Avg)
    return temp


def NCq(had,flavor,z):
    gamma=1.06
    delta=0.07
    temp_zfactor=(z**gamma)*((1-z)**(delta))*((gamma+delta)**(gamma+delta))/((gamma**gamma)*(delta**delta))
    if((str(had)=="pi+")&(flavor==2)):
        MCv = 0.49
    elif((str(had)=="pi+")&(flavor==-1)):
        MCv = 0.49
    elif((str(had)=="pi-")&(flavor==1)):
        MCv = 0.49
    elif((str(had)=="pi-")&(flavor==-2)):
        MCv = 0.49
    else:
        MCv = -1
    tempNCq=MCv*temp_zfactor
    return tempNCq



def pp2avg(z):
    return 0.2 + 0.5*(z**2)


def A0_cosphi_BM(y,z,pht,m1,mc,QQ,kperp2Avg,pperp2Avg,eCharg):
    temp1 = (2*(2-y)*(tf.sqrt(1-y)))/(1+(1-y)**2)
    temp2 = (2*eCharg*pht)/(m1*mc*tf.sqrt(QQ))
    temp3 = (pperp2Avg)/(pht2avgBM(pperp2Avg,mc,z,m1,kperp2Avg)**4)
    temp4 = tf.exp(pht**2/pperp2Avg - pht**2/pht2avgBM(pperp2Avg,mc,z,m1,kperp2Avg))
    temp5 = ((kBM2Avg(m1,kperp2Avg)**2)*(pc2Avg(pperp2Avg,mc)**2))/(kperp2Avg*pperp2Avg)
    temp6 = (z**2)*kBM2Avg(m1,kperp2Avg)*(pht**2 - pht2avgBM(pperp2Avg,mc,z,m1,kperp2Avg))+ pc2Avg(pperp2Avg,mc)*pht2avgBM(pperp2Avg,mc,z,m1,kperp2Avg)
    tempfinal = temp1*temp2*temp3*temp4*temp5*temp6
    return tempfinal


def A0_cos2phi_BM(y,z,pht,m1,mc,QQ,kperp2Avg,pperp2Avg,eCharg):
    temp1 = (2*(2-y))/(1+(1-y)**2)
    temp2 = (-eCharg*(pht**2))/(m1*mc)
    temp3 = (pperp2Avg)/(pht2avgBM(pperp2Avg,mc,z,m1,kperp2Avg)**3)
    temp4 = tf.exp(pht**2/pperp2Avg - pht**2/pht2avgBM(pperp2Avg,mc,z,m1,kperp2Avg))
    temp5 = ((kBM2Avg(m1,kperp2Avg)**2)*(pc2Avg(pperp2Avg,mc)**2))/(kperp2Avg*pperp2Avg)
    temp6 = (z**2)*kBM2Avg(m1,kperp2Avg)*(pht**2 - pht2avgBM(pperp2Avg,mc,z,m1,kperp2Avg))+ pc2Avg(pperp2Avg,mc)*pht2avgBM(pperp2Avg,mc,z,m1,kperp2Avg)
    tempfinal = temp1*temp2*temp3*temp4*temp5*temp6
    return tempfinal


def A0_cosphi_Cahn(y,z,pht,QQ,kperp2Avg,pperp2Avg,eCharg):
    temp1 = (2*(2-y)*(tf.sqrt(1-y)))/(1+(1-y)**2)
    temp2 = (-2*eCharg*pht)/(tf.sqrt(QQ))
    temp3 = (z*kperp2Avg)/(phT2Avg(pperp2Avg,kperp2Avg,z))
    tempfinal = temp1*temp2*temp3
    return tempfinal


def A0_cos2phi_Cahn(y,z,pht,QQ,kperp2Avg,pperp2Avg,eCharg):
    temp1 = (2*(2-y))/(1+(1-y)**2)
    temp2 = (2*eCharg*pht**pht)/(QQ)
    temp3 = (z*z*kperp2Avg*kperp2Avg)/(phT2Avg(pperp2Avg,kperp2Avg,z)**2)
    tempfinal = temp1*temp2*(temp3**2)
    return tempfinal

def A0_BM(y,z,QQ,pht,m1,mc,kperp2Avg,pperp2Avg,eCharg):
    temp = A0_cosphi_BM(y,z,pht,m1,mc,QQ,kperp2Avg,pperp2Avg,eCharg) - tf.sqrt(QQ)*A0_cos2phi_BM(y,z,pht,m1,mc,QQ,kperp2Avg,pperp2Avg,eCharg)
    return temp


def A0_Cahn(y,z,QQ,pht,m1,mc,kperp2Avg,pperp2Avg,eCharg):
    temp = A0_cosphi_Cahn(y,z,pht,QQ,kperp2Avg,pperp2Avg,eCharg) - tf.sqrt(QQ)*A0_cos2phi_Cahn(y,z,pht,QQ,kperp2Avg,pperp2Avg,eCharg)
    return temp
    
    
def create_nn_model(name, hidden_layers=Hidden_Layers, activation='relu'):
    inp = tf.keras.Input(shape=(1))
    #width = hp.Int('units', min_value=32, max_value=512, step=32)
    width = 64
    #Norm_inp = tf.keras.layers.BatchNormalization()(inp)
    #initializer = tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.05,seed=None)
    initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=None)
    #initializer = tf.keras.initializers.Zeros() 0.00000000
    x = tf.keras.layers.Dense(width, activation=activation, kernel_initializer = initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(inp)
    x1 = tf.keras.layers.Dense(width, activation=activation, kernel_initializer = initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(x)
#     x2 = tf.keras.layers.Dense(width, activation=activation, kernel_initializer = initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(x1)
#     x3 = tf.keras.layers.Dense(width, activation=activation, kernel_initializer = initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(x2)
#     x4 = tf.keras.layers.Dense(width, activation=activation, kernel_initializer = initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(x3)
#     x5 = tf.keras.layers.Dense(width, activation=activation, kernel_initializer = initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(x4)
    # x6 = tf.keras.layers.Dense(width, activation=activation, kernel_initializer = initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(x5)
    # x7 = tf.keras.layers.Dense(width, activation=activation, kernel_initializer = initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(x6)
    nnout = tf.keras.layers.Dense(1, kernel_initializer = initializer)(x1)
    mod = tf.keras.Model(inp, nnout, name=name)
    return mod



def SIDIS_Model():
    x = tf.keras.Input(shape=(1), name='x')
    z = tf.keras.Input(shape=(1), name='z')
    y = tf.keras.Input(shape=(1), name='y')
    QQ = tf.keras.Input(shape=(1), name='Q2')
    phT = tf.keras.Input(shape=(1), name='phT')
    uexpr = tf.keras.Input(shape=(1), name='uexpr')
    ubarexpr = tf.keras.Input(shape=(1), name='ubarexpr')
    dexpr = tf.keras.Input(shape=(1), name='dexpr')
    dbarexpr = tf.keras.Input(shape=(1), name='dbarexpr')
    sexpr = tf.keras.Input(shape=(1), name='sexpr')
    sbarexpr = tf.keras.Input(shape=(1), name='sbarexpr')
    
#     kperp2Avg=0.03
#     pperp2Avg=0.12
#     eCharg = 1
#     m1 = 0.1
#     mc = 0.5
    kperp2Avg=0.03
    pperp2Avg = pp2avg(z)
    eCharg = 1
    m1 = 0.3
    mc = 1.22

    NNs=[]
    for i in ['nnu','nnubar','nnd','nndbar','nns','nnsbar']:
        NNs.append(create_nn_model(i)(x))

    nncomb = tf.keras.layers.Concatenate()(NNs)
    denominator = tf.keras.layers.Add()([uexpr, ubarexpr, dexpr, dbarexpr, sexpr, sbarexpr])
    exprcomb = tf.keras.layers.Concatenate()([uexpr, ubarexpr, dexpr, dbarexpr, sexpr, sbarexpr])
    numerator = tf.keras.layers.Dot(axes=1)([nncomb, exprcomb])
    quo = Quotient()([numerator, denominator])
    #zpht = tf.keras.layers.Concatenate()([z, phT])
    a0 = A0_BM(y,z,QQ,phT,m1,mc,kperp2Avg,pperp2Avg,eCharg)
    a0_C = A0_Cahn(y,z,QQ,phT,m1,mc,kperp2Avg,pperp2Avg,eCharg)
    temp_BM = tf.keras.layers.Multiply()([a0, quo])
    temp_Asym = tf.keras.layers.Add()([temp_BM,a0_C])
    return tf.keras.Model([x, y, z, phT, QQ, uexpr, ubarexpr, dexpr, dbarexpr, sexpr, sbarexpr],
                         temp_Asym)


class DataSIDIS(object):
    # def __init__(self, pdfset='cteq61',
    #              ff_PIp='NNFF10_PIp_nlo', ff_PIm='NNFF10_PIm_nlo', ff_PIsum='NNFF10_PIsum_nlo',
    #              ff_KAp='NNFF10_KAp_nlo', ff_KAm='NNFF10_KAm_nlo'):
    def __init__(self, pdfset='cteq61',
                 ff_PIp='DSS14_NLO_Pip', ff_PIm='DSS14_NLO_Pim', ff_PIsum='DSS14_NLO_PiSum',
                 ff_KAp='DSS17_NLO_KaonPlus', ff_KAm='DSS17_NLO_KaonMinus'):
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
        
        self.gamma=1.06
        self.delta=0.07

        self.ffDict = {0: self.ffDataPIp,
                       1: self.ffDataPIm,
                       2: self.ffDataPIsum,
                       3: self.ffDataKAp,
                       4: self.ffDataKAm}

    def pdf(self, flavor, x, QQ):
        return np.array([self.pdfData.xfxQ2(flavor, ax, qq) for ax, qq in zip(x, QQ)])

    
    def ff(self, func, flavor, z, QQ):
        return np.array([func.xfxQ2(flavor, az, qq) for az, qq in zip(z, QQ)])
    
    
    def NCqv(self,had,flavor,z):
        temp_zfactor=(z**self.gamma)*((1-z)**(self.delta))*((self.gamma+self.delta)**(self.gamma+self.delta))/((self.gamma**self.gamma)*(self.delta**self.delta))
        if((str(had)=="pi+")&(flavor==2)):
            MCv = 0.49
        elif((str(had)=="pi+")&(flavor==-1)):
            MCv = 0.49
        elif((str(had)=="pi-")&(flavor==1)):
            MCv = 0.49
        elif((str(had)=="pi-")&(flavor==-2)):
            MCv = 0.49
        else:
            MCv = -1
        tempNCq=MCv*temp_zfactor
        return np.array(tempNCq)

    
    def makeData(self, df, hadrons, dependencies):

        data = {'x': [],
             'y': [],
             'z': [],
             'phT': [],
             'Q2': [],
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

        asym = []
        err = []
        
        hads = []
        QQs =[]
        y = []
        deps = []

        df = df.loc[df['hadron'].isin(hadrons), :]
        df = df.loc[df['1D_dependence'].isin(dependencies), :]
        #X = np.array(df[['x', 'z', 'phT', 'Q2', 'hadron']])
        for i, had in enumerate(['pi+', 'pi-', 'pi0', 'k+', 'k-']):
            sliced = df.loc[df['hadron'] == had, :]
            asym += list(sliced['Asym'])
            err += list(sliced['dAsym'])

            x = sliced['x']
            y = sliced['y']
            z = sliced['z']
            QQ = sliced['Q2']
            hads = sliced['hadron']
            QQs = sliced['Q2']
            phTs = sliced['phT']
            deps = sliced['1D_dependence']
            
            data['uexpr'] += list(self.eu**2 * self.pdf(2, x, QQ) * self.ff(self.ffDict[i], 2, z, QQ)*self.NCqv(i,2,z))
            data['ubarexpr'] += list(self.eubar**2 * self.pdf(-2, x, QQ) * self.ff(self.ffDict[i], -2, z, QQ)*self.NCqv(i,-2,z))
            data['dexpr'] += list(self.ed**2 * self.pdf(1, x, QQ) * self.ff(self.ffDict[i], 1, z, QQ)*self.NCqv(i,1,z))
            data['dbarexpr'] += list(self.edbar**2 * self.pdf(-1, x, QQ) * self.ff(self.ffDict[i], -1, z, QQ)*self.NCqv(i,-1,z))
            data['sexpr'] += list(self.es**2 * self.pdf(3, x, QQ) * self.ff(self.ffDict[i], 3, z, QQ)*self.NCqv(i,3,z))
            data['sbarexpr'] += list(self.esbar**2 * self.pdf(-3, x, QQ) * self.ff(self.ffDict[i], -3, z, QQ)*self.NCqv(i,-3,z))
            data['x'] += list(x)
            data['y'] += list(y)
            data['z'] += list(z)
            data['phT'] += list(phTs)
            data['Q2'] += list(QQ)
            
            kins['hadron']+= list(hads)
            kins['Q2']+= list(QQs)
            kins['x']+= list(x)
            kins['y']+= list(y)
            kins['z']+= list(z)
            kins['phT']+= list(phTs)
            kins['1D_dependence']+= list(deps)


        for key in data.keys():
            data[key] = np.array(data[key])
            
        for key in kins.keys():
            kins[key] = np.array(kins[key])

        return kins, data, data[dependencies[0]], np.array(asym), np.array(err)
    
    
    

def GenSIDISReplicaData(datasetdf):
    data_dictionary = {'hadron':[],
                      'Q2': [],
                      'x': [],
                      'y': [],
                      'z': [],
                      'phT': [],
                      'Asym':[],
                      'dAsym': [],
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
            data_dictionary['Asym']+= list(Yhat)
            data_dictionary['dAsym']+= list(Yerr)
    return pd.DataFrame(data_dictionary)


SIDISdataann = DataSIDIS()

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
    tempdf.to_csv('./SIDIS_Replica_Data/rep'+str(replica_number)+'.csv')
    T_Kins, T_Xplt, T_DEP, T_yplt, T_errplt = SIDISdataann.makeData(tempdf, ['pi+', 'pi-', 'pi0', 'k+', 'k-'], ['x', 'y', 'z', 'phT'])
    
#     print(T_Xplt)
    trn_X, tst_X, trn_y, tst_y, trn_err, tst_err = trn_tst(T_Xplt, T_yplt, T_errplt)

    sivModel = SIDIS_Model()
    sivModel.compile(
        optimizer = tf.keras.optimizers.Adam(Learning_Rate),
        loss = tf.keras.losses.MeanSquaredError()
        )

    history = sivModel.fit(trn_X, trn_y, validation_data=(tst_X, tst_y), epochs=EPOCHS, callbacks=[modify_LR], batch_size=64, verbose=2)
    
    #sivModel.save(str(replica_number) + '.h5', save_format='h5')
    sivModel.save(save_path + str(replica_number) + '.h5', save_format='h5')
    tempdf = pd.DataFrame()
    tempdf["Train_Loss"] = history.history['loss']
    #[-100:]
    tempdf["Val_Loss"] = history.history['val_loss']
    #tempdf.to_csv('reploss_'+str(replica_number)+'.csv')
    tempdf.to_csv(str(Losses_Folder)+'/reploss_'+str(replica_number)+'.csv')
    plt.figure(1)
    plt.plot(history.history['loss'])
    #plt.savefig('train_loss'+str(replica_number)+'.pdf')
    plt.savefig(str(Losses_Folder)+'/Plots'+'/train_loss'+str(replica_number)+'.pdf')
    plt.figure(2)
    plt.plot(history.history['val_loss'])
    #plt.savefig('val_loss'+str(replica_number)+'.pdf')
    plt.savefig(str(Losses_Folder)+'/Plots'+'/val_loss'+str(replica_number)+'.pdf')

for i in range(0,100):    
    run_replica(i)