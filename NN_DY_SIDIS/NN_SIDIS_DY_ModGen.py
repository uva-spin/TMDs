#######################################################################
############ Generating Replicas for NN model : SIDIS #################
############ Written by Ishara Fernando & Nick Newton #################
############ Last upgrade: Oct-19-2021 ################################
#######################################################################


import tensorflow as tf
import pandas as pd
import numpy as np
import lhapdf
import matplotlib.pyplot as plt

import functions

#################################################################
##################### Hyperparameters ###########################
#################################################################

Hidden_Layers=2
Nodes_per_HL=256
Learning_Rate = 0.0001
EPOCHS = 5000

############################### Data Files ##################################

### SIDIS ###

#herm9_SIDIS = pd.read_csv('./Data/HERMES_p_2009.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
herm20_SIDIS = pd.read_csv('./Data/HERMES_p_2020.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
comp9_SIDIS = pd.read_csv('./Data/COMPASS_d_2009.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
comp15_SIDIS = pd.read_csv('./Data/COMPASS_p_2015.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')

### DY ###

comp15_DY = pd.read_csv('./Data/COMPASS_p_DY_2017.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')

#df_SIDIS = pd.concat([herm9_SIDIS, herm20_SIDIS, comp9_SIDIS, comp15_SIDIS])
df_SIDIS = pd.concat([herm20_SIDIS, comp9_SIDIS, comp15_SIDIS])

df_DY= pd.concat([comp15_DY])


def chisquare(y, yhat, err):
    return np.sum(((y - yhat)/err)**2)

### This is for SIDIS ###
class A0(tf.keras.layers.Layer):
    def __init__(self, kperp2avg=.57, pperp2avg=.12, **kwargs):
        super(A0, self).__init__(name='a0')
        self.m1 = tf.Variable(1., name='m1')
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
        ks2avg = (self.kperp2avg*self.m1**2)/(self.m1**2 + self.kperp2avg) #correct 
        topfirst = (z**2 * self.kperp2avg + self.pperp2avg) * ks2avg**2 #correct
        bottomfirst = (z**2 * ks2avg + self.pperp2avg)**2 * self.kperp2avg #correct
        exptop = pht**2 * z**2 * (ks2avg - self.kperp2avg) #correct
        expbottom = (z**2 * ks2avg + self.pperp2avg) * (z**2 * self.kperp2avg + self.pperp2avg) #correct
        last = tf.sqrt(2*self.e) * z * pht / self.m1 #correct      
        return (topfirst/bottomfirst) * tf.exp(-exptop/expbottom) * last
    

### This is for DY ###
class B0(tf.keras.layers.Layer):
    def __init__(self, kperp2avg=.57, pperp2avg=.12, **kwargs):
        super(B0, self).__init__(name='b0')
        self.m1 = tf.Variable(1., name='m1')
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
        #z = inputs[:, 0]
        #pht = inputs[:, 1]
        qT = inputs[:, 1]
        ks2avg = (self.kperp2avg*self.m1**2)/(self.m1**2 + self.kperp2avg) 
        topfirst = (self.kperp2avg + self.kperp2avg) * ks2avg**2 
        bottomfirst = (ks2avg + self.kperp2avg)**2 * self.kperp2avg 
        exptop = -qT**2 / (ks2avg + self.kperp2avg) 
        expbottom = -qT**2 / (self.kperp2avg + self.kperp2avg) 
        last = tf.sqrt(2*self.e) * qT / self.m1       
        return (topfirst/bottomfirst) * tf.exp(-exptop/expbottom) * last


### The following quotient can be used in both SIDIS & DY (check) ###    
class Quotient(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Quotient, self).__init__()
    
    def call(self, inputs):
        if len(inputs) != 2 or inputs[0].shape[1] != 1:
            raise Exception('must be two tensors of shape (?, 1)')
        return inputs[0]/inputs[1]


### Here we create models for each quark-flavor (only input is x) ###    
def create_nn(inp, name, hidden_layers=Hidden_Layers, width=Nodes_per_HL, activation='relu'):
    x = tf.keras.layers.Dense(width, activation=activation)(inp)
    for i in range(hidden_layers-1):
        x = tf.keras.layers.Dense(width, activation=activation)(x)
    nnout = tf.keras.layers.Dense(1, name=name)(x)
    return nnout


def createModel_SIDIS():
    x = tf.keras.Input(shape=(1), name='x')
    z = tf.keras.Input(shape=(1), name='z')
    phT = tf.keras.Input(shape=(1), name='phT')
    uexpr = tf.keras.Input(shape=(1), name='uexpr')
    ubarexpr = tf.keras.Input(shape=(1), name='ubarexpr')
    dexpr = tf.keras.Input(shape=(1), name='dexpr')
    dbarexpr = tf.keras.Input(shape=(1), name='dbarexpr')
    sexpr = tf.keras.Input(shape=(1), name='sexpr')
    sbarexpr = tf.keras.Input(shape=(1), name='sbarexpr')

    nnuout = create_nn(x, 'nnu')
    nndout = create_nn(x, 'nnd')
    nnsout = create_nn(x, 'nns')
    nnubarout = create_nn(x, 'nnubar')
    nndbarout = create_nn(x, 'nndbar')
    nnsbarout = create_nn(x, 'nnsbar')

    nncomb = tf.keras.layers.Concatenate()([nnuout, nnubarout, nndout, nndbarout, nnsout, nnsbarout])
    denominator = tf.keras.layers.Add()([uexpr, ubarexpr, dexpr, dbarexpr, sexpr, sbarexpr])
    exprcomb = tf.keras.layers.Concatenate()([uexpr, ubarexpr, dexpr, dbarexpr, sexpr, sbarexpr])
    numerator = tf.keras.layers.Dot(axes=1)([nncomb, exprcomb])
    quo = Quotient()([numerator, denominator])
    zpht = tf.keras.layers.Concatenate()([z, phT])
    a0 = A0()(zpht)
    siv = tf.keras.layers.Multiply()([a0, quo])
    return tf.keras.Model([x, z, phT, uexpr, ubarexpr, dexpr, dbarexpr, sexpr, sbarexpr],
                         siv)



def createModel_DY():
    x1 = tf.keras.Input(shape=(1), name='x1')
    x2 = tf.keras.Input(shape=(1), name='x2')
    qT = tf.keras.Input(shape=(1), name='qT')
    # the following expressions are the ones multiply by Nq
    Uexpr_x1 = tf.keras.Input(shape=(1), name='Uexpr_x1')
    Uexpr_x2 = tf.keras.Input(shape=(1), name='Uexpr_x2')
    Ubarexpr_x1 = tf.keras.Input(shape=(1), name='Ubarexpr_x1')
    Ubarexpr_x2 = tf.keras.Input(shape=(1), name='Ubarexpr_x2')
    Dexpr_x1 = tf.keras.Input(shape=(1), name='Dexpr_x1')
    Dexpr_x2 = tf.keras.Input(shape=(1), name='Dexpr_x2')
    Dbarexpr_x1 = tf.keras.Input(shape=(1), name='Dbarexpr_x1')
    Dbarexpr_x2 = tf.keras.Input(shape=(1), name='Dbarexpr_x2')
    Sexpr_x1 = tf.keras.Input(shape=(1), name='Sexpr_x1')
    Sexpr_x2 = tf.keras.Input(shape=(1), name='Sexpr_x2')
    Sbarexpr_x1 = tf.keras.Input(shape=(1), name='Sbarexpr_x1')
    Sbarexpr_x2 = tf.keras.Input(shape=(1), name='Sbarexpr_x2')
    #Dexpr = tf.keras.Input(shape=(1), name='Dexpr')
    #Sexpr = tf.keras.Input(shape=(1), name='Sexpr')
  
    nnu_x1 = create_nn(x1, 'nnu')
    nnu_x2 = create_nn(x2, 'nnu')
    nnubar_x1 = create_nn(x1, 'nnubar')
    nnubar_x2 = create_nn(x2, 'nnubar')
    nnd_x1 = create_nn(x1, 'nnd')
    nnd_x2 = create_nn(x2, 'nnd')
    nndbar_x1 = create_nn(x1, 'nndbar')
    nndbar_x2 = create_nn(x2, 'nndbar')
    nns_x1 = create_nn(x1, 'nns')
    nns_x2 = create_nn(x2, 'nns')
    nnsbar_x1 = create_nn(x1, 'nnsbar')
    nnsbar_x2 = create_nn(x2, 'nnsbar')

    nncomb = tf.keras.layers.Concatenate()([nnu_x1, nnu_x2, nnd_x1, nnd_x2, nns_x1, nns_x2, 
             nnubar_x1, nnubar_x2, nndbar_x1, nndbar_x2, nnsbar_x1, nnsbar_x2])
    denominator = tf.keras.layers.Add()([Uexpr_x1, Uexpr_x2, Ubarexpr_x1, Ubarexpr_x2, Dexpr_x1, Dexpr_x2, Dbarexpr_x1, Dbarexpr_x2, Sexpr_x1, Sexpr_x2, Sbarexpr_x1, Sbarexpr_x2])
    exprcomb = tf.keras.layers.Concatenate()([Uexpr_x1, Uexpr_x2, Ubarexpr_x1, Ubarexpr_x2, Dexpr_x1, Dexpr_x2, Dbarexpr_x1, Dbarexpr_x2, Sexpr_x1, Sexpr_x2, Sbarexpr_x1, Sbarexpr_x2])
    numerator = tf.keras.layers.Dot(axes=1)([nncomb, exprcomb])
    quo = Quotient()([numerator, denominator])
    #zpht = tf.keras.layers.Concatenate()([z, phT])
    b0 = B0()(qT)
    siv = tf.keras.layers.Multiply()([b0, quo])
    return tf.keras.Model([x1, x2, qT, Uexpr_x1, Uexpr_x2, Ubarexpr_x1, Ubarexpr_x2, Dexpr_x1, Dexpr_x2, Dbarexpr_x1, Dbarexpr_x2, Sexpr_x1, Sexpr_x2, Sbarexpr_x1, Sbarexpr_x2],
                         siv)


sivModel_SIDIS = createModel_SIDIS()
sivModel_DY = createModel_DY()

inputdict_SIDIS = {'x': tf.constant([1., 2.]),
             'z': tf.constant([1., 2.]),
             'phT': tf.constant([1., 2.]),
             'uexpr': tf.constant([1., 2.]),
             'ubarexpr': tf.constant([1., 2.]),
             'dexpr': tf.constant([1., 2.]),
             'dbarexpr': tf.constant([1., 2.]),
             'sexpr': tf.constant([1., 2.]),
             'sbarexpr': tf.constant([1., 2.])}


inputdict_DY = {'x1': tf.constant([1., 2.]),
             'x2': tf.constant([1., 2.]),
             'qT': tf.constant([1., 2.]),
             'Uexpr_x1': tf.constant([1., 2.]),
             'Uexpr_x2': tf.constant([1., 2.]),
             'Ubarexpr_x1': tf.constant([1., 2.]),
             'Ubarexpr_x2': tf.constant([1., 2.]),
             'Dexpr_x1': tf.constant([1., 2.]),
             'Dexpr_x2': tf.constant([1., 2.]),
             'Dbarexpr_x1': tf.constant([1., 2.]),
             'Dbarexpr_x2': tf.constant([1., 2.]),
             'Sexpr_x1': tf.constant([1., 2.]),
             'Sexpr_x2': tf.constant([1., 2.]),
             'Sbarexpr_x1': tf.constant([1., 2.]),
             'Sbarexpr_x2': tf.constant([1., 2.])}


#inputdict_DY = {'x1': tf.constant([1., 2.]),
#             'x2': tf.constant([1., 2.]),
#             'qT': tf.constant([1., 2.]),
#             'Uexpr': tf.constant([1., 2.]),
#             'Dexpr': tf.constant([1., 2.]),
#             'Sexpr': tf.constant([1., 2.])}


sivModel_SIDIS(inputdict_SIDIS)
sivModel_DY(inputdict_DY)

############################################## SIDIS data ###########################################################

class SIDIS_DataANN(object):
    def __init__(self, pdf_SIDISset='cteq61',
                 ff_PIp='NNFF10_PIp_nlo', ff_PIm='NNFF10_PIm_nlo', ff_PIsum='NNFF10_PIsum_nlo',
                 ff_KAp='NNFF10_KAp_nlo', ff_KAm='NNFF10_KAm_nlo'):
        '''
        Get data in proper format for neural network
        '''
        self.pdf_SIDISData = lhapdf.mkPdf(pdf_SIDISset)
        self.ffDataPIp = lhapdf.mkPdf(ff_PIp, 0)
        self.ffDataPIm = lhapdf.mkPdf(ff_PIm, 0)
        self.ffDataPIsum = lhapdf.mkPdf(ff_PIsum, 0)
        self.ffDataKAp = lhapdf.mkPdf(ff_KAp, 0)
        self.ffDataKAm = lhapdf.mkPdf(ff_KAm, 0)
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
    

    def pdf_SIDIS(self, flavor, x, QQ):
        return np.array([self.pdf_SIDISData.xfxQ2(flavor, ax, qq) for ax, qq in zip(x, QQ)])
    
    
    def ff(self, func, flavor, z, QQ):
        return np.array([func.xfxQ2(flavor, az, qq) for az, qq in zip(z, QQ)])    
    

    def makeData(self, df_SIDIS, hadrons, dependencies):
        
        data = {'x': [],
             'z': [],
             'phT': [],
             'uexpr': [],
             'ubarexpr': [],
             'dexpr': [],
             'dbarexpr': [],
             'sexpr': [],
             'sbarexpr': []}
        
        y = []
        err = []
        
        df_SIDIS = df_SIDIS.loc[df_SIDIS['hadron'].isin(hadrons), :]
        df_SIDIS = df_SIDIS.loc[df_SIDIS['1D_dependence'].isin(dependencies), :]
        #X = np.array(df_SIDIS[['x', 'z', 'phT', 'Q2', 'hadron']])
        for i, had in enumerate(['pi+', 'pi-', 'pi0', 'k+', 'k-']):
            sliced = df_SIDIS.loc[df_SIDIS['hadron'] == had, :]
            y += list(sliced['Siv'])
            err += list(sliced['tot_err'])
            
            x = sliced['x']
            z = sliced['z']
            QQ = sliced['Q2']
            data['uexpr'] += list(self.eu**2 * self.pdf_SIDIS(2, x, QQ) * self.ff(self.ffDict[i], 2, z, QQ))
            data['ubarexpr'] += list(self.eubar**2 * self.pdf_SIDIS(-2, x, QQ) * self.ff(self.ffDict[i], -2, z, QQ))
            data['dexpr'] += list(self.ed**2 * self.pdf_SIDIS(1, x, QQ) * self.ff(self.ffDict[i], 1, z, QQ))
            data['dbarexpr'] += list(self.edbar**2 * self.pdf_SIDIS(-1, x, QQ) * self.ff(self.ffDict[i], -1, z, QQ))
            data['sexpr'] += list(self.es**2 * self.pdf_SIDIS(3, x, QQ) * self.ff(self.ffDict[i], 3, z, QQ))
            data['sbarexpr'] += list(self.esbar**2 * self.pdf_SIDIS(-3, x, QQ) * self.ff(self.ffDict[i], -3, z, QQ))

            data['x'] += list(x)
            data['z'] += list(z)
            data['phT'] += list(sliced['phT'])
        
        for key in data.keys():
            data[key] = np.array(data[key])
        
        print(data)
        return data, np.array(y), np.array(err)                      


############################################## DY data ###########################################################

class DY_DataANN(object):
    def __init__(self, pdf_set='cteq61'):
        '''
        Get data in proper format for neural network
        '''
        self.pdf_DYData = lhapdf.mkPdf(pdf_set)

        self.eu = 2/3
        self.eubar = -2/3
        self.ed = -1/3
        self.edbar = 1/3
        self.es = -1/3
        self.esbar = 1/3
        
        #self.ffDict = {0: self.ffDataPIp,
        #               1: self.ffDataPIm,
        #               2: self.ffDataPIsum,
        #               3: self.ffDataKAp,
        #               4: self.ffDataKAm}
    

    def pdf_DY(self, flavor, x, QM):
        return np.array([self.pdf_DYData.xfxQ(flavor, ax, qq) for ax, qq in zip(x, QM)])
    
 
    def makeData(self, df_DY, dependencies):
        
        data = {'x1': [],
             'x2': [],
             'qT': [],
             'Uexpr_x1': [],
             'Uexpr_x2': [],
             'Ubarexpr_x1': [],
             'Ubarexpr_x2': [],
             'Dexpr_x1': [],
             'Dexpr_x2': [],
             'Dbarexpr_x1': [],
             'Dbarexpr_x2': [],
             'Sexpr_x1': [],
             'Sexpr_x2': [],
             'Sbarexpr_x1': [],
             'Sbarexpr_x2': []}
        
        y_DY = []
        err_DY = []
        
        df_DY = df_DY.loc[df_DY['Dependence'].isin(dependencies), :]
        for i, dep in enumerate(['x1', 'x2', 'xF', 'QT', 'QM']):
            sliced = df_DY.loc[df_DY['Dependence'] == dep, :]
            y_DY += list(sliced['Siv'])
            err_DY += list(sliced['tot_err'])
            
            x1 = sliced['x1']
            x2 = sliced['x2']
            QQ = sliced['QM']
            data['Uexpr_x1'] += list(self.eu**2 * self.pdf_DY(2, x1, QQ) * self.pdf_DY(-2, x2, QQ))
            data['Uexpr_x2'] += list(self.eu**2 * self.pdf_DY(2, x2, QQ) * self.pdf_DY(-2, x1, QQ))
            data['Ubarexpr_x1'] += list(self.eubar**2 * self.pdf_DY(-2, x1, QQ) * self.pdf_DY(2, x2, QQ))
            data['Ubarexpr_x2'] += list(self.eubar**2 * self.pdf_DY(-2, x2, QQ) * self.pdf_DY(2, x1, QQ))
            data['Dexpr_x1'] += list(self.ed**2 * self.pdf_DY(1, x1, QQ) * self.pdf_DY(-1, x2, QQ))
            data['Dexpr_x2'] += list(self.ed**2 * self.pdf_DY(1, x2, QQ) * self.pdf_DY(-1, x1, QQ))
            data['Dbarexpr_x1'] += list(self.edbar**2 * self.pdf_DY(-1, x1, QQ) * self.pdf_DY(1, x2, QQ))
            data['Dbarexpr_x2'] += list(self.edbar**2 * self.pdf_DY(-1, x2, QQ) * self.pdf_DY(1, x1, QQ))
            data['Sexpr_x1'] += list(self.es**2 * self.pdf_DY(3, x1, QQ) * self.pdf_DY(-3, x2, QQ))
            data['Sexpr_x2'] += list(self.es**2 * self.pdf_DY(3, x2, QQ) * self.pdf_DY(-3, x1, QQ))
            data['Sbarexpr_x1'] += list(self.esbar**2 * self.pdf_DY(-3, x1, QQ) * self.pdf_DY(3, x2, QQ))
            data['Sbarexpr_x2'] += list(self.esbar**2 * self.pdf_DY(-3, x2, QQ) * self.pdf_DY(3, x1, QQ))

            #data['Dexpr'] += list((self.ed**2 * self.pdf_DY(1, x1, QQ) * self.pdf_DY(-1, x2, QQ))
            #                    + self.edbar**2 * self.pdf_DY(-1, x1, QQ) * self.pdf_DY(1, x2, QQ))
            #data['Sexpr'] += list((self.es**2 * self.pdf_DY(3, x1, QQ) * self.pdf_DY(-3, x2, QQ))
            #                    + self.esbar**2 * self.pdf_DY(-3, x1, QQ) * self.pdf_DY(3, x2, QQ))

            data['x1'] += list(x1)
            data['x2'] += list(x2)
            data['qT'] += list(sliced['QT'])
        
        for key in data.keys():
            data[key] = np.array(data[key])
        
        print(data)
        return data, np.array(y_DY), np.array(err_DY)                      

 #################################################################################################################

        
SIDIS_datann = SIDIS_DataANN()
X_SIDIS, y_SIDIS, err_SIDIS = SIDIS_datann.makeData(df_SIDIS, ['pi+', 'pi-', 'pi0', 'k+', 'k-'], ['x', 'z', 'phT'])

def trn_tst(X, y, err, split=0.1):
    tstidxs = np.random.choice(list(range(len(y))), size=int(len(y)*split), replace=False)
    
    tst_X = {k: v[tstidxs] for k, v in X.items()}
    trn_X = {k: np.delete(v, tstidxs) for k, v in X.items()}
    
    tst_y = y[tstidxs]
    trn_y = np.delete(y, tstidxs)
    
    tst_err = err[tstidxs]
    trn_err = np.delete(err, tstidxs)
    
    return trn_X, tst_X, trn_y, tst_y, trn_err, tst_err


trn_X_SIDIS, tst_X_SIDIS, trn_y_SIDIS, tst_y_SIDIS, trn_err_SIDIS, tst_err_SIDIS = trn_tst(X_SIDIS, y_SIDIS, err_SIDIS)


#sivModel.compile(
#    optimizer = tf.keras.optimizers.Adam(Learning_Rate),
#    loss = tf.keras.losses.MeanSquaredError()
#    )


def trainReplicas(X, y, err, numReplicas):
    for i in range(numReplicas):
        yrep = np.random.normal(y, err)
        
        sivModel_SIDIS = createModel_SIDIS()
        
        sivModel_SIDIS.compile(
            optimizer = tf.keras.optimizers.Adam(Learning_Rate),
            loss = tf.keras.losses.MeanSquaredError()
            )
        
        #sivModel.fit(X, yrep, epochs=50, verbose=2)
        sivModel_SIDIS.fit(X, y, sample_weight=(1/trn_err_SIDIS**2), validation_data=(tst_X_SIDIS, tst_y_SIDIS), epochs=EPOCHS, verbose=2)
        
        sivModel_SIDIS.save('Models_SIDIS_100/rep' + str(i) + '.h5', save_format='h5')
        
        
trainReplicas(trn_X_SIDIS, trn_y_SIDIS, trn_err_SIDIS, 100)