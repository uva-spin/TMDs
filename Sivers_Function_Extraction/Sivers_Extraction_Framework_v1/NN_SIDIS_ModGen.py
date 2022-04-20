import os
import tensorflow as tf
import pandas as pd
import numpy as np
import lhapdf
import matplotlib.pyplot as plt



Generic_path = 'Pseudo-Data/'
Replica_Output_Folder='Set_3_Replicas'
RepFolder = Generic_path + Replica_Output_Folder
RepFilesArray = os.listdir(RepFolder)
save_path = 'Models_SIDIS_Set_3/rep'

#################################################################
##################### Hyperparameters ###########################
#################################################################

Hidden_Layers=2
Nodes_per_HL=256
Learning_Rate = 0.00001
EPOCHS = 600


### Here we create models for each quark-flavor (only input is x) ###

# def create_nn_model(inp, name, hidden_layers=Hidden_Layers, width=Nodes_per_HL, activation='relu'):
#     x = tf.keras.layers.Dense(width, activation=activation)(inp)
#     for i in range(hidden_layers-1):
#         x = tf.keras.layers.Dense(width, activation=activation)(x)
#     nnout = tf.keras.layers.Dense(1, name=name)(x)
#     return nnout

### Here we create models for each quark-flavor (only input is x) ###
def create_nn_model(name, hidden_layers=Hidden_Layers, width=Nodes_per_HL, activation='relu'):
    inp = tf.keras.Input(shape=(1))
    x = tf.keras.layers.Dense(width, activation=activation)(inp)
    for i in range(hidden_layers-1):
        x = tf.keras.layers.Dense(width, activation=activation)(x)
        #xdrop = tf.keras.layers.Dropout((0.25))(x)
    nnout = tf.keras.layers.Dense(1)(x)
    #moddrop =  tf.keras.layers.Dropout((0.25))(nnout)
    #mod = tf.keras.Model(inp, moddrop, name=name)
    #moddrop =  tf.keras.layers.Dropout((0.5))(mod)
    mod = tf.keras.Model(inp, nnout, name=name)
    return mod

# def create_nn_model(name):
#     inp = tf.keras.Input(shape=(1))
#     L1 = tf.keras.layers.Dense(32, activation='relu')(inp) 
#     L2 = tf.keras.layers.Dense(256, activation='relu')(L1)
#     L3= tf.keras.layers.Dense(256, activation='relu')(L2)
#     L4 = tf.keras.layers.Dense(32, activation='relu')(L3)
#     nnout = tf.keras.layers.Dense(1)(L4)
#     moddrop =  tf.keras.layers.Dropout((0.25))(nnout)
#     mod = tf.keras.Model(inp, moddrop, name=name)
#     return mod


def chisquare(y, yhat, err):
    return np.sum(((y - yhat)/err)**2)


class A0(tf.keras.layers.Layer):
    def __init__(self, kperp2avg=0.57, pperp2avg=0.12, **kwargs):
        super(A0, self).__init__(name='a0')
        self.m1 = tf.Variable(1., name='m1')
        #self.m1 = tf.Variable(35., name='m1')
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

    NNs=[]
    for i in ['nnu','nnubar','nnd','nndbar','nns','nnsbar']:
        NNs.append(create_nn_model(i)(x))
        #NNs.append(create_nn_model(x,i))

    nncomb = tf.keras.layers.Concatenate()(NNs)
    denominator = tf.keras.layers.Add()([uexpr, ubarexpr, dexpr, dbarexpr, sexpr, sbarexpr])
    exprcomb = tf.keras.layers.Concatenate()([uexpr, ubarexpr, dexpr, dbarexpr, sexpr, sbarexpr])
    numerator = tf.keras.layers.Dot(axes=1)([nncomb, exprcomb])
    quo = Quotient()([numerator, denominator])
    zpht = tf.keras.layers.Concatenate()([z, phT])
    a0 = A0()(zpht)    
    siv = tf.keras.layers.Multiply()([a0, quo])
    return tf.keras.Model([x, z, phT, uexpr, ubarexpr, dexpr, dbarexpr, sexpr, sbarexpr], siv)



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
        
        y = []
        err = []
        
        df = df.loc[df['hadron'].isin(hadrons), :]
        df = df.loc[df['1D_dependence'].isin(dependencies), :]
        #X = np.array(df[['x', 'z', 'phT', 'Q2', 'hadron']])
        for i, had in enumerate(['pi+', 'pi-', 'pi0', 'k+', 'k-']):
            sliced = df.loc[df['hadron'] == had, :]
            y += list(sliced['Siv_Rep'])
            err += list(sliced['Siv_Rep_err'])
            
            x = sliced['x']
            z = sliced['z']
            QQ = sliced['Q2']
            data['uexpr'] += list(self.eu**2 * self.pdf(2, x, QQ) * self.ff(self.ffDict[i], 2, z, QQ))
            data['ubarexpr'] += list(self.eubar**2 * self.pdf(-2, x, QQ) * self.ff(self.ffDict[i], -2, z, QQ))
            data['dexpr'] += list(self.ed**2 * self.pdf(1, x, QQ) * self.ff(self.ffDict[i], 1, z, QQ))
            data['dbarexpr'] += list(self.edbar**2 * self.pdf(-1, x, QQ) * self.ff(self.ffDict[i], -1, z, QQ))
            data['sexpr'] += list(self.es**2 * self.pdf(3, x, QQ) * self.ff(self.ffDict[i], 3, z, QQ))
            data['sbarexpr'] += list(self.esbar**2 * self.pdf(-3, x, QQ) * self.ff(self.ffDict[i], -3, z, QQ))

            data['x'] += list(x)
            data['z'] += list(z)
            data['phT'] += list(sliced['phT'])
        
        for key in data.keys():
            data[key] = np.array(data[key])
        
        return data, np.array(y), np.array(err)                      
       
        

def trn_tst(X, y, err, split=0.2):
    tstidxs = np.random.choice(list(range(len(y))), size=int(len(y)*split), replace=False)
    
    tst_X = {k: v[tstidxs] for k, v in X.items()}
    trn_X = {k: np.delete(v, tstidxs) for k, v in X.items()}
    
    tst_y = y[tstidxs]
    trn_y = np.delete(y, tstidxs)
    
    tst_err = err[tstidxs]
    trn_err = np.delete(err, tstidxs)
    
    return trn_X, tst_X, trn_y, tst_y, trn_err, tst_err

#print(len(RepFilesArray))


# def trainReplicas(Rep):
#     for i in range(Rep):
#         print('*******************')
#         print('Here is replica'+str(i))
#         print(RepFilesArray[i])
#         df = pd.read_csv(RepFolder+'/'+ RepFilesArray[i]).dropna(axis=0, how='all').dropna(axis=1, how='all')
#         # df = pd.read_csv(RepFolder+'/'+ RepFilesArray[i])
        
#         dataSIDIS = DataSIDIS()
#         X, y, err = dataSIDIS.makeData(df, ['pi+', 'pi-', 'pi0', 'k+', 'k-'], ['x', 'z', 'phT'])

#         trn_X, tst_X, trn_y, tst_y, trn_err, tst_err = trn_tst(X, y, err)

#         #print(trn_X)
#         sivModel = SIDIS_Model()
#         sivModel.compile(
#             optimizer = tf.keras.optimizers.Adam(Learning_Rate),
#             loss = tf.keras.losses.MeanSquaredError()
#             )

#         # # #sivModel.fit(X, yrep, epochs=50, verbose=2)
#         # # #sivModel.fit(trn_X, trn_y, sample_weight=(1/trn_err**2)/342, validation_data=(tst_X, tst_y), epochs=EPOCHS, verbose=2)
#         sivModel.fit(trn_X, trn_y, validation_data=(tst_X, tst_y), epochs=EPOCHS, verbose=2)
        
#         sivModel.save('Models_SIDIS/rep' + str(i) + '.h5', save_format='h5')

#         print('*********** Done ********************')
        
        
#trainReplicas(80)

for i in range(200):
    print('*******************')
    print('Here is replica '+str(i))
    print(RepFilesArray[i])
    df = pd.read_csv(RepFolder+'/'+ RepFilesArray[i]).dropna(axis=0, how='all').dropna(axis=1, how='all')
    # df = pd.read_csv(RepFolder+'/'+ RepFilesArray[i])
    
    dataSIDIS = DataSIDIS()
    X, y, err = dataSIDIS.makeData(df, ['pi+', 'pi-', 'pi0', 'k+', 'k-'], ['x', 'z', 'phT'])

    trn_X, tst_X, trn_y, tst_y, trn_err, tst_err = trn_tst(X, y, err)

    #print(trn_X)
    sivModel = SIDIS_Model()
    sivModel.compile(
        optimizer = tf.keras.optimizers.Adam(Learning_Rate),
        loss = tf.keras.losses.MeanSquaredError()
        )
    
    # # #sivModel.fit(X, yrep, epochs=50, verbose=2)
    # # #sivModel.fit(trn_X, trn_y, sample_weight=(1/trn_err**2)/342, validation_data=(tst_X, tst_y), epochs=EPOCHS, verbose=2)
    sivModel.fit(trn_X, trn_y, validation_data=(tst_X, tst_y), epochs=EPOCHS, verbose=2)
    
    sivModel.save(save_path + str(i) + '.h5', save_format='h5')
    #sivModel.save('rep'+ str(i) + '.h5', save_format='h5')

    print('******* Done *********') 