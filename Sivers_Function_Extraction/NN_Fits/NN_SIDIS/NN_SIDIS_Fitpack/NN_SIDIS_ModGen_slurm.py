import tensorflow as tf
import pandas as pd
import numpy as np
import lhapdf
import matplotlib.pyplot as plt
import sys

#################################################################
##################### Hyperparameters ###########################
#################################################################

Hidden_Layers=2
Nodes_per_HL=256
Learning_Rate = 0.0001
EPOCHS = 10000

#################################################################
save_path = 'Models_SIDIS/rep'

#herm9 = pd.read_csv('./Data/HERMES_p_2009.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
herm20 = pd.read_csv('./Data/HERMES_p_2020.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
comp9 = pd.read_csv('./Data/COMPASS_d_2009.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
comp15 = pd.read_csv('./Data/COMPASS_p_2015.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')

#df = pd.concat([herm9, herm20, comp9, comp15])
df = pd.concat([herm20, comp9, comp15])

# def dataslicercombined(df, hadrons, dependencies):
#     df = df.loc[df['hadron'].isin(hadrons), :]
#     df = df.loc[df['1D_dependence'].isin(dependencies), :]
#     X = np.array(df[['x', 'z', 'phT', 'Q2', 'hadron']])
#     for i, had in enumerate(['pi+', 'pi-', 'pi0', 'k+', 'k-']):
#         X[X[:, 4] == had, 4] = i
#     X = X.astype('float')
#     return X, np.array(df['Siv']), np.array(df['tot_err'])


def chisquare(y, yhat, err):
    return np.sum(((y - yhat)/err)**2)

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
    
    
class Quotient(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Quotient, self).__init__()
    
    def call(self, inputs):
        if len(inputs) != 2 or inputs[0].shape[1] != 1:
            raise Exception('must be two tensors of shape (?, 1)')
        return inputs[0]/inputs[1]
    
# def create_nn(inp, name, hidden_layers=Hidden_Layers, width=Nodes_per_HL, activation='relu'):
#     x = tf.keras.layers.Dense(width, activation=activation)(inp)
#     for i in range(hidden_layers-1):
#         x = tf.keras.layers.Dense(width, activation=activation)(x)
#     nnout = tf.keras.layers.Dense(1, name=name)(x)
#     return nnout

# def create_nn(name, hidden_layers=Hidden_Layers, width=Nodes_per_HL, activation='relu'):
#     inp = tf.keras.Input(shape=(1))
#     x = tf.keras.layers.Dense(width, activation=activation)(inp)
#     for i in range(hidden_layers-1):
#         x = tf.keras.layers.Dense(width, activation=activation)(x)
#         #xdrop = tf.keras.layers.Dropout((0.25))(x)
#     nnout = tf.keras.layers.Dense(1)(x)
#     mod = tf.keras.Model(inp, nnout, name=name)
#     return mod

def create_nn(name, hidden_layers=Hidden_Layers, width=Nodes_per_HL, activation='relu'):
    inp = tf.keras.Input(shape=(1))
    initializer = tf.keras.initializers.Zeros()
    x = tf.keras.layers.Dense(width, activation=activation, kernel_initializer = initializer)(inp)
    for i in range(hidden_layers-1):
        x = tf.keras.layers.Dense(width, activation=activation, kernel_initializer = initializer)(x)
        #xdrop = tf.keras.layers.Dropout((0.25))(x)
    nnout = tf.keras.layers.Dense(1, kernel_initializer = initializer)(x)
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
        NNs.append(create_nn(i)(x))

    #nncomb = tf.keras.layers.Concatenate()([nnuout, nnubarout, nndout, nndbarout, nnsout, nnsbarout])
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


inputdict = {'x': tf.constant([1., 2.]),
             'z': tf.constant([1., 2.]),
             'phT': tf.constant([1., 2.]),
             'uexpr': tf.constant([1., 2.]),
             'ubarexpr': tf.constant([1., 2.]),
             'dexpr': tf.constant([1., 2.]),
             'dbarexpr': tf.constant([1., 2.]),
             'sexpr': tf.constant([1., 2.]),
             'sbarexpr': tf.constant([1., 2.])}


class DataSIDIS(object):
    def __init__(self, pdfset='cteq61',
                 ff_PIp='NNFF10_PIp_nlo', ff_PIm='NNFF10_PIm_nlo', ff_PIsum='NNFF10_PIsum_nlo',
                 ff_KAp='NNFF10_KAp_nlo', ff_KAm='NNFF10_KAm_nlo'):

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
    
    def makeReplicas(self, df, hadrons, dependencies):        
        df = df.loc[df['hadron'].isin(hadrons), :]
        df = df.loc[df['1D_dependence'].isin(dependencies), :]
        return df

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
            y += list(sliced['Siv'])
            err += list(sliced['tot_err'])
            
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
        
        print(data)
        return data, np.array(y), np.array(err)                      
       
        
def trn_tst(X, y, err, split=0.1):
    tstidxs = np.random.choice(list(range(len(y))), size=int(len(y)*split), replace=False)
    
    tst_X = {k: v[tstidxs] for k, v in X.items()}
    trn_X = {k: np.delete(v, tstidxs) for k, v in X.items()}
    
    tst_y = y[tstidxs]
    trn_y = np.delete(y, tstidxs)
    
    tst_err = err[tstidxs]
    trn_err = np.delete(err, tstidxs)
    
    return trn_X, tst_X, trn_y, tst_y, trn_err, tst_err



Replica_Output_File='Replicas.csv'
dataSIDIS = DataSIDIS()
X, y, err = dataSIDIS.makeData(df, ['pi+', 'pi-', 'pi0', 'k+', 'k-'], ['x', 'z', 'phT'])

def run_replica():
    i = sys.argv[1]
    sivModel = SIDIS_Model()
    sivModel(inputdict)
    sivModel.compile(
        optimizer = tf.keras.optimizers.Adam(Learning_Rate),
        #optimizer = tf.keras.optimizers.SGD(lr=Learning_Rate, momentum=0.8, decay=0.0001),
        loss = tf.keras.losses.MeanSquaredError()
        )
    ### This is to train with the central values    
    if i==0:
        trn_X, tst_X, trn_y, tst_y, trn_err, tst_err = trn_tst(X, y, err)
        sivModel.fit(trn_X, trn_y, sample_weight=(1/trn_err**2), validation_data=(tst_X, tst_y), epochs=EPOCHS, verbose=2)
        replica_number = i
        sivModel.save(save_path + str(replica_number) + '.h5', save_format='h5')

    else:
        tempdf=pd.DataFrame()    
        yrep = np.random.normal(y, err)
        tempdf["Siv_Rep"]= yrep
        sampled_replica_values = tempdf["Siv_Rep"].to_numpy()
        #tempdf[str(i)]= yrep
        #sampled_replica_values = tempdf[str(i)]
        with open(Replica_Output_File, 'a') as csv_file:
            np.savetxt(csv_file, [sampled_replica_values], delimiter=',') #,fmt='%s')
        
        trn_X, tst_X, trn_y, tst_y, trn_err, tst_err = trn_tst(X, yrep, err)
        sivModel.fit(trn_X, trn_y, sample_weight=(1/trn_err**2), validation_data=(tst_X, tst_y), epochs=EPOCHS, verbose=2)
    
        # Get replica slurm_array_ID from slurm file input parameter
        replica_number = sys.argv[1]
        #replica_number = i
        
        # line_count = 0
        # with open(Replica_Output_File) as csv_file:
        #     csv_reader = csv.reader(csv_file, delimiter=',')
        #     for row in csv_reader:
        #         line_count += 1
        
        sivModel.save(save_path + str(replica_number) + '.h5', save_format='h5')

run_replica()
