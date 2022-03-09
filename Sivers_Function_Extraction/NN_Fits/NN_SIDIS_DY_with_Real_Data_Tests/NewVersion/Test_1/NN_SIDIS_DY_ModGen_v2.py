#######################################################################
############ Generating Replicas for NN model : SIDIS #################
############ Written by Ishara Fernando & Nick Newton #################
############ Last upgrade: December-09-2021 ################################
#######################################################################


from calendar import EPOCH
import tensorflow as tf
import pandas as pd
import numpy as np
import lhapdf
import matplotlib.pyplot as plt

#import Mod_Gen_functions

Hidden_Layers=2
Nodes_per_HL=20
Learning_Rate = 0.0001
EPOCHS = 5
REPLICAS = 2
Batch_Size = 6
optimizer = tf.keras.optimizers.Adam(Learning_Rate)
loss_fn = tf.keras.losses.MeanSquaredError()


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
        #print(inputs)
        #z = inputs[:, 0]
        #pht = inputs[:, 1]
        qT = inputs[:, 0]
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
def create_nn_model(name, hidden_layers=Hidden_Layers, width=Nodes_per_HL, activation='relu'):
    inp = tf.keras.Input(shape=(1))
    x = tf.keras.layers.Dense(width, activation=activation)(inp)
    for i in range(hidden_layers-1):
        x = tf.keras.layers.Dense(width, activation=activation)(x)
    nnout = tf.keras.layers.Dense(1)(x)
    mod = tf.keras.Model(inp, nnout, name=name)
    return mod


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


    modnnu = create_nn_model('nnu')
    modnnubar = create_nn_model('nnubar')
    modnnd = create_nn_model('nnd')
    modnndbar = create_nn_model('nndbar')
    modnns = create_nn_model('nns')
    modnnsbar = create_nn_model('nnsbar')

    nnu_x1 = modnnu(x1)
    nnu_x2 = modnnu(x2)
    nnubar_x1 = modnnubar(x1)
    nnubar_x2 = modnnubar(x2)
    nnd_x1 = modnnd(x1)
    nnd_x2 = modnnd(x2)
    nndbar_x1 = modnndbar(x1)
    nndbar_x2 = modnndbar(x2)  
    nns_x1 = modnns(x1)
    nns_x2 = modnns(x2)
    nnsbar_x1 = modnnsbar(x1)
    nnsbar_x2 = modnnsbar(x2)

    nncomb = tf.keras.layers.Concatenate()([nnu_x1, nnu_x2, nnd_x1, nnd_x2, nns_x1, nns_x2,
             nnubar_x1, nnubar_x2, nndbar_x1, nndbar_x2, nnsbar_x1, nnsbar_x2])
    denominator = tf.keras.layers.Add()([Uexpr_x1, Uexpr_x2, Ubarexpr_x1, Ubarexpr_x2, Dexpr_x1, Dexpr_x2, Dbarexpr_x1, Dbarexpr_x2, Sexpr_x1, Sexpr_x2, Sbarexpr_x1, Sbarexpr_x2])
    exprcomb = tf.keras.layers.Concatenate()([Uexpr_x1, Uexpr_x2, Ubarexpr_x1, Ubarexpr_x2, Dexpr_x1, Dexpr_x2, Dbarexpr_x1, Dbarexpr_x2, Sexpr_x1, Sexpr_x2, Sbarexpr_x1, Sbarexpr_x2])
    numerator = tf.keras.layers.Dot(axes=1)([nncomb, exprcomb])
    quo = Quotient()([numerator, denominator])
    b0 = B0()(qT)
    siv = tf.keras.layers.Multiply()([b0, quo])
    return tf.keras.Model([x1, x2, qT, Uexpr_x1, Uexpr_x2, Ubarexpr_x1, Ubarexpr_x2, Dexpr_x1, Dexpr_x2, Dbarexpr_x1, Dbarexpr_x2, Sexpr_x1, Sexpr_x2, Sbarexpr_x1, Sbarexpr_x2],
                         siv)


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


#####################################
#### Initialization of the models ###
#####################################

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


################################################################################
### Preparation of the terms in expressions using dictionaries defined above ###
################################################################################

#sivModel_SIDIS(inputdict_SIDIS)
#sivModel_DY(inputdict_DY)

############################################## SIDIS data ###########################################################

class SIDIS_DataANN(object):
    def __init__(self, pdf_SIDISset='cteq61',
                 ff_PIp='NNFF10_PIp_nlo', ff_PIm='NNFF10_PIm_nlo', ff_PIsum='NNFF10_PIsum_nlo',
                 ff_KAp='NNFF10_KAp_nlo', ff_KAm='NNFF10_KAm_nlo'):
        '''
        Get data in proper format for neural network
        '''
        self.pdf_SIDISData = lhapdf.mkPDF(pdf_SIDISset)
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
            data['uexpr'] += list(self.eu**2 * (1/x)*(self.pdf_SIDIS(2, x, QQ)) * self.ff(self.ffDict[i], 2, z, QQ))
            data['ubarexpr'] += list(self.eubar**2 * (1/x)*(self.pdf_SIDIS(-2, x, QQ)) * self.ff(self.ffDict[i], -2, z, QQ))
            data['dexpr'] += list(self.ed**2 * (1/x)*(self.pdf_SIDIS(1, x, QQ)) * self.ff(self.ffDict[i], 1, z, QQ))
            data['dbarexpr'] += list(self.edbar**2 * (1/x)*(self.pdf_SIDIS(-1, x, QQ) )* self.ff(self.ffDict[i], -1, z, QQ))
            data['sexpr'] += list(self.es**2 * (1/x)*(self.pdf_SIDIS(3, x, QQ)) * self.ff(self.ffDict[i], 3, z, QQ))
            data['sbarexpr'] += list(self.esbar**2 * (1/x)*(self.pdf_SIDIS(-3, x, QQ)) * self.ff(self.ffDict[i], -3, z, QQ))

            data['x'] += list(x)
            data['z'] += list(z)
            data['phT'] += list(sliced['phT'])

        for key in data.keys():
            data[key] = np.array(data[key])

        #print(data)
        return data, np.array(y), np.array(err)


############################################## DY data ###########################################################

class DY_DataANN(object):
    def __init__(self, pdf_set='cteq61'):
        '''
        Get data in proper format for neural network
        '''
        self.pdf_DYData = lhapdf.mkPDF(pdf_set)

        self.eu = 2/3
        self.eubar = -2/3
        self.ed = -1/3
        self.edbar = 1/3
        self.es = -1/3
        self.esbar = 1/3


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
            data['Uexpr_x1'] += list(self.eu**2 * (1/x1)*(self.pdf_DY(2, x1, QQ)) * (1/x2)*(self.pdf_DY(-2, x2, QQ)))
            data['Uexpr_x2'] += list(self.eu**2 * (1/x1)*(self.pdf_DY(2, x2, QQ)) * (1/x2)*(self.pdf_DY(-2, x1, QQ)))
            data['Ubarexpr_x1'] += list(self.eubar**2 * (1/x1)*(self.pdf_DY(-2, x1, QQ)) * (1/x2)*(self.pdf_DY(2, x2, QQ)))
            data['Ubarexpr_x2'] += list(self.eubar**2 * (1/x2)*(self.pdf_DY(-2, x2, QQ)) * (1/x1)*(self.pdf_DY(2, x1, QQ)))
            data['Dexpr_x1'] += list(self.ed**2 * (1/x1)*(self.pdf_DY(1, x1, QQ)) * (1/x2)*(self.pdf_DY(-1, x2, QQ)))
            data['Dexpr_x2'] += list(self.ed**2 * (1/x2)*(self.pdf_DY(1, x2, QQ)) * (1/x1)*(self.pdf_DY(-1, x1, QQ)))
            data['Dbarexpr_x1'] += list(self.edbar**2 * (1/x1)*(self.pdf_DY(-1, x1, QQ)) * (1/x2)*(self.pdf_DY(1, x2, QQ)))
            data['Dbarexpr_x2'] += list(self.edbar**2 * (1/x2)*(self.pdf_DY(-1, x2, QQ)) * (1/x1)*(self.pdf_DY(1, x1, QQ)))
            data['Sexpr_x1'] += list(self.es**2 * (1/x1)*(self.pdf_DY(3, x1, QQ)) * (1/x2)*(self.pdf_DY(-3, x2, QQ)))
            data['Sexpr_x2'] += list(self.es**2 * (1/x2)*(self.pdf_DY(3, x2, QQ)) * (1/x1)*(self.pdf_DY(-3, x1, QQ)))
            data['Sbarexpr_x1'] += list(self.esbar**2 * (1/x1)*(self.pdf_DY(-3, x1, QQ)) * (1/x2)*(self.pdf_DY(3, x2, QQ)))
            data['Sbarexpr_x2'] += list(self.esbar**2 * (1/x2)*(self.pdf_DY(-3, x2, QQ)) * (1/x1)*(self.pdf_DY(3, x1, QQ)))


            data['x1'] += list(x1)
            data['x2'] += list(x2)
            data['qT'] += list(sliced['QT'])

        for key in data.keys():
            data[key] = np.array(data[key])

        #print(data)
        return data, np.array(y_DY), np.array(err_DY)

 #################################################################################################################



def trn_tst(X, y, err, split=0.1):
    tstidxs = np.random.choice(list(range(len(y))), size=int(len(y)*split), replace=False)

    tst_X = {k: v[tstidxs] for k, v in X.items()}
    trn_X = {k: np.delete(v, tstidxs) for k, v in X.items()}

    tst_y = y[tstidxs]
    trn_y = np.delete(y, tstidxs)

    tst_err = err[tstidxs]
    trn_err = np.delete(err, tstidxs)

    return trn_X, tst_X, trn_y, tst_y, trn_err, tst_err





####################################################################################################################


def nnq(model, x, hadronstr):
    if not hadronstr in ['nnu','nnubar','nnd','nndbar','nns','nnsbar']:
        raise Exception('hadronstr must be one of nnu, nnd, nns, nnubar, nndbar, nnsbar')
    lilmod = tf.keras.backend.function(x,model.get_layer(hadronstr).output,hadronstr)
    return lilmod


def q_val(model, x, flavor):
    refDict = {-3: 'nnsbar',
               -2: 'nnubar',
               -1: 'nndbar',
               1: 'nnd',
               2: 'nnu',
               3: 'nns'}
    nnqval = nnq(model, np.array([x]), refDict[flavor])
    return ((nnqval)[0, :])    

# def trainReplicas(X, y, err, numReplicas):
#     for i in range(numReplicas):
#         yrep = np.random.normal(y, err)

#         sivModel_SIDIS = createModel_SIDIS()

#         sivModel_SIDIS.compile(
#             optimizer = tf.keras.optimizers.Adam(Learning_Rate),
#             loss = tf.keras.losses.MeanSquaredError()
#             )

#         #sivModel.fit(X, yrep, epochs=50, verbose=2)
#         sivModel_SIDIS.fit(X_SIDIS, y_SIDIS, sample_weight=(1/trn_err_SIDIS**2), validation_data=(tst_X_SIDIS, tst_y_SIDIS), epochs=EPOCHS, verbose=2)
        
#         #temp_a0=sivModel_SIDIS.get_layer('a0').m1.numpy()
#         #print(temp_a0)
#         sivModel_SIDIS.save('Models_test/rep' + str(i) + '.h5', save_format='h5')
#         #temp_nnu=sivModel_SIDIS.get_layer('nnu')(np.array([0.1]))
#         #print(temp_nnu)     
#         nnu_w_SIDIS=sivModel_SIDIS.get_layer('nnu').weights
#         # nnubar_w_SIDIS=sivModel_SIDIS.get_layer('nnubar').weights
#         # nnd_w_SIDIS=sivModel_SIDIS.get_layer('nnd').weights
#         # nndbar_w_SIDIS=sivModel_SIDIS.get_layer('nndbar').weights
#         # nns_w_SIDIS=sivModel_SIDIS.get_layer('nns').weights
#         # nnsbar_w_SIDIS=sivModel_SIDIS.get_layer('nnsbar').weights
        
#         nnu_w_DY=sivModel_DY.get_layer('nnu').setWeights(nnu_w_SIDIS) 
#         #print(nnu_w,nnd_w)                     


def yield_batch(X,Y,sample_weights,batch_size):
    batch_indices=np.random.choice(range(len(Y)),size=batch_size,replace=False)
    sample_weights=sample_weights[batch_indices]
    Y=Y[batch_indices]
    X={k:V[batch_indices] for k,V in X.items()}
    return X,Y,sample_weights



def customfit(replicaID, model1, model2, epchs, X1, y1, sample_weights1, X1Val, y1Val, sample_weights1Val,
X2, y2, sample_weights2, X2Val, y2Val, sample_weights2Val, loss_fn, optimizer, batch_size):
    #epochs = 2
    Lowest_Val_Loss = np.Inf
    Num_since_LowestVal = 0
    #for epoch in range(epchs):
    epoch = 0
    while True:
        
        if Num_since_LowestVal > 5:
            break

        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        #for step, (x_batch_train, y_batch_train, sample_weights_batch) in enumerate(yield_batch(X,y,sample_weights,batch_size)):
        for step in range((len(y1)+len(y2))//batch_size):
            if np.random.uniform()<len(y1)/(len(y1)+len(y2)):
                ismodel1=True
                x_batch, y_batch, sample_weights_batch = yield_batch(X1,y1,sample_weights1, batch_size)
                #x_val, y_val, yerr_val = X1Val, y1Val, sample_weights1Val
                model=model1
            else:
                ismodel1=False
                x_batch, y_batch, sample_weights_batch = yield_batch(X2,y2,sample_weights2, batch_size)
                #x_val, y_val, yerr_val = X2Val, y2Val, sample_weights2Val
                model=model2


            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:

                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.

                model.compile(optimizer=optimizer,loss=loss_fn)
                model.fit(x_batch, y_batch, sample_weight=sample_weights_batch, epochs=epchs, verbose=2)
                y_hat = model(x_batch, training=True)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                loss_value = loss_fn(y_batch, y_hat, sample_weight=sample_weights_batch)

                #print(X1Val)



            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # if (Total_Validation < Lowest_Val_Loss):
            #     Lowest_Val_Loss = Total_Validation
            #     model.save('SIDISmodels/rep' +str(replicaID) + '.h5', save_format='h5')
            #     model.save('DYmodels/rep' +str(replicaID) + '.h5', save_format='h5')


            if ismodel1:
                model2.get_layer('b0').m1 = model1.get_layer('a0').m1
                for quark_type in ['nnu','nnubar','nnd','nndbar','nns','nnsbar']:
                    weigths = model1.get_layer(quark_type).get_weights()
                    model2.get_layer(quark_type).set_weights(weigths)
                #model.save('SIDISmodels/rep' + str(step) + '.h5', save_format='h5')
            else:
                model1.get_layer('a0').m1 = model2.get_layer('b0').m1
                for quark_type in ['nnu','nnubar','nnd','nndbar','nns','nnsbar']:
                    weigths = model2.get_layer(quark_type).get_weights()
                    model1.get_layer(quark_type).set_weights(weigths)
                #model.save('DYmodels/rep' + str(step) + '.h5', save_format='h5')
                

            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * batch_size))

        # Compute the validation loss value for this minibatch.
        SIDIS_y_hat_Val = model1(X1Val, training=False)
        SIDIS_Validation_loss = loss_fn(y1Val, SIDIS_y_hat_Val, sample_weight=sample_weights1Val)

        #print(X2Val)
        DY_y_hat_Val = model2(X2Val, training=False)
        DY_Validation_loss = loss_fn(y2Val, DY_y_hat_Val, sample_weight=sample_weights2Val)

        Total_Validation = (len(y1Val)*SIDIS_Validation_loss + len(y2Val)*DY_Validation_loss)/(len(y1Val)+len(y2Val))

        Num_since_LowestVal += 1

        if (Total_Validation < Lowest_Val_Loss):
            Lowest_Val_Loss = Total_Validation
            model.save('SIDISmodels/rep' +str(replicaID) + '.h5', save_format='h5')
            model.save('DYmodels/rep' +str(replicaID) + '.h5', save_format='h5')
            Num_since_LowestVal = 0

        epoch += 1     



def trainReplicas_new(epochs, X1, y1, err1, X1Val, y1Val, err1Val, X2, y2, err2, X2Val, y2Val, err2Val, loss_fn, optimizer, numReplicas, batch_size):
    for i in range(numReplicas):
        yrep1 = np.random.normal(y1, err1)
        yrep2 = np.random.normal(y2, err2)
        yrep1Val = np.random.normal(y1Val, err1Val)
        yrep2Val = np.random.normal(y2Val, err2Val)
        #yrep1 = y1
        #yrep2 = y2


        sivModel_SIDIS = createModel_SIDIS()
        sivModel_DY = createModel_DY()

        customfit(i,sivModel_SIDIS, sivModel_DY, epochs, X1, yrep1, np.square(1/err1), X1Val, yrep1Val, np.square(1/err1Val),
        X2, yrep2, np.square(1/err2), X2Val, yrep2Val, np.square(1/err2Val),loss_fn, optimizer, batch_size)

        ### Remember to save the models
        

#trainReplicas(trn_X_SIDIS, trn_y_SIDIS, trn_err_SIDIS, 1)

if __name__=='__main__': ## double underscore means dundered 

    #################################################################
    ##################### Hyperparameters ###########################
    #################################################################
    #print("Hello this works!")
    # Hidden_Layers=2
    # Nodes_per_HL=256
    # Learning_Rate = 0.0001
    # EPOCHS = 5000

    # Hidden_Layers=2
    # Nodes_per_HL=10
    # Learning_Rate = 0.0001
    # EPOCHS = 5

    ############################### Data Files ##################################

    ### SIDIS ###

    herm9_SIDIS = pd.read_csv('Data/HERMES_p_2009.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
    herm20_SIDIS = pd.read_csv('Data/HERMES_p_2020.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
    comp9_SIDIS = pd.read_csv('Data/COMPASS_d_2009.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
    comp15_SIDIS = pd.read_csv('Data/COMPASS_p_2015.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')

    ### DY ###

    comp15_DY = pd.read_csv('Data/COMPASS_p_DY_2017.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')

    #df_SIDIS = pd.concat([herm9_SIDIS, herm20_SIDIS, comp9_SIDIS, comp15_SIDIS])
    df_SIDIS = pd.concat([herm20_SIDIS, comp9_SIDIS, comp15_SIDIS])

    df_DY= pd.concat([comp15_DY])


    SIDIS_datann = SIDIS_DataANN()
    X_SIDIS, y_SIDIS, err_SIDIS = SIDIS_datann.makeData(df_SIDIS, ['pi+', 'pi-', 'pi0', 'k+', 'k-'], ['x', 'z', 'phT'])
    trn_X_SIDIS, tst_X_SIDIS, trn_y_SIDIS, tst_y_SIDIS, trn_err_SIDIS, tst_err_SIDIS = trn_tst(X_SIDIS, y_SIDIS, err_SIDIS)

    DY_datann = DY_DataANN()
    X_DY, y_DY, err_DY = DY_datann.makeData(df_DY, ['x1', 'x2', 'xF', 'QT', 'QM'])
    trn_X_DY, tst_X_DY, trn_y_DY, tst_y_DY, trn_err_DY, tst_err_DY = trn_tst(X_DY, y_DY, err_DY)

    # optimizer = tf.keras.optimizers.Adam(Learning_Rate)
    # loss_fn = tf.keras.losses.MeanSquaredError()

    #trainReplicas_new(EPOCHS, X_SIDIS, y_SIDIS, err_SIDIS, X_DY, y_DY, err_DY, loss_fn, optimizer, REPLICAS, Batch_Size)
    trainReplicas_new(EPOCHS, trn_X_SIDIS, trn_y_SIDIS, trn_err_SIDIS, tst_X_SIDIS, tst_y_SIDIS, tst_err_SIDIS, 
    trn_X_DY, trn_y_DY, trn_err_DY, tst_X_DY, tst_y_DY, tst_err_DY, loss_fn, optimizer, REPLICAS, Batch_Size)

    

            







