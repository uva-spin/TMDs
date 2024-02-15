import tensorflow as tf
import pandas as pd
import numpy as np
import lhapdf
import matplotlib.pyplot as plt
import os

Org_Data_path = '../Data/'
E288 = 'E288.csv'
E605 = 'E605.csv'
E772 = 'E772.csv'

e288 = pd.read_csv(Org_Data_path + E288).dropna(axis=0, how='all').dropna(axis=1, how='all')
e605 = pd.read_csv(Org_Data_path + E605).dropna(axis=0, how='all').dropna(axis=1, how='all')
e772 = pd.read_csv(Org_Data_path + E772).dropna(axis=0, how='all').dropna(axis=1, how='all')

df = pd.concat([e288])
df_test = df.head(10)

pt_values = np.array(df_test['PT'])
qm_values = np.array(df_test['QM'])

# Set seeds for reproducibility
from tensorflow.keras import backend as K

os.environ['PYTHONHASHSEED'] = '0'
tf.random.set_seed(1)
np.random.seed(1)

L1_reg = 10**(-12)

def create_dnn_model(name, hidden_layers=2, width=5, activation='relu'):
    inp = tf.keras.Input(shape=(2,))
    initializer = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=42)
    x = tf.keras.layers.Dense(width, activation=activation, kernel_initializer=initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(inp)
    x1 = tf.keras.layers.Dense(width, activation=activation, kernel_initializer=initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(x)
    x2 = tf.keras.layers.Dense(width, activation=activation, kernel_initializer=initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(x1)
    nnout = tf.keras.layers.Dense(1, kernel_initializer=initializer)(x2)
    mod = tf.keras.Model(inp, nnout, name=name)
    return mod

# Define the integrand function using the DNN model
def integrand(modl1, modl2, k, pT, qM):
    int1 = modl1.predict(np.array([[k, qM]]))
    int2 = modl2.predict(np.array([[pT-k, qM]]))
    return int1 * int2

# Numerical integration using Simpson's rule
def integrate_SqSqbar(modl1, modl2, a, b, pT, qM, num_points=4):
    k_values = np.linspace(a, b, num_points)
    h = (b - a) / (num_points - 1)

    integral = integrand(modl1, modl2, a, pT, qM) + integrand(modl1, modl2, b, pT, qM)

    for i in range(1, num_points - 1, 2):
        k = k_values[i]
        integral += 4 * integrand(modl1, modl2, k, pT, qM)

    for i in range(2, num_points - 2, 2):
        k = k_values[i]
        integral += 2 * integrand(modl1, modl2, k, pT, qM)

    integral *= h / 3

    return integral

def integrate_SqSqbar_array(modl1, modl2, a, b, pTarray, QMarray):
    tempSqSqbar = []
    lengtharray = pTarray.shape[0]
    for i in range(lengtharray):
        tempSqSqbar.append(integrate_SqSqbar(modl1, modl2, a, b, pTarray[i], QMarray[i]))
    return np.array(tempSqSqbar)


def createModel_DY():
    # Define input layers
    xA = tf.keras.Input(shape=(1), name='xA')
    xB = tf.keras.Input(shape=(1), name='xB')
    PT = tf.keras.Input(shape=(1), name='PT')
    QM = tf.keras.Input(shape=(1), name='QM')
    fuxA_fubarxB = tf.keras.Input(shape=(1), name='fuxA_fubarxB')
    fubarxA_fuxB = tf.keras.Input(shape=(1), name='fubarxA_fuxB')    
    fdxA_fdbarxB = tf.keras.Input(shape=(1), name='fdxA_fdbarxB')   
    fdbarxA_fdxB = tf.keras.Input(shape=(1), name='fdbarxA_fdxB')    
    fsxA_fsbarxB = tf.keras.Input(shape=(1), name='fsxA_fsbarxB')
    fsbarxA_fsxB = tf.keras.Input(shape=(1), name='fsbarxA_fsxB')

    # Create the DNN models
    u_model = create_dnn_model('nnu')
    d_model = create_dnn_model('nnd')
    s_model = create_dnn_model('nns')
    ubar_model = create_dnn_model('nnubar')
    dbar_model = create_dnn_model('nndbar')
    sbar_model = create_dnn_model('nnsbar')
    
    # Set integration limits for kperp
    a, b = 0, 1  
    
    SuSubar = integrate_SqSqbar_array(u_model, ubar_model, a, b, pt_values, qm_values)
    SubarSu = integrate_SqSqbar_array(ubar_model, u_model, a, b, pt_values, qm_values)
    SdSdbar = integrate_SqSqbar_array(d_model, dbar_model, a, b, pt_values, qm_values)
    SdbarSd = integrate_SqSqbar_array(dbar_model, d_model, a, b, pt_values, qm_values)
    SsSsbar = integrate_SqSqbar_array(s_model, sbar_model, a, b, pt_values, qm_values)
    SsbarSs = integrate_SqSqbar_array(sbar_model, s_model, a, b, pt_values, qm_values)
    
    # Define element-wise multiplications with broadcasting
    SuSubarfuxA_fubarxB = tf.keras.layers.Lambda(lambda x: x[0] * x[1], name='multiply_SuSubarfuxA_fubarxB')([SuSubar, fuxA_fubarxB])
    SubarSufubarxA_fuxB = tf.keras.layers.Lambda(lambda x: x[0] * x[1], name='multiply_SubarSufubarxA_fuxB')([SubarSu, fubarxA_fuxB])
    SdSdbarfdxA_fdbarxB = tf.keras.layers.Lambda(lambda x: x[0] * x[1], name='multiply_SdSdbarfdxA_fdbarxB')([SdSdbar, fdxA_fdbarxB])
    SdbarSdfdbarxA_fdxB = tf.keras.layers.Lambda(lambda x: x[0] * x[1], name='multiply_SdbarSdfdbarxA_fdxB')([SdbarSd, fdbarxA_fdxB])
    SsSsbarfsxA_fsbarxB = tf.keras.layers.Lambda(lambda x: x[0] * x[1], name='multiply_SsSsbarfsxA_fsbarxB')([SsSsbar, fsxA_fsbarxB])
    SsbarSsfsbarxA_fsxB = tf.keras.layers.Lambda(lambda x: x[0] * x[1], name='multiply_SsbarSsfsbarxA_fsxB')([SsbarSs, fsbarxA_fsxB])

    # Sum all arrays element-wise
    FUU1 = tf.keras.layers.Add(name='sum_arrays')([SuSubarfuxA_fubarxB, SubarSufubarxA_fuxB, SdSdbarfdxA_fdbarxB, SdbarSdfdbarxA_fdxB, SsSsbarfsxA_fsbarxB, SsbarSsfsbarxA_fsxB])
    
    return tf.keras.Model(inputs=[xA, xB, PT, QM, fuxA_fubarxB, fubarxA_fuxB, fdxA_fdbarxB, fdbarxA_fdxB, fsxA_fsbarxB, fsbarxA_fsxB],
                          outputs=FUU1)

############################################## DY data ###########################################################

class DY_DataANN(object):
    def __init__(self, pdf_set='cteq61'):

        self.pdf_DYData = lhapdf.mkPDF(pdf_set)

        self.eu = 2/3
        self.eubar = -2/3
        self.ed = -1/3
        self.edbar = 1/3
        self.es = -1/3
        self.esbar = 1/3


    def pdf_DY(self, flavor, x, QM):
        return np.array([self.pdf_DYData.xfxQ(flavor, ax, qq) for ax, qq in zip(x, QM)])


    def makeData(self, df_DY):

        data = {'xA': [],
             'xB': [],
             'PT': [],
             'QM': [],
             'fuxA_fubarxB': [],
             'fubarxA_fuxB': [],
             'fdxA_fdbarxB': [],
             'fdbarxA_fdxB': [],
             'fsxA_fsbarxB': [],
             'fsbarxA_fsxB': []}

        y_DY = []
        err_DY = []


        y_DY += list(df_DY['CS'])
        err_DY += list(df_DY['error'])

        x1 = df_DY['xA']
        x2 = df_DY['xB']
        QQ = df_DY['QM']
        data['fuxA_fubarxB'] += list(self.eu**2 * (1/x1)*(self.pdf_DY(2, x1, QQ)) * (1/x2)*(self.pdf_DY(-2, x2, QQ)))
        data['fubarxA_fuxB'] += list(self.eubar**2 * (1/x1)*(self.pdf_DY(-2, x1, QQ)) * (1/x2)*(self.pdf_DY(2, x2, QQ)))
        data['fdxA_fdbarxB'] += list(self.ed**2 * (1/x1)*(self.pdf_DY(1, x1, QQ)) * (1/x2)*(self.pdf_DY(-1, x2, QQ)))
        data['fdbarxA_fdxB'] += list(self.edbar**2 * (1/x1)*(self.pdf_DY(-1, x1, QQ)) * (1/x2)*(self.pdf_DY(1, x2, QQ)))
        data['fsxA_fsbarxB'] += list(self.es**2 * (1/x1)*(self.pdf_DY(3, x1, QQ)) * (1/x2)*(self.pdf_DY(-3, x2, QQ)))
        data['fsbarxA_fsxB'] += list(self.esbar**2 * (1/x1)*(self.pdf_DY(-3, x1, QQ)) * (1/x2)*(self.pdf_DY(3, x2, QQ)))

        data['xA'] += list(x1)
        data['xB'] += list(x2)
        data['PT'] += list(df_DY['PT'])
        data['QM'] += list(QQ)

        for key in data.keys():
            data[key] = np.array(data[key])

        #print(data)
        return data, np.array(y_DY), np.array(err_DY)

    
DYdata = DY_DataANN()

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


T_Xplt, T_yplt, T_errplt = DYdata.makeData(df)
    
trn_X, tst_X, trn_y, tst_y, trn_err, tst_err = trn_tst(T_Xplt, T_yplt, T_errplt)

Model_DY = createModel_DY()
Model_DY.compile(optimizer = tf.keras.optimizers.Adam(0.01),loss = tf.keras.losses.MeanSquaredError())

Model_DY.summary()

Model_DY.fit(trn_X, trn_y, validation_data=(tst_X, tst_y), epochs=300, verbose=2)
