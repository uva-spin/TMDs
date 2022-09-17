import os
import tensorflow as tf
import pandas as pd
import numpy as np
import lhapdf
import matplotlib.pyplot as plt
import csv
import sys
import random
import re
import multiprocessing

#SIDIS_Models_folder = '/scratch/cee9hc/Sivers_NN_Models/Train_Real/Models_SIDIS_v7'
#SIDIS_models_array=os.listdir(SIDIS_Models_folder)

save_path = '/scratch/cee9hc/Sivers_NN_Models/Train_Real/Fit3/Models_DY_wRD/'


DY_Repl_Folder = 'DY_wRD_Replica_Data'
Losses_Folder = 'DY_wRD_Replicas_Losses'

DY_Projected_DATA = pd.read_csv('./NN_SIDIS_Fit_Results/Result_DY_from_SIDIS_minus.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
DY_Projected_DATA_df = pd.concat([DY_Projected_DATA])

DY_COMPASS17_DATA = pd.read_csv('./Data/COMPASS_p_DY_2017.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
DY_COMPASS17_DATA_df = pd.concat([DY_COMPASS17_DATA])

#df = pd.concat([DY_Projected_DATA])
df = pd.concat([DY_Projected_DATA, DY_COMPASS17_DATA])

uncert_frac = 1

Hidden_Layers=7
Nodes_per_HL=256
Learning_Rate = 0.0001
#EPOCHS = 10000
EPOCHS = 500
L1_reg = 10**(-12)
modify_LR = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',factor=0.9,patience=400,mode='auto')
EarlyStop = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=1000)


def create_nn_model(name, hidden_layers=Hidden_Layers, width=Nodes_per_HL, activation='relu6'):
    inp = tf.keras.Input(shape=(1))
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
    
    
def nnq(model, x, hadronstr):
    if not hadronstr in ['nnu', 'nnd', 'nns', 'nnubar', 'nndbar', 'nnsbar']:
        raise Exception('hadronstr must be one of nnu, nnd, nns, nnubar, nndbar, nnsbar')
    mod_out = tf.keras.backend.function(model.get_layer(hadronstr).input,
                                       model.get_layer(hadronstr).output)
    return mod_out(x)


class B0(tf.keras.layers.Layer):
    def __init__(self, kperp2avg=0.25, pperp2avg=.12, **kwargs):
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


##########################################################################    
###################### DY Definitions ####################################
##########################################################################

class DY_Hadron(object):
    def __init__(self, kperp2avg=0.25, pperp2avg=.12, pdfset='cteq61'):
        #self.pdfData = lhapdf.mkPDF(pdfset)
        self.pdf_DYData = lhapdf.mkPDF(pdfset)
        # needs to be extended to generalize for kaons
        self.kperp2avg = kperp2avg
        self.pperp2avg = pperp2avg
        self.eu = 2/3
        self.eubar = -2/3
        self.ed = -1/3
        self.edbar = 1/3
        self.es = -1/3
        self.esbar = 1/3
        self.e = 1
        
    
    def pdf(self, flavor, x, QQ):
        return np.array([self.pdfData.xfxQ2(flavor, ax, qq) for ax, qq in zip(x, QQ)])

    
    def pdf_DY(self, flavor, x, QM):
        return np.array([self.pdf_DYData.xfxQ(flavor, ax, qq) for ax, qq in zip(x, QM)])
    
    
    def B0(self, qT, m1):
        ks2avg = (self.kperp2avg*m1**2)/(m1**2 + self.kperp2avg)
        topfirst = (self.kperp2avg + self.kperp2avg) * ks2avg**2
        bottomfirst = (ks2avg + self.kperp2avg)**2 * self.kperp2avg
        exptop = -qT**2 / (ks2avg + self.kperp2avg)
        expbottom = -qT**2 / (self.kperp2avg + self.kperp2avg)
        last = np.sqrt(2*self.e) * qT / m1
        return (topfirst/bottomfirst) * np.exp(-exptop/expbottom) * last
    


class Sivers_DY(DY_Hadron):
    def __init__(self, kperp2avg=0.25, pperp2avg=.12, pdfset='cteq61'):
        
        super().__init__(kperp2avg=kperp2avg, pperp2avg=pperp2avg, pdfset=pdfset)
    
        
    def sivers(self, model, kins, SIGN):
        x1 = kins[:, 0]
        x2 = kins[:, 1]
        qT = kins[:, 2]
        QQ = kins[:, 3] 
        m1 = model.get_layer('a0').m1.numpy()
        b0 = self.B0(qT, m1)
        NNuX1 = SIGN*nnq(model , np.array(x1), 'nnu')[:,0]
        NNuX2 = SIGN*nnq(model , np.array(x2), 'nnu')[:,0]
        NNubarX1 = SIGN*nnq(model , np.array(x1), 'nnubar')[:,0]
        NNubarX2 = SIGN*nnq(model , np.array(x2), 'nnubar')[:,0]
        NNdX1 = SIGN*nnq(model , np.array(x1), 'nnd')[:,0]
        NNdX2 = SIGN*nnq(model , np.array(x2), 'nnd')[:,0]
        NNdbarX1 = SIGN*nnq(model , np.array(x1), 'nndbar')[:,0]
        NNdbarX2 = SIGN*nnq(model , np.array(x2), 'nndbar')[:,0]
        NNsX1 = SIGN*nnq(model , np.array(x1), 'nns')[:,0]
        NNsX2 = SIGN*nnq(model , np.array(x2), 'nns')[:,0]
        NNsbarX1 = SIGN*nnq(model , np.array(x1), 'nnsbar')[:,0]
        NNsbarX2 = SIGN*nnq(model , np.array(x2), 'nnsbar')[:,0]
        temp_top = NNuX1 * self.eu**2 * self.pdf_DY(2, x1, QQ) * self.pdf_DY(-2, x2, QQ)
        + NNuX2 * self.eu**2 * self.pdf_DY(2, x2, QQ) * self.pdf_DY(-2, x1, QQ)
        + NNubarX1 * self.eubar**2 * self.pdf_DY(-2, x1, QQ) * self.pdf_DY(2, x2, QQ)
        + NNubarX2 * self.eubar**2 * self.pdf_DY(-2, x2, QQ) * self.pdf_DY(2, x1, QQ)
        + NNdX1 * self.ed**2 * self.pdf_DY(1, x1, QQ) * self.pdf_DY(-1, x2, QQ)
        + NNdX2 * self.ed**2 * self.pdf_DY(1, x2, QQ) * self.pdf_DY(-1, x1, QQ)
        + NNdbarX1 * self.edbar**2 * self.pdf_DY(-1, x1, QQ) * self.pdf_DY(1, x2, QQ)
        + NNdbarX2 * self.edbar**2 * self.pdf_DY(-1, x2, QQ) * self.pdf_DY(1, x1, QQ)
        + NNsX1 * self.es**2 * self.pdf_DY(3, x1, QQ) * self.pdf_DY(-3, x2, QQ)
        + NNsX2 * self.es**2 * self.pdf_DY(3, x2, QQ) * self.pdf_DY(-3, x1, QQ)
        + NNsbarX1 * self.esbar**2 * self.pdf_DY(-3, x1, QQ) * self.pdf_DY(3, x2, QQ)
        + NNsbarX2 * self.esbar**2 * self.pdf_DY(-3, x2, QQ) * self.pdf_DY(3, x1, QQ)
        temp_bottom = self.eu**2 * self.pdf_DY(2, x1, QQ) * self.pdf_DY(-2, x2, QQ)
        + self.eu**2 * self.pdf_DY(2, x2, QQ) * self.pdf_DY(-2, x1, QQ)
        + self.eubar**2 * self.pdf_DY(-2, x1, QQ) * self.pdf_DY(2, x2, QQ)
        + self.eubar**2 * self.pdf_DY(-2, x2, QQ) * self.pdf_DY(2, x1, QQ)
        + self.ed**2 * self.pdf_DY(1, x1, QQ) * self.pdf_DY(-1, x2, QQ)
        + self.ed**2 * self.pdf_DY(1, x2, QQ) * self.pdf_DY(-1, x1, QQ)
        + self.edbar**2 * self.pdf_DY(-1, x1, QQ) * self.pdf_DY(1, x2, QQ)
        + self.edbar**2 * self.pdf_DY(-1, x2, QQ) * self.pdf_DY(1, x1, QQ)
        + self.es**2 * self.pdf_DY(3, x1, QQ) * self.pdf_DY(-3, x2, QQ)
        + self.es**2 * self.pdf_DY(3, x2, QQ) * self.pdf_DY(-3, x1, QQ)
        + self.esbar**2 * self.pdf_DY(-3, x1, QQ) * self.pdf_DY(3, x2, QQ)
        + self.esbar**2 * self.pdf_DY(-3, x2, QQ) * self.pdf_DY(3, x1, QQ)
        temp_siv_had = b0*((temp_top)/(temp_bottom))
        return temp_siv_had
    


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

def chisquare(y, yhat, err):
    return np.sum(((y - yhat)/err)**2)

def calc_yhat(model, X):
    return model.predict(X)
    

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
        
        kins = {'Dependence': [],
               'x1': [],
               'x2': [],
               'xF': [],
               'QT': [],
               'QM': []}

        y_DY = []
        err_DY = []

        sivm = []
        QQs =[]
        xF = []
        deps = []

        df_DY = df_DY.loc[df_DY['Dependence'].isin(dependencies), :]
        for i, dep in enumerate(['x1', 'x2', 'xF', 'QT', 'QM']):
            sliced = df_DY.loc[df_DY['Dependence'] == dep, :]
            y_DY += list(sliced['Siv'])
            err_DY += list(sliced['tot_err'])

            #sivm = sliced['Siv_mod']
            deps = sliced['Dependence']
            # x1 is beam, x2 is target in the COMPASS Data
            # x1 is target, and x2 is beam in the formalism
            x2 = sliced['x1']
            x1 = sliced['x2']
            xF = sliced['xF']
            QT = sliced['QT']
            QQ = sliced['QM']
            data['Uexpr_x1'] += list(self.eu**2 * (1/x1)*(self.pdf_DY(2, x1, QQ)) * (1/x2)*(self.pdf_DY(-2, x2, QQ)))
            data['Uexpr_x2'] += list(self.eu**2 * (1/x2)*(self.pdf_DY(2, x2, QQ)) * (1/x1)*(self.pdf_DY(-2, x1, QQ)))
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


            #kins['Siv_mod']+= list(sivm)
            kins['Dependence']+= list(deps)
            kins['x1']+= list(x1)
            kins['x2']+= list(x2)
            kins['xF']+= list(xF)
            kins['QT']+= list(QT)
            kins['QM']+= list(QQ)

        for key in data.keys():
            data[key] = np.array(data[key])

        for key in kins.keys():
            kins[key] = np.array(kins[key])

        #print(data)
        return kins, data, np.array(y_DY), np.array(err_DY)
        
def trn_tst(X, y, err, split=0.1):
    tstidxs = np.random.choice(list(range(len(y))), size=int(len(y)*split), replace=False)
    
    tst_X = {k: v[tstidxs] for k, v in X.items()}
    trn_X = {k: np.delete(v, tstidxs) for k, v in X.items()}
    
    tst_y = y[tstidxs]
    trn_y = np.delete(y, tstidxs)
    
    tst_err = err[tstidxs]
    trn_err = np.delete(err, tstidxs)
    
    return trn_X, tst_X, trn_y, tst_y, trn_err, tst_err
    

DY_datann = DY_DataANN()



def GenDYReplicaData(tempdf):
    #tempdf=pd.read_csv(datafile)
    #tempMod=np.array(tempdf['Siv_mod'],dtype=object)
    tempDEP=np.array(tempdf['Dependence'],dtype=object)
    tempX1=np.array(tempdf['x1'],dtype=object)
    tempX2=np.array(tempdf['x2'],dtype=object)
    tempXF=np.array(tempdf['xF'],dtype=object)
    tempQT=np.array(tempdf['QT'],dtype=object)
    tempQM=np.array(tempdf['QM'],dtype=object)
    tempSivErr=np.array(tempdf['tot_err'])
    tempSivTh=np.array(tempdf['Siv'])
    data_dictionary={"Dependence":[],"x1":[],"x2":[],"xF":[],"QT":[],"QM":[],"Siv":[],"tot_err":[]}
    #data_dictionary["Siv_mod"]=tempMod
    data_dictionary["Dependence"]=tempDEP
    data_dictionary["x1"]=tempX1
    data_dictionary["x2"]=tempX2
    data_dictionary["xF"]=tempXF
    data_dictionary["QT"]=tempQT
    data_dictionary["QM"]=tempQM
    data_dictionary["tot_err"]=tempSivErr
    data_dictionary["Siv"]=np.random.normal(tempSivTh, tempSivErr)
    return pd.DataFrame(data_dictionary)
    


def run_replica(i):
    job_number = sys.argv[1]
    replica_number = 10*int(sys.argv[1]) + i
    #replica_number = 9999
    NN_Pseudo_DY = GenDYReplicaData(df)
    NN_Pseudo_DY.to_csv(str(DY_Repl_Folder)+'/rep'+str(replica_number)+'.csv')
    #print(NN_Pseudo_DY)
    
    DY_datann = DY_DataANN()
    Kins_DY, X_DY, y_DY, err_DY = DY_datann.makeData(NN_Pseudo_DY, ['x1', 'x2', 'xF', 'QT', 'QM'])

    #print(len(X_DY),len(y_DY),len(err_DY))
    # #tempdf["Siv_Rep"]= y_DY
        
    trn_X, tst_X, trn_y, tst_y, trn_err, tst_err = trn_tst(X_DY, y_DY, err_DY)

    #print(trn_y)
    sivModel_DY = createModel_DY()
    sivModel_DY.compile(
        optimizer = tf.keras.optimizers.Adam(Learning_Rate),
        loss = tf.keras.losses.MeanSquaredError()
        )
    #, callbacks=[modify_LR,EarlyStop]
    history = sivModel_DY.fit(trn_X, trn_y, sample_weight=(1/trn_err**2), validation_data=(tst_X, tst_y), epochs=EPOCHS, callbacks=[modify_LR], batch_size=300, verbose=2)
    #sivModel.fit(trn_X, trn_y, sample_weight=(1/trn_err**2), validation_data=(tst_X, tst_y), epochs=EPOCHS, verbose=2)
    
    sivModel_DY.save(save_path +'rep'+ str(replica_number) + '.h5', save_format='h5')
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
 

#run_replica()
if __name__ == '__main__':
    jobs = []
    for i in range(10):
        p = multiprocessing.Process(target=run_replica, args=(i,))
        jobs.append(p)
        p.start()

