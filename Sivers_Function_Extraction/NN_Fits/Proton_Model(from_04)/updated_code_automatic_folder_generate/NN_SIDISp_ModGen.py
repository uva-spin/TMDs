import os
import tensorflow as tf
import pandas as pd
import numpy as np
import lhapdf
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import csv
import sys
#import keras_tuner as kt


# Check if the folder exists
def create_folders(folder_name):
    if not os.path.exists(folder_name):
        # Create the folder
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully.")
    else:
        print(f"Folder '{folder_name}' already exists.")


create_folders('SIDIS_Models')
create_folders('SIDIS_losses')
create_folders('SIDIS_loss_Plots')
create_folders('Sivers_kp_dep')
create_folders('Sivers_x_dep')
create_folders('Sivers_Asym')
create_folders('Sivers_DY_Asym')
create_folders('NN_SIDIS_3DAsym_Proj')


save_path = './SIDIS_Models/rep'
Grids3d_path = '../Gen_3DGrids/'
Org_Data_path = '../Data/'

HERMES2009 = 'HERMES_p_2009.csv'
HERMES2020 = 'HERMES_p_2020.csv'
COMPASS2009 = 'COMPASS_d_2009.csv'
COMPASS2015 = 'COMPASS_p_2015.csv'

SQ_DY_DATA = 'SpinQuest.csv'

#SIDIS_Repl_Folder = '/project/UVA-Spin/Ishara/Sivers/PseudoDataNNFit_300rep_LR_0000110/Replica_Data'
Losses_Folder = './SIDIS_losses'
Losses_Plots_Folder ='./SIDIS_loss_Plots'
Sivers_kp_Folder = './Sivers_kp_dep'
Sivers_x_Folder = './Sivers_x_dep'
Sivers_Asym_Folder = './Sivers_Asym'
Sivers_3D_Asym_Folder = './NN_SIDIS_3DAsym_Proj'
Sivers_DY_Asym_Folder = './Sivers_DY_Asym'

herm09 = pd.read_csv(Org_Data_path + HERMES2009).dropna(axis=0, how='all').dropna(axis=1, how='all')
herm20 = pd.read_csv(Org_Data_path + HERMES2020).dropna(axis=0, how='all').dropna(axis=1, how='all')
comp09 = pd.read_csv(Org_Data_path + COMPASS2009).dropna(axis=0, how='all').dropna(axis=1, how='all')
comp15 = pd.read_csv(Org_Data_path + COMPASS2015).dropna(axis=0, how='all').dropna(axis=1, how='all')


herm09_3D_DATA = pd.read_csv(Grids3d_path+'HERMES2009Grid.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
herm20_3D_DATA = pd.read_csv(Grids3d_path+'HERMES2020Grid.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
comp09_3D_DATA = pd.read_csv(Grids3d_path+'COMPASS2009Grid.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
comp15_3D_DATA = pd.read_csv(Grids3d_path+'COMPASS2015Grid.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')

#df = pd.concat([comp09])
df = pd.concat([herm09,herm20,comp15])

comp17_DY_DATA = pd.read_csv('../Data/COMPASS_p_DY_2017.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
COMPASS_DY2017 = pd.concat([comp17_DY_DATA])

SQ_DY_DATA = pd.read_csv(Org_Data_path + SQ_DY_DATA).dropna(axis=0, how='all').dropna(axis=1, how='all')


JLab11_DATA = pd.read_csv('../Data/JLAB_2011_neutron.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
JLAB2011 = pd.concat([JLab11_DATA])

# Hidden_Layers=2
# Nodes_per_HL=256
# Learning_Rate = 0.00001
# EPOCHS = 300
# L1_reg = 10**(-12)
# BATCHSIZE = 50
# ACTIVATION = 'relu6'
# modify_LR = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',factor=0.9,patience=200,mode='auto')
# EarlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', min_delta=0, restore_best_weights = "False", patience=1000)
# uncert_frac = 1

Hidden_Layers=7
Nodes_per_HL=550
# Learning_Rate = 0.00001
Learning_Rate = 0.00005
EPOCHS = 400
# EPOCHS = 500
L1_reg = 10**(-12)
BATCHSIZE = 300
ACTIVATION = 'relu6'
modify_LR = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',factor=0.9,patience=200,mode='auto')
EarlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', min_delta=0, restore_best_weights = "False", patience=1000)
uncert_frac = 1

validation_fraction = 0.0

#Nreplicas = 100

def create_nn_model(name, hidden_layers=Hidden_Layers, width=Nodes_per_HL, activation=ACTIVATION):
    inp = tf.keras.Input(shape=(1))
    initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=None)
    x = tf.keras.layers.Dense(width, activation=activation, kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg), activity_regularizer=tf.keras.regularizers.L1(L1_reg))(inp)
    for i in range(hidden_layers-1):
        x = tf.keras.layers.Dense(width, activation=activation, kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg), activity_regularizer=tf.keras.regularizers.L1(L1_reg))(x)
    nnout = tf.keras.layers.Dense(1, kernel_initializer = initializer)(x)
    mod = tf.keras.Model(inp, nnout, name=name)
    return mod

def chisquare(y, yhat, err):
    return np.sum(((y - yhat)/err)**2)

        
class A0(tf.keras.layers.Layer):
    def __init__(self, kperp2avg=.57, pperp2avg=.12, **kwargs):
        super(A0, self).__init__(name='a0')
        self.m1 = tf.Variable(5., name='m1')
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


       
class SIDISDataANN(object):
    # def __init__(self, pdfset='cteq61',
    #              ff_PIp='NNFF10_PIp_nlo', ff_PIm='NNFF10_PIm_nlo', ff_PIsum='NNFF10_PIsum_nlo',
    #              ff_KAp='NNFF10_KAp_nlo', ff_KAm='NNFF10_KAm_nlo'):
    def __init__(self, pdfset='NNPDF40_nlo_as_01180',
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
            #yy = sliced['y']
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
            #kins['y']+= list(yy)
            kins['z']+= list(z)
            kins['phT']+= list(phTs)
            kins['1D_dependence']+= list(deps)


        for key in data.keys():
            data[key] = np.array(data[key])
            
        for key in kins.keys():
            kins[key] = np.array(kins[key])

        return kins, data, data[dependencies[0]], np.array(y), np.array(err)
    

SIDISdataann = SIDISDataANN()

def GenSIDISReplicaData(datasetdf):
    data_dictionary = {'hadron':[],
                      'Q2': [],
                      'x': [],
                      'z': [],
                      'phT': [],
                      'Siv':[],
                      'tot_err': [],
                      '1D_dependence': []}
    # TempYhat = np.array(datasetdf['Siv'])
    # Yerr = np.array(datasetdf['tot_err'])
    TempYhat = np.array(datasetdf['x'])
    Yerr = np.array(datasetdf['x'])
    Yhat = np.random.normal(TempYhat, uncert_frac*Yerr)
    data_dictionary['hadron']= np.array(datasetdf['hadron'])
    data_dictionary['Q2']= np.array(datasetdf['Q2'])
    data_dictionary['x']= np.array(datasetdf['x'])
    #data_dictionary['y']= np.array(datasetdf['y'])
    data_dictionary['z']= np.array(datasetdf['z'])
    data_dictionary['phT']= np.array(datasetdf['phT'])
    data_dictionary['1D_dependence']= np.array(datasetdf['1D_dependence'])
    data_dictionary['Siv']= np.array(Yhat)
    data_dictionary['tot_err']= Yerr
    return pd.DataFrame(data_dictionary)
    

#################################################################################
###################  Viewing the model points ###################################
#################################################################################

def calc_yhat(model, X):
    return model.predict(X)


def SiversAsym(df,model,hadron,dependence):
    T_Kins, T_Xplt, T_DEP, T_yplt, T_errplt = SIDISdataann.makeData(df, [hadron], [dependence])
    results = list(calc_yhat(model, T_Xplt))
    #print(results)
    return np.array(results)

def GenReplicaResult(datasetdf, model):
    data_dictionary = {'hadron':[],
                      'Q2': [],
                      'x': [],
                      'z': [],
                      'phT': [],
                      'Siv':[],
                      'tot_err': [],
                      '1D_dependence': []}
    temp_hads = pd.unique(datasetdf['hadron'])
    temp_deps = pd.unique(datasetdf['1D_dependence'])
    SiversA = []
    SiversA_err = []
    for i in temp_hads:
        for j in temp_deps:
            T_Kins, T_Xplt, T_DEP, T_yplt, T_errplt = SIDISdataann.makeData(datasetdf, [str(i)], [str(j)])
            #T_Kins, T_Xplt, T_DEP, T_yplt, T_errplt = datann.makeData(datasetdf, ['pi+'], ['x'])
            results = SiversAsym(datasetdf, model,i, j)
            Yhat = results.flatten()
            data_dictionary['hadron']+= list(T_Kins['hadron'])
            data_dictionary['Q2']+= list(T_Kins['Q2'])
            data_dictionary['x']+= list(T_Kins['x'])
            #data_dictionary['y']+= list(T_Kins['y'])
            data_dictionary['z']+= list(T_Kins['z'])
            data_dictionary['phT']+= list(T_Kins['phT'])
            data_dictionary['1D_dependence']+= list(T_Kins['1D_dependence'])
            #data_dictionary['Siv']+= list(T_yplt)
            data_dictionary['tot_err']+= list(T_errplt)
            data_dictionary['Siv'] += list(Yhat)
            #print(Yhat)
    return pd.DataFrame(data_dictionary)
            
    
    
###############################################################################


def trn_tst(X, y, err, split=validation_fraction):
    tstidxs = np.random.choice(list(range(len(y))), size=int(len(y)*split), replace=False)
    
    tst_X = {k: v[tstidxs] for k, v in X.items()}
    trn_X = {k: np.delete(v, tstidxs) for k, v in X.items()}
    
    tst_y = y[tstidxs]
    trn_y = np.delete(y, tstidxs)
    
    tst_err = err[tstidxs]
    trn_err = np.delete(err, tstidxs)
    
    return trn_X, tst_X, trn_y, tst_y, trn_err, tst_err



def nnq(model, x, hadronstr):
    if not hadronstr in ['nnu', 'nnd', 'nns', 'nnubar', 'nndbar', 'nnsbar']:
        raise Exception('hadronstr must be one of nnu, nnd, nns, nnubar, nndbar, nnsbar')
    mod_out = tf.keras.backend.function(model.get_layer(hadronstr).input,
                                       model.get_layer(hadronstr).output)
    return mod_out(x)


#######################################################################
########## Sivers Function ############################################
#######################################################################



def h(model, kperp):
    m1 = model.get_layer('a0').m1.numpy()
    e = model.get_layer('a0').e.numpy()
    return tf.sqrt(2*e) * (kperp/m1) * tf.math.exp(-kperp**2/m1**2)


def fqp(x, QQ, kperp2avg, kperp, flavor):
    #had = functions_new.Hadron()
    #had = DY_Hadron()
    had = SIDISDataANN()
    #fq = had.pdf_DY(flavor, x, QQ)
    fq = had.pdf(flavor, x, QQ)
    return fq*(1/(np.pi*kperp2avg))*np.exp(-kperp**2/kperp2avg)


#kperp_vals=np.array(list(range(150)))/500
kperp_vals=np.array(np.linspace(0.001,1.5,150))
#kperp_vals=tf.constant(kperp_vals)
fqpu = fqp([0.1], [2.4], 0.57, kperp_vals, 2)
fqpd = fqp([0.1], [2.4], 0.57, kperp_vals, 1)
fqps = fqp([0.1], [2.4], 0.57, kperp_vals, 3)
fqpubar = fqp([0.1], [2.4], 0.57, kperp_vals, -2)
fqpdbar = fqp([0.1], [2.4], 0.57, kperp_vals, -1)
fqpsbar = fqp([0.1], [2.4], 0.57, kperp_vals, -3)



def xsivdist(model, x, QQ, kperp2avg, flavor, kperp):
    refDict = {-3: 'nnsbar',
               -2: 'nnubar',
               -1: 'nndbar',
               1: 'nnd',
               2: 'nnu',
               3: 'nns'}
    nnqval = nnq(model, np.array([x]), refDict[flavor])
    #nnqval = nnq(model , np.array([x]), refDict[flavor])[:,0] np.array([x])
    hval = h(model, kperp)
    if(flavor == -3):
        fqpval = fqpsbar
    if(flavor == -2):
        fqpval = fqpubar
    if(flavor == -1):
        fqpval = fqpdbar
    if(flavor == 1):
        fqpval = fqpd
    if(flavor == 2):
        fqpval = fqpu
    if(flavor == 3):
        fqpval = fqps
    #fqpval = fqp([x], [QQ], kperp2avg, kperp, flavor)
    return ((2*nnqval*hval*fqpval)[0, :])


def xsivdistFromReplicas(model, x, QQ, kperp2avg, kperp):
    tempfu = []
    tempfd = []
    tempfs = []
    tempfubar = []
    tempfdbar = []
    tempfsbar = []
    t = model
    tempfu.append(list(xsivdist(t, x, QQ, kperp2avg, 2, kperp)))
    tempfd.append(list(xsivdist(t, x, QQ, kperp2avg, 1, kperp)))
    tempfs.append(list(xsivdist(t, x, QQ, kperp2avg, 3, kperp)))
    tempfubar.append(list(xsivdist(t, x, QQ, kperp2avg, -2, kperp)))
    tempfdbar.append(list(xsivdist(t, x, QQ, kperp2avg, -1, kperp)))
    tempfsbar.append(list(xsivdist(t, x, QQ, kperp2avg, -3, kperp)))
    return np.array(tempfu),np.array(tempfubar),np.array(tempfd),np.array(tempfdbar),np.array(tempfs),np.array(tempfsbar)


def SivDistBandsCSVgen(model, x, QQ, kperp2avg, kperp, numSigma=1):
    data_dictionary={"kperp":[],"fu":[],"fuErr":[],"fubar":[],"fubarErr":[],"fd":[],"fdErr":[],"fdbar":[],"fdbarErr":[],"fs":[],"fsErr":[],"fsbar":[],"fsbarErr":[]}
    results = xsivdistFromReplicas(model, x, QQ, kperp2avg, kperp)
    tempfu = results[0].mean(axis=0)
    tempfuErr = results[0].std(axis=0)
    tempfubar = results[1].mean(axis=0)
    tempfubarErr = results[1].std(axis=0)
    tempfd = results[2].mean(axis=0)
    tempfdErr = results[2].std(axis=0)
    tempfdbar = results[3].mean(axis=0)
    tempfdbarErr = results[3].std(axis=0)
    tempfs = results[4].mean(axis=0)
    tempfsErr = results[4].std(axis=0)
    tempfsbar = results[5].mean(axis=0)
    tempfsbarErr = results[5].std(axis=0)
    kp = np.array(kperp)
    data_dictionary["kperp"]=kp
    data_dictionary["fu"]=tempfu
    data_dictionary["fuErr"]=tempfuErr
    data_dictionary["fubar"]=tempfubar
    data_dictionary["fubarErr"]=tempfubarErr
    data_dictionary["fd"]=tempfd
    data_dictionary["fdErr"]=tempfdErr
    data_dictionary["fdbar"]=tempfdbar
    data_dictionary["fdbarErr"]=tempfdbarErr
    data_dictionary["fs"]=tempfs
    data_dictionary["fsErr"]=tempfsErr
    data_dictionary["fsbar"]=tempfsbar
    data_dictionary["fsbarErr"]=tempfsbarErr
    return pd.DataFrame(data_dictionary)



#######################################################################
########## Sivers Function Moments############################################
#######################################################################

Mp = 0.938

def fqpx(x, QQ, flavor):
    lenX = len(x)
    had = SIDISDataANN()
    fq=[]
    for i in range(0,lenX):
        fq.append(had.pdf(flavor, x[[i]], QQ)[0])
    return np.array(fq)
    
x_vals=np.array(np.linspace(0.001,1,500))
#x_vals=tf.constant(x_vals)
fqpxu = fqpx(x_vals, [2.4], 2)
fqpxd = fqpx(x_vals, [2.4], 1)
fqpxs = fqpx(x_vals, [2.4], 3)
fqpxubar = fqpx(x_vals, [2.4], -2)
fqpxdbar = fqpx(x_vals, [2.4], -1)
fqpxsbar = fqpx(x_vals, [2.4], -3)



def xsivdistX(model, x, QQ, kperp2avg, flavor):
    refDict = {-3: 'nnsbar',
               -2: 'nnubar',
               -1: 'nndbar',
               1: 'nnd',
               2: 'nnu',
               3: 'nns'}
    nnqval = nnq(model, x, refDict[flavor])
    m1 = model.get_layer('a0').m1.numpy()
    e = model.get_layer('a0').e.numpy()
    ff = np.sqrt(e/2)*kperp2avg*(m1**3)/(Mp*((kperp2avg + m1**2)**2))
    if(flavor == -3):
        fqpval = fqpxsbar
    if(flavor == -2):
        fqpval = fqpxubar
    if(flavor == -1):
        fqpval = fqpxdbar
    if(flavor == 1):
        fqpval = fqpxd
    if(flavor == 2):
        fqpval = fqpxu
    if(flavor == 3):
        fqpval = fqpxs
    #fqpval = fqp([x], [QQ], kperp2avg, kperp, flavor)
    return ((nnqval*fqpval*ff)[0, :])



def xsivdistFromReplicasX(model, x, QQ, kperp2avg):
    tempfu = []
    tempfd = []
    tempfs = []
    tempfubar = []
    tempfdbar = []
    tempfsbar = []
    t = model
    tempfu.append(list(xsivdistX(t, x, QQ, kperp2avg, 2)))
    tempfd.append(list(xsivdistX(t, x, QQ, kperp2avg, 1)))
    tempfs.append(list(xsivdistX(t, x, QQ, kperp2avg, 3)))
    tempfubar.append(list(xsivdistX(t, x, QQ, kperp2avg, -2)))
    tempfdbar.append(list(xsivdistX(t, x, QQ, kperp2avg, -1)))
    tempfsbar.append(list(xsivdistX(t, x, QQ, kperp2avg, -3)))
    return np.array(tempfu),np.array(tempfubar),np.array(tempfd),np.array(tempfdbar),np.array(tempfs),np.array(tempfsbar)

    
    
def SivDistBandsCSVgenX(model, x, QQ, kperp2avg):
    data_dictionary={"x":[],"fu":[],"fuErr":[],"fubar":[],"fubarErr":[],"fd":[],"fdErr":[],"fdbar":[],"fdbarErr":[],"fs":[],"fsErr":[],"fsbar":[],"fsbarErr":[]}
    results = xsivdistFromReplicasX(model, x, QQ, kperp2avg)
    tempfu = results[0].mean(axis=0)
    tempfuErr = results[0].std(axis=0)
    tempfubar = results[1].mean(axis=0)
    tempfubarErr = results[1].std(axis=0)
    tempfd = results[2].mean(axis=0)
    tempfdErr = results[2].std(axis=0)
    tempfdbar = results[3].mean(axis=0)
    tempfdbarErr = results[3].std(axis=0)
    tempfs = results[4].mean(axis=0)
    tempfsErr = results[4].std(axis=0)
    tempfsbar = results[5].mean(axis=0)
    tempfsbarErr = results[5].std(axis=0)
    data_dictionary["x"]=x
    data_dictionary["fu"]=tempfu
    data_dictionary["fuErr"]=tempfuErr
    data_dictionary["fubar"]=tempfubar
    data_dictionary["fubarErr"]=tempfubarErr
    data_dictionary["fd"]=tempfd
    data_dictionary["fdErr"]=tempfdErr
    data_dictionary["fdbar"]=tempfdbar
    data_dictionary["fdbarErr"]=tempfdbarErr
    data_dictionary["fs"]=tempfs
    data_dictionary["fsErr"]=tempfsErr
    data_dictionary["fsbar"]=tempfsbar
    data_dictionary["fsbarErr"]=tempfsbarErr
    return pd.DataFrame(data_dictionary)


######################################################################

def SiversAsym(df,model,hadron,dependence):
    T_Kins, T_Xplt, T_DEP, T_yplt, T_errplt = SIDISdataann.makeData(df, [hadron], [dependence])
    results = list(calc_yhat(model, T_Xplt))
    return np.array(results)
    


def GenPseudo(datasetdf, model):
    data_dictionary = {'hadron':[],
                      'Q2': [],
                      'x': [],
                      'z': [],
                      'phT': [],
                      'Siv':[],
                      'tot_err': [],
                      '1D_dependence': []}
    temp_hads = pd.unique(datasetdf['hadron'])
    temp_deps = pd.unique(datasetdf['1D_dependence'])
    SiversA = []
    SiversA_err = []
    for i in temp_hads:
        for j in temp_deps:
            T_Kins, T_Xplt, T_DEP, T_yplt, T_errplt = SIDISdataann.makeData(datasetdf, [str(i)], [str(j)])
            #T_Kins, T_Xplt, T_DEP, T_yplt, T_errplt = datann.makeData(datasetdf, ['pi+'], ['x'])
            results = SiversAsym(datasetdf, model,i, j)
            Yhat = results.flatten()
            data_dictionary['hadron']+= list(T_Kins['hadron'])
            data_dictionary['Q2']+= list(T_Kins['Q2'])
            data_dictionary['x']+= list(T_Kins['x'])
            #data_dictionary['y']+= list(T_Kins['y'])
            data_dictionary['z']+= list(T_Kins['z'])
            data_dictionary['phT']+= list(T_Kins['phT'])
            data_dictionary['1D_dependence']+= list(T_Kins['1D_dependence'])
            #data_dictionary['Siv']+= list(T_yplt)
            data_dictionary['tot_err']+= list(T_errplt)
            data_dictionary['Siv'] += list(Yhat)
            #print(Yhat)
    return pd.DataFrame(data_dictionary)


##########################################################################    
###################### DY Definitions ####################################
##########################################################################

class DY_Hadron(object):
    def __init__(self, kperp2avg=.25, pperp2avg=.12, pdfset='NNPDF40_nlo_as_01180'):
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
        return (topfirst/bottomfirst) * tf.exp(-exptop/expbottom) * last
    


class Sivers_DY(DY_Hadron):
    def __init__(self, kperp2avg=.25, pperp2avg=.12, pdfset='NNPDF40_nlo_as_01180'):
        
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



def Project_COMPASS_DY(tempdf, model, sign):
    ##############################
    tempDEP=np.array(tempdf['Dependence'],dtype=object)
    tempX1=np.array(tempdf['x1'],dtype=object)
    tempX2=np.array(tempdf['x2'],dtype=object)
    tempXF=np.array(tempdf['xF'],dtype=object)
    tempQT=np.array(tempdf['QT'],dtype=object)
    tempQM=np.array(tempdf['QM'],dtype=object)
    #tempSivErr=np.array(tempdf['tot_err'],dtype=object)
    #tempSivTh=np.array(tempdf['Siv'],dtype=object)
    data_dictionary={"Dependence":[],"x1":[],"x2":[],"xF":[],"QT":[],"QM":[],"Siv":[],"tot_err":[]}
    data_dictionary["Dependence"]=tempDEP
    data_dictionary["x1"]=tempX1
    data_dictionary["x2"]=tempX2
    data_dictionary["xF"]=tempXF
    data_dictionary["QT"]=tempQT
    data_dictionary["QM"]=tempQM
    #data_dictionary["tot_err"]=tempSivErr
    SivDY=Sivers_DY()
    ############################################
    temp_Siv_Mean=[]
    temp_Siv_Std=[]
    for i in range(len(tempDEP)):
        temp=np.array([[data_dictionary["x1"][i],data_dictionary["x2"][i],
                        data_dictionary["QT"][i],data_dictionary["QM"][i]]])
        tem_Siv_model_val = SivDY.sivers(model,temp, sign)[0]
        temp_mod_array=np.array(tem_Siv_model_val)
        temp_Siv_Mean.append(np.mean(temp_mod_array))
        temp_Siv_Std.append(np.std(temp_mod_array))            
    ############################################
    #data_dictionary["Siv"]=np.array(tempSivTh)
    # data_dictionary["Siv"]=np.array(temp_Siv_Mean)
    # data_dictionary["tot_err"]=np.array(temp_Siv_Std)
    DY_sign = -1
    data_dictionary["Siv"]=DY_sign*np.array(temp_Siv_Mean)
    data_dictionary["tot_err"]=np.array(temp_Siv_Std)
    return pd.DataFrame(data_dictionary)
    #return tempDEP
    

        
def run_replica():
    #replica_number = i
    replica_number = sys.argv[1]
    #replica_number = 9998 #This is for testing purposes
    herm09_gen = GenSIDISReplicaData(herm09)
    herm20_gen = GenSIDISReplicaData(herm20)
    #comp09_gen = GenSIDISReplicaData(comp09)
    comp15_gen = GenSIDISReplicaData(comp15)
    tempdf = pd.concat([herm09_gen, herm20_gen, comp15_gen])
    #########
    herm09_3D_gen = GenSIDISReplicaData(herm09_3D_DATA)
    herm20_3D_gen = GenSIDISReplicaData(herm20_3D_DATA)
    comp09_3D_gen = GenSIDISReplicaData(comp09_3D_DATA)
    comp15_3D_gen = GenSIDISReplicaData(comp15_3D_DATA)
    #tempdf = pd.concat([comp09_gen])
    #print("Here printing pseudo data")
    #tempdf.to_csv(str(SIDIS_Repl_Folder)+'/rep_'+str(replica_number)+'.csv')

    T_Kins, T_Xplt, T_DEP, T_yplt, T_errplt = SIDISdataann.makeData(tempdf, ['pi+', 'pi-', 'pi0', 'k+', 'k-'], ['x', 'z', 'phT'])
    
    trn_X, tst_X, trn_y, tst_y, trn_err, tst_err = trn_tst(T_Xplt, T_yplt, T_errplt)

    sivModel = SIDIS_Model()
    sivModel.compile(
        optimizer = tf.keras.optimizers.Adam(Learning_Rate),
        loss = tf.keras.losses.MeanSquaredError()
        )
    #, callbacks=[modify_LR,EarlyStop],EarlyStop
    #history = sivModel.fit(trn_X, trn_y, sample_weight=(1/trn_err**2), validation_data=(tst_X, tst_y), epochs=EPOCHS, callbacks=[modify_LR], batch_size=300, verbose=2)
    history = sivModel.fit(trn_X, trn_y, validation_data=(tst_X, tst_y), epochs=EPOCHS, callbacks=[modify_LR], batch_size=BATCHSIZE, verbose=2, shuffle=True)
    sivModel.save(save_path + str(replica_number) + '.h5', save_format='h5')

    tempdfF = pd.DataFrame()
    tempdfF["Train_Loss"] = history.history['loss']
    #tempdfF["Val_Loss"] = history.history['val_loss']
    tempdfF.to_csv(str(Losses_Folder)+'/reploss_'+str(replica_number)+'.csv')
    # [-100:]
    plt.figure(5)
    plt.plot(history.history['loss'],'-', color = 'blue', label = 'Training Loss')
    #plt.plot(history.history['val_loss'],'-', color = 'orange', label = 'Validation Loss')
    #plt.xlim([10000,15000])
    #plt.ylim([0,5])
    plt.ylabel('Loss',fontsize=15)
    plt.xlabel('Epoch',fontsize=15)
    #plt.legend(loc=1,fontsize=15,handlelength=3)
    plt.savefig(str(Losses_Plots_Folder)+'/losses'+str(replica_number)+'.pdf')
    #plt.savefig(str(Losses_Folder)+'/Plos'+'/losses.pdf')
    ###########################
    fSivCSV_kp = SivDistBandsCSVgen(sivModel, 0.1, 2.4, 0.57, kperp_vals)
    fSivCSV_kp.to_csv(str(Sivers_kp_Folder)+'/Sivfuncs_Kdep_'+str(replica_number)+'.csv')
    fSivCSV_x = SivDistBandsCSVgenX(sivModel, x_vals, 2.4, 0.57)
    fSivCSV_x.to_csv(str(Sivers_x_Folder)+'/Sivfuncs_Xdep_'+str(replica_number)+'.csv')
    AsymSiv_H09_CSV = GenPseudo(herm09_gen, sivModel)
    AsymSiv_H09_CSV.to_csv(str(Sivers_Asym_Folder)+'/Siv_Asym_H09_'+str(replica_number)+'.csv')
    AsymSiv_H20_CSV = GenPseudo(herm20_gen, sivModel)
    AsymSiv_H20_CSV.to_csv(str(Sivers_Asym_Folder)+'/Siv_Asym_H20_'+str(replica_number)+'.csv')
    AsymSiv_C20_CSV = GenPseudo(comp15_gen, sivModel)
    AsymSiv_C20_CSV.to_csv(str(Sivers_Asym_Folder)+'/Siv_Asym_C15_'+str(replica_number)+'.csv')
    #################################################
    AsymSiv_H09_3D_CSV = GenPseudo(herm09_3D_gen, sivModel)
    AsymSiv_H09_3D_CSV.to_csv(str(Sivers_3D_Asym_Folder)+'/Siv_3D_Asym_H09_'+str(replica_number)+'.csv')
    AsymSiv_H20_3D_CSV = GenPseudo(herm20_3D_gen, sivModel)
    AsymSiv_H20_3D_CSV.to_csv(str(Sivers_3D_Asym_Folder)+'/Siv_3D_Asym_H20_'+str(replica_number)+'.csv')
    AsymSiv_C20_3D_CSV = GenPseudo(comp15_3D_gen, sivModel)
    AsymSiv_C20_3D_CSV.to_csv(str(Sivers_3D_Asym_Folder)+'/Siv_3D_Asym_C15_'+str(replica_number)+'.csv')
    #################################################
    NN_SQ_DY_from_SIDIS_minus=Project_COMPASS_DY(SQ_DY_DATA, sivModel,-1)
    NN_SQ_DY_from_SIDIS_minus.to_csv(str(Sivers_DY_Asym_Folder)+'/Result_SpinQuest_DY_from_SIDIS_minus_'+str(replica_number)+'.csv')    
    NN_SQ_DY_from_SIDIS_plus=Project_COMPASS_DY(SQ_DY_DATA, sivModel, 1)
    NN_SQ_DY_from_SIDIS_plus.to_csv(str(Sivers_DY_Asym_Folder)+'/Result_SpinQuest_DY_from_SIDIS_plus_'+str(replica_number)+'.csv')
    NN_COMPASS_DY_from_SIDIS_minus=Project_COMPASS_DY(COMPASS_DY2017, sivModel,-1)
    NN_COMPASS_DY_from_SIDIS_minus.to_csv(str(Sivers_DY_Asym_Folder)+'/Result_COMPASS_DY2017_DY_from_SIDIS_minus_'+str(replica_number)+'.csv')    
    NN_COMPASS_DY_from_SIDIS_plus=Project_COMPASS_DY(COMPASS_DY2017, sivModel, 1)
    NN_COMPASS_DY_from_SIDIS_plus.to_csv(str(Sivers_DY_Asym_Folder)+'/Result_COMPASS_DY2017_DY_from_SIDIS_plus_'+str(replica_number)+'.csv')
    # NN_SIDIS_JLAB2011=GenPseudo(JLAB2011, sivModel)
    # NN_SIDIS_JLAB2011.to_csv(str(Sivers_Asym_Folder)+'/Result_SIDIS_JLAB2011_'+str(replica_number)+'.csv') COMPASS_DY2017

    #Sivers_Asym_Folder


#for i in range(0,1):
run_replica()
