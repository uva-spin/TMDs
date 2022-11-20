import tensorflow as tf
import pandas as pd
import numpy as np
import lhapdf
import matplotlib.pyplot as plt
#import functions_develop
#import natsort
import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

Models_folder = './SIDIS_Models'
folders_array=os.listdir(Models_folder)
#folders_array=natsort.natsorted(folders_array)
OutputFolder='./NN_SIDIS_Fit_Results'
#os.mkdir(OutputFolder)
numreplicas_SIDIS=len(folders_array)
#numreplicas_SIDIS=100
print(numreplicas_SIDIS)

#Org_Data_path = './PseudoData_Parm2/'


def calc_yhat(model, X):
    return model.predict(X)


class A0(tf.keras.layers.Layer):
    def __init__(self, kperp2avg=.57, pperp2avg=.12, **kwargs):
        super(A0, self).__init__(name='a0')
        self.m1 = tf.Variable(7., name='m1')
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
        ks2avg = (self.kperp2avg*(self.m1**2))/(self.m1**2 + self.kperp2avg) #correct
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


SIDISmodelsArray = []
for i in range(numreplicas_SIDIS):
    testmodel = tf.keras.models.load_model(str(Models_folder)+'/' + str(folders_array[i]),custom_objects={'A0': A0, 'Quotient': Quotient})
    SIDISmodelsArray.append(testmodel)
    
SIDISmodelsArray = np.array(SIDISmodelsArray)

class SIDISDataANN(object):
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

        self.ffDict = {0: self.ffDataPIp,
                       1: self.ffDataPIm,
                       2: self.ffDataPIsum,
                       3: self.ffDataKAp,
                       4: self.ffDataKAm}

    def pdf(self, flavor, x, QQ):
        pdf_val = np.array([self.pdfData.xfxQ2(flavor, ax, qq) for ax, qq in zip(x, QQ)])
        return tf.constant(pdf_val)

    
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
    
SIDISdatann = SIDISDataANN()


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
    
kperp_vals=np.array(list(range(150)))/100
kperp_vals=tf.constant(kperp_vals)    
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
   
# def xsivdist(model, x, QQ, kperp2avg, flavor, kperp):
#     refDict = {-3: 'nnsbar',
#               -2: 'nnubar',
#               -1: 'nndbar',
#               1: 'nnd',
#               2: 'nnu',
#               3: 'nns'}
#     nnqval = nnq(model, np.array([x]), refDict[flavor])
#     #nnqval = nnq(model , np.array([x]), refDict[flavor])[:,0] np.array([x])
#     hval = h(model, kperp)
#     fqpval = fqp([x], [QQ], kperp2avg, kperp, flavor)
#     return ((2*nnqval*hval*fqpval)[0, :])



def xsivdistFromReplicas(numReplicas, x, QQ, kperp2avg, kperp):
    tempfu = []
    tempfd = []
    tempfs = []
    tempfubar = []
    tempfdbar = []
    tempfsbar = []
    for i in range(numReplicas):
        #t = tf.keras.models.load_model(Models_folder+'/'+ str(folders_array[i]), 
        #                                 custom_objects={'A0': A0, 'Quotient': Quotient})
        t = SIDISmodelsArray[i]
        tempfu.append(list(xsivdist(t, x, QQ, kperp2avg, 2, kperp)))
        tempfd.append(list(xsivdist(t, x, QQ, kperp2avg, 1, kperp)))
        tempfs.append(list(xsivdist(t, x, QQ, kperp2avg, 3, kperp)))
        tempfubar.append(list(xsivdist(t, x, QQ, kperp2avg, -2, kperp)))
        tempfdbar.append(list(xsivdist(t, x, QQ, kperp2avg, -1, kperp)))
        tempfsbar.append(list(xsivdist(t, x, QQ, kperp2avg, -3, kperp)))
    return np.array(tempfu),np.array(tempfubar),np.array(tempfd),np.array(tempfdbar),np.array(tempfs),np.array(tempfsbar)

    
    
def SivDistBandsCSVgen(numReplicas, x, QQ, kperp2avg, kperp, numSigma=1):
    data_dictionary={"kperp":[],"fu":[],"fuErr":[],"fubar":[],"fubarErr":[],"fd":[],"fdErr":[],"fdbar":[],"fdbarErr":[],"fs":[],"fsErr":[],"fsbar":[],"fsbarErr":[]}
    results = xsivdistFromReplicas(numReplicas, x, QQ, kperp2avg, kperp)
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
    
# kperp_vals=np.array(list(range(150)))/100
# kperp_vals=tf.constant(kperp_vals)
#print(kperp_vals)
import time
start_time = time.time()
fSivCSV = SivDistBandsCSVgen(numreplicas_SIDIS, 0.1, 2.4, 0.57, kperp_vals)
fSivCSV.to_csv(str(OutputFolder)+'/'+'Sivfuncs.csv')
print("--- %s seconds ---" % (time.time() - start_time))

