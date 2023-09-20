#import cudf
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
OutputFolder='NN_SIDIS_3DAsym_Proj'
#os.mkdir(OutputFolder)
numreplicas_SIDIS=len(folders_array)
#numreplicas_SIDIS=5
print(numreplicas_SIDIS)

Org_Data_path = '../Gen_3DGrids/'

#cudf.DataFrame
# herm9_DATA = cudf.read_csv(str(Org_Data_path)+'HERMES2009_Pseudo.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# herm20_DATA = cudf.read_csv(str(Org_Data_path)+'HERMES2020_Pseudo.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# comp9_DATA = cudf.read_csv(str(Org_Data_path)+'COMPASS2009_Pseudo.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# comp15_DATA = cudf.read_csv(str(Org_Data_path)+'COMPASS2015_Pseudo.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
herm9_DATA = pd.read_csv(str(Org_Data_path)+'HERMES2009Grid.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
herm20_DATA = pd.read_csv(str(Org_Data_path)+'HERMES2020Grid.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
comp9_DATA = pd.read_csv(str(Org_Data_path)+'COMPASS2009Grid.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
comp15_DATA = pd.read_csv(str(Org_Data_path)+'COMPASS2015Grid.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')


#comp17_DY_DATA = pd.read_csv('../Data/COMPASS_p_DY_2017.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')


HERMES2009 = pd.concat([herm9_DATA])
HERMES2020 = pd.concat([herm20_DATA])
COMPASS2009 = pd.concat([comp9_DATA])
COMPASS2015 = pd.concat([comp15_DATA])


SIDISdf = pd.concat([HERMES2009,HERMES2020,COMPASS2009,COMPASS2015])

#COMPASS_DY2017 = pd.concat([comp17_DY_DATA])



def calc_yhat(model, X):
    return model.predict(X)


class A0(tf.keras.layers.Layer):
    def __init__(self, kperp2avg=.57, pperp2avg=.12, **kwargs):
        super(A0, self).__init__(name='a0')
        self.m1 = tf.Variable(5., name='m1')
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
               'z': [],
               'phT': [],
               '1D_dependence': []}

        #y = []
        #err = []
        
        hads = []
        QQs =[]
        #yy = []
        deps = []

        df = df.loc[df['hadron'].isin(hadrons), :]
        df = df.loc[df['1D_dependence'].isin(dependencies), :]
        #X = np.array(df[['x', 'z', 'phT', 'Q2', 'hadron']])
        for i, had in enumerate(['pi+', 'pi-', 'pi0', 'k+', 'k-']):
            sliced = df.loc[df['hadron'] == had, :]
            #y += list(sliced['Siv'])
            #err += list(sliced['tot_err'])

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

        #return kins, data, data[dependencies[0]], np.array(y), np.array(err)
        return kins, data, data[dependencies[0]]
    
SIDISdatann = SIDISDataANN()


def SiversAsym(df,numReplicas,hadron,dependence):
    T_Kins, T_Xplt, T_DEP= SIDISdatann.makeData(df, [hadron], [dependence])
    results = []
    for i in range(numReplicas):
        #folders_array
        #testmodel = tf.keras.models.load_model(str(Models_folder)+'/' + str(folders_array[i]),custom_objects={'A0': A0, 'Quotient': Quotient})
        #testmodel = tf.keras.models.load_model(str(Models_folder)+'/rep' + str(i) + '.h5',custom_objects={'A0': A0, 'Quotient': Quotient})
        testmodel = SIDISmodelsArray[i]
        results.append(list(calc_yhat(testmodel, T_Xplt)))
    return np.array(results)



def GenPseudo(datasetdf, numReplicas, numSigma=1):
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
    #print(temp_deps)
    SiversA = []
    SiversA_err = []
    for i in temp_hads:
        for j in temp_deps:
            T_Kins, T_Xplt, T_DEP = SIDISdatann.makeData(datasetdf, [str(i)], [str(j)])
            #T_Kins, T_Xplt, T_DEP, T_yplt, T_errplt = datann.makeData(datasetdf, ['pi+'], ['x'])
            results = SiversAsym(datasetdf, numReplicas,i, j)
            Yhat = np.array(results.mean(axis=0))
            Yhat = Yhat.flatten()
            #print(str(i),str(j),len(Yhat),len(T_Kins['x']))
            Yerr = np.array(results.std(axis=0))
            Yerr = Yerr.flatten()
            #print(len(Yerr))
            SiversA.append(Yhat)
            SiversA_err.append(Yerr)
            data_dictionary['hadron']+= list(T_Kins['hadron'])
            data_dictionary['Q2']+= list(T_Kins['Q2'])
            data_dictionary['x']+= list(T_Kins['x'])
            #data_dictionary['y']+= list(T_Kins['y'])
            data_dictionary['z']+= list(T_Kins['z'])
            data_dictionary['phT']+= list(T_Kins['phT'])
            data_dictionary['1D_dependence']+= list(T_Kins['1D_dependence'])
            data_dictionary['Siv']+= list(Yhat)
            data_dictionary['tot_err']+= list(Yerr)
            #print(len(data_dictionary['hadron']),len(data_dictionary['Q2']),len(data_dictionary['x']),len(data_dictionary['z']),len(data_dictionary['phT']))
    #print(np.concatenate(t_hads))
    #data_dictionary['Siv'] = np.concatenate(SiversA)
    #data_dictionary['tot_err'] = np.concatenate(SiversA_err)
    #data_dictionary['Siv_NNFit'] = np.concatenate(SiversA)
    #data_dictionary['Siv_NNStd'] = np.concatenate(SiversA_err)
    #print(len(data_dictionary['hadron']))
    #print(len(data_dictionary['Siv']))
    #print(data_dictionary)
    return pd.DataFrame(data_dictionary)




def nnq(model, x, hadronstr):
    if not hadronstr in ['nnu', 'nnd', 'nns', 'nnubar', 'nndbar', 'nnsbar']:
        raise Exception('hadronstr must be one of nnu, nnd, nns, nnubar, nndbar, nnsbar')
    mod_out = tf.keras.backend.function(model.get_layer(hadronstr).input,
                                       model.get_layer(hadronstr).output)
    return mod_out(x)



    
    
NN_Pseudo_SIDIS_HERMES2009=GenPseudo(HERMES2009, numreplicas_SIDIS, numSigma=1)
NN_Pseudo_SIDIS_HERMES2009.to_csv(str(OutputFolder)+'/Projected_SIDIS_HERMES2009.csv')

NN_Pseudo_SIDIS_HERMES2020=GenPseudo(HERMES2020, numreplicas_SIDIS, numSigma=1)
NN_Pseudo_SIDIS_HERMES2020.to_csv(str(OutputFolder)+'/Projected_SIDIS_HERMES2020.csv')

NN_Pseudo_SIDIS_COMPASS2009=GenPseudo(COMPASS2009, numreplicas_SIDIS, numSigma=1)
NN_Pseudo_SIDIS_COMPASS2009.to_csv(str(OutputFolder)+'/Projected_SIDIS_COMPASS2009.csv')

NN_Pseudo_SIDIS_COMPASS2015=GenPseudo(COMPASS2015, numreplicas_SIDIS, numSigma=1)
NN_Pseudo_SIDIS_COMPASS2015.to_csv(str(OutputFolder)+'/Projected_SIDIS_COMPASS2015.csv')
