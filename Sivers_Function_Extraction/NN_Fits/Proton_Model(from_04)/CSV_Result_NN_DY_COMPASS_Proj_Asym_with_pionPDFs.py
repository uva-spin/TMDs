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
#JAM21PionPDFnlo

Models_folder = './SIDIS_Models'
folders_array=os.listdir(Models_folder)
#folders_array=natsort.natsorted(folders_array)
OutputFolder='./NN_SIDIS_Fit_Results'
#os.mkdir(OutputFolder)
numreplicas_SIDIS=len(folders_array)
#numreplicas_SIDIS=5
print(numreplicas_SIDIS)

#Org_Data_path = '../Data/'

herm9_DATA = pd.read_csv('../Data/HERMES_p_2009.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
herm20_DATA = pd.read_csv('../Data/HERMES_p_2020.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
comp9_DATA = pd.read_csv('../Data/COMPASS_d_2009.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
comp15_DATA = pd.read_csv('../Data/COMPASS_p_2015.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
#cudf.DataFrame
# herm9_DATA = cudf.read_csv(str(Org_Data_path)+'HERMES2009_Pseudo.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# herm20_DATA = cudf.read_csv(str(Org_Data_path)+'HERMES2020_Pseudo.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# comp9_DATA = cudf.read_csv(str(Org_Data_path)+'COMPASS2009_Pseudo.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# comp15_DATA = cudf.read_csv(str(Org_Data_path)+'COMPASS2015_Pseudo.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# herm9_DATA = pd.read_csv(str(Org_Data_path)+'HERMES2009_3DPseudo.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# herm20_DATA = pd.read_csv(str(Org_Data_path)+'HERMES2020_3DPseudo.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# comp9_DATA = pd.read_csv(str(Org_Data_path)+'COMPASS2009_3DPseudo.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# comp15_DATA = pd.read_csv(str(Org_Data_path)+'COMPASS2015_3DPseudo.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')


comp17_DY_DATA = pd.read_csv('../Data/COMPASS_p_DY_2017.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')


HERMES2009 = pd.concat([herm9_DATA])
HERMES2020 = pd.concat([herm20_DATA])
COMPASS2009 = pd.concat([comp9_DATA])
COMPASS2015 = pd.concat([comp15_DATA])


# SIDISdf = pd.concat([HERMES2009,HERMES2020,COMPASS2009,COMPASS2015])
SIDISdf = pd.concat([HERMES2009,HERMES2020,COMPASS2015])

COMPASS_DY2017 = pd.concat([comp17_DY_DATA])



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

        y = []
        err = []
        
        hads = []
        QQs =[]
        #yy = []
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
    
SIDISdatann = SIDISDataANN()


def SiversAsym(df,numReplicas,hadron,dependence):
    T_Kins, T_Xplt, T_DEP, T_yplt, T_errplt = SIDISdatann.makeData(df, [hadron], [dependence])
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
    SiversA = []
    SiversA_err = []
    for i in temp_hads:
        for j in temp_deps:
            T_Kins, T_Xplt, T_DEP, T_yplt, T_errplt = SIDISdatann.makeData(datasetdf, [str(i)], [str(j)])
            #T_Kins, T_Xplt, T_DEP, T_yplt, T_errplt = datann.makeData(datasetdf, ['pi+'], ['x'])
            results = SiversAsym(datasetdf, numReplicas,i, j)
            Yhat = np.array(results.mean(axis=0))
            Yhat = Yhat.flatten()
            Yerr = np.array(results.std(axis=0))
            Yerr = Yerr.flatten()
            SiversA.append(Yhat)
            SiversA_err.append(Yerr)
            data_dictionary['hadron']+= list(T_Kins['hadron'])
            data_dictionary['Q2']+= list(T_Kins['Q2'])
            data_dictionary['x']+= list(T_Kins['x'])
            #data_dictionary['y']+= list(T_Kins['y'])
            data_dictionary['z']+= list(T_Kins['z'])
            data_dictionary['phT']+= list(T_Kins['phT'])
            data_dictionary['1D_dependence']+= list(T_Kins['1D_dependence'])
            data_dictionary['Siv']+= list(T_yplt)
            data_dictionary['tot_err']+= list(T_errplt)
            #print(Yhat)
    #print(np.concatenate(t_hads))
    data_dictionary['Siv'] = np.concatenate(SiversA)
    data_dictionary['tot_err'] = np.concatenate(SiversA_err)
    #data_dictionary['Siv_NNFit'] = np.concatenate(SiversA)
    #data_dictionary['Siv_NNStd'] = np.concatenate(SiversA_err)
    #print(len(data_dictionary['hadron']))
    #print(len(data_dictionary['Siv']))
    #print(data_dictionary)
    return pd.DataFrame(data_dictionary)


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

        DYdata = {'x1': [],
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
        
        DYkins = {'Siv_mod':[],
               'Dependence': [],
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

            sivm = sliced['Siv_mod']
            deps = sliced['Dependence']
            x1 = sliced['x1']
            x2 = sliced['x2']
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


            kins['Siv_mod']+= list(sivm)
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
    



def nnq(model, x, hadronstr):
    if not hadronstr in ['nnu', 'nnd', 'nns', 'nnubar', 'nndbar', 'nnsbar']:
        raise Exception('hadronstr must be one of nnu, nnd, nns, nnubar, nndbar, nnsbar')
    mod_out = tf.keras.backend.function(model.get_layer(hadronstr).input,
                                       model.get_layer(hadronstr).output)
    return mod_out(x)



##########################################################################    
###################### DY Definitions ####################################
##########################################################################

class DY_Hadron(object):
    def __init__(self, kperp2avg=.25, pperp2avg=.12, pdfset='cteq61', pion_pdfset= 'JAM21PionPDFnlo'):
        #self.pdfData = lhapdf.mkPDF(pdfset)
        self.pdf_DYData = lhapdf.mkPDF(pdfset)
        self.pion_pdf_DYData = lhapdf.mkPDF(pion_pdfset)
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
        
    def pion_pdf_DY(self, flavor, x, QM):
        return np.array([self.pion_pdf_DYData.xfxQ(flavor, ax, qq) for ax, qq in zip(x, QM)])
    
    
    def B0(self, qT, m1):
        ks2avg = (self.kperp2avg*m1**2)/(m1**2 + self.kperp2avg)
        topfirst = (self.kperp2avg + self.kperp2avg) * ks2avg**2
        bottomfirst = (ks2avg + self.kperp2avg)**2 * self.kperp2avg
        exptop = -qT**2 / (ks2avg + self.kperp2avg)
        expbottom = -qT**2 / (self.kperp2avg + self.kperp2avg)
        last = np.sqrt(2*self.e) * qT / m1
        return (topfirst/bottomfirst) * tf.exp(-exptop/expbottom) * last
    

# With pion pdfs
class Sivers_DY(DY_Hadron):
    def __init__(self, kperp2avg=.25, pperp2avg=.12, pdfset='cteq61'):
        
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
        temp_top = NNuX1 * self.eu**2 * self.pion_pdf_DY(2, x1, QQ) * self.pdf_DY(-2, x2, QQ)
        + NNuX2 * self.eu**2 * self.pion_pdf_DY(2, x2, QQ) * self.pdf_DY(-2, x1, QQ)
        + NNubarX1 * self.eubar**2 * self.pion_pdf_DY(-2, x1, QQ) * self.pdf_DY(2, x2, QQ)
        + NNubarX2 * self.eubar**2 * self.pion_pdf_DY(-2, x2, QQ) * self.pdf_DY(2, x1, QQ)
        + NNdX1 * self.ed**2 * self.pion_pdf_DY(1, x1, QQ) * self.pdf_DY(-1, x2, QQ)
        + NNdX2 * self.ed**2 * self.pion_pdf_DY(1, x2, QQ) * self.pdf_DY(-1, x1, QQ)
        + NNdbarX1 * self.edbar**2 * self.pion_pdf_DY(-1, x1, QQ) * self.pdf_DY(1, x2, QQ)
        + NNdbarX2 * self.edbar**2 * self.pion_pdf_DY(-1, x2, QQ) * self.pdf_DY(1, x1, QQ)
        + NNsX1 * self.es**2 * self.pion_pdf_DY(3, x1, QQ) * self.pdf_DY(-3, x2, QQ)
        + NNsX2 * self.es**2 * self.pion_pdf_DY(3, x2, QQ) * self.pdf_DY(-3, x1, QQ)
        + NNsbarX1 * self.esbar**2 * self.pion_pdf_DY(-3, x1, QQ) * self.pdf_DY(3, x2, QQ)
        + NNsbarX2 * self.esbar**2 * self.pion_pdf_DY(-3, x2, QQ) * self.pdf_DY(3, x1, QQ)
        temp_bottom = self.eu**2 * self.pion_pdf_DY(2, x1, QQ) * self.pdf_DY(-2, x2, QQ)
        + self.eu**2 * self.pion_pdf_DY(2, x2, QQ) * self.pdf_DY(-2, x1, QQ)
        + self.eubar**2 * self.pion_pdf_DY(-2, x1, QQ) * self.pdf_DY(2, x2, QQ)
        + self.eubar**2 * self.pion_pdf_DY(-2, x2, QQ) * self.pdf_DY(2, x1, QQ)
        + self.ed**2 * self.pion_pdf_DY(1, x1, QQ) * self.pdf_DY(-1, x2, QQ)
        + self.ed**2 * self.pion_pdf_DY(1, x2, QQ) * self.pdf_DY(-1, x1, QQ)
        + self.edbar**2 * self.pion_pdf_DY(-1, x1, QQ) * self.pdf_DY(1, x2, QQ)
        + self.edbar**2 * self.pion_pdf_DY(-1, x2, QQ) * self.pdf_DY(1, x1, QQ)
        + self.es**2 * self.pion_pdf_DY(3, x1, QQ) * self.pdf_DY(-3, x2, QQ)
        + self.es**2 * self.pion_pdf_DY(3, x2, QQ) * self.pdf_DY(-3, x1, QQ)
        + self.esbar**2 * self.pion_pdf_DY(-3, x1, QQ) * self.pdf_DY(3, x2, QQ)
        + self.esbar**2 * self.pion_pdf_DY(-3, x2, QQ) * self.pdf_DY(3, x1, QQ)
        temp_siv_had = b0*((temp_top)/(temp_bottom))
        return temp_siv_had
    


def GenDYKinematicsFromSIDIS(datasetdf):
    SIDISdata_dictionary = {'hadron':[],
                      'Q2': [],
                      'x': [],
                      'z': [],
                      'phT': [],
                      '1D_dependence': []}
    temp_hads = pd.unique(datasetdf['hadron'])
    temp_deps = pd.unique(datasetdf['1D_dependence'])
    DYdata_dictionary = {'x1':[],
                      'x2': [],
                      'xF': [],
                      'QT': [],
                      'QM': [],
                      'Dependence': []}
    for i in temp_hads:
        for j in temp_deps:
            T_Kins, T_Xplt, T_DEP, T_yplt, T_errplt = SIDISdatann.makeData(datasetdf, [str(i)], [str(j)])
            #Yhat = np.random.normal(T_yplt, T_errplt)
            #Yerr = T_errplt
            #DYdata_dictionary['hadron']+= list(T_Kins['hadron'])
            DYdata_dictionary['QM']+= list(np.sqrt(T_Kins['Q2']))
            # DYdata_dictionary['x1']+= list(T_Kins['z'])
            # DYdata_dictionary['x2']+= list(T_Kins['x'])
            DYdata_dictionary['x1']+= list(T_Kins['x'])
            DYdata_dictionary['x2']+= list(T_Kins['z'])
            DYdata_dictionary['xF']+= list(T_Kins['z']-T_Kins['x'])
            DYdata_dictionary['QT']+= list(T_Kins['phT'])
            if T_Kins['1D_dependence'][0] == 'x':
                TDep = ['x1' for k in range(0,len(T_Kins['1D_dependence']))]
            elif T_Kins['1D_dependence'][0] == 'z':
                TDep = ['x2' for k in range(0,len(T_Kins['1D_dependence']))]
            elif T_Kins['1D_dependence'][0] == 'phT':
                TDep = ['QT' for k in range(0,len(T_Kins['1D_dependence']))]
            #DYdata_dictionary['Dependence']+= list(T_Kins['1D_dependence'])
            DYdata_dictionary['Dependence']+= list(TDep)
            #data_dictionary['Siv']+= list(Yhat)
            #data_dictionary['tot_err']+= list(Yerr)
    return pd.DataFrame(DYdata_dictionary)

#GenDYKinematicsFromSIDIS(SIDISdf)

def Project_DY_Data_All_Models(SIDIStempdf, numReplicas, sign):
    ##############################
    tempdf = GenDYKinematicsFromSIDIS(SIDIStempdf)
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
        tem_Siv_model_val=[]
        for j in range(numReplicas):
            #print(data_path+'/'+ models_folder[j])
            #t = tf.keras.models.load_model(str(Models_folder)+'/' + str(folders_array[j]),custom_objects={'A0': A0, 'Quotient': Quotient})
            t = SIDISmodelsArray[j]
            tem_Siv_model_val.append(SivDY.sivers(t,temp, sign)[0])
        temp_mod_array=np.array(tem_Siv_model_val)
        temp_Siv_Mean.append(np.mean(temp_mod_array))
        temp_Siv_Std.append(np.std(temp_mod_array))            
    ############################################
    #data_dictionary["Siv"]=np.array(tempSivTh)
    # data_dictionary["Siv"]=np.array(temp_Siv_Mean)
    # data_dictionary["tot_err"]=np.array(temp_Siv_Std)
    # return pd.DataFrame(data_dictionary)
    #DY_sign = -1
    DY_sign = -1
    data_dictionary["Siv"]=DY_sign*np.array(temp_Siv_Mean)
    data_dictionary["tot_err"]=np.array(temp_Siv_Std)
    return pd.DataFrame(data_dictionary)


##### Projecting COMPASS DY Asymmetries ######

def Project_COMPASS_DY(tempdf, numReplicas, sign):
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
        tem_Siv_model_val=[]
        for j in range(numReplicas):
            #print(data_path+'/'+ models_folder[j])
            #t = tf.keras.models.load_model(str(Models_folder)+'/' + str(folders_array[j]),custom_objects={'A0': A0, 'Quotient': Quotient})
            t = SIDISmodelsArray[j]
            tem_Siv_model_val.append(SivDY.sivers(t,temp, sign)[0])
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
    
    
# NN_Pseudo_SIDIS_HERMES2009=GenPseudo(HERMES2009, numreplicas_SIDIS, numSigma=1)
# NN_Pseudo_SIDIS_HERMES2009.to_csv(str(OutputFolder)+'/Result_SIDIS_HERMES2009.csv')

# NN_Pseudo_SIDIS_HERMES2020=GenPseudo(HERMES2020, numreplicas_SIDIS, numSigma=1)
# NN_Pseudo_SIDIS_HERMES2020.to_csv(str(OutputFolder)+'/Result_SIDIS_HERMES2020.csv')

# NN_Pseudo_SIDIS_COMPASS2009=GenPseudo(COMPASS2009, numreplicas_SIDIS, numSigma=1)
# NN_Pseudo_SIDIS_COMPASS2009.to_csv(str(OutputFolder)+'/Result_SIDIS_COMPASS2009.csv')

# NN_Pseudo_SIDIS_COMPASS2015=GenPseudo(COMPASS2015, numreplicas_SIDIS, numSigma=1)
# NN_Pseudo_SIDIS_COMPASS2015.to_csv(str(OutputFolder)+'/Result_SIDIS_COMPASS2015.csv')

NN_COMPASS_DY_from_SIDIS_minus=Project_COMPASS_DY(COMPASS_DY2017, numreplicas_SIDIS,-1)
NN_COMPASS_DY_from_SIDIS_minus.to_csv(str(OutputFolder)+'/Result_COMPASS_DY_from_SIDIS_minus_with_pionDFs.csv')

NN_COMPASS_DY_from_SIDIS_plus=Project_COMPASS_DY(COMPASS_DY2017, numreplicas_SIDIS, 1)
NN_COMPASS_DY_from_SIDIS_plus.to_csv(str(OutputFolder)+'/Result_COMPASS_DY_from_SIDIS_plus_with_pionDFs.csv')

# NN_Pseudo_DY_from_SIDIS_plus=Project_DY_Data_All_Models(SIDISdf, numreplicas_SIDIS, 1)
# NN_Pseudo_DY_from_SIDIS_plus.to_csv(str(OutputFolder)+'/Result_DY_from_SIDIS_plus.csv')

# NN_Pseudo_DY_from_SIDIS_minus=Project_DY_Data_All_Models(SIDISdf, numreplicas_SIDIS,-1)
# NN_Pseudo_DY_from_SIDIS_minus.to_csv(str(OutputFolder)+'/Result_DY_from_SIDIS_minus.csv')
