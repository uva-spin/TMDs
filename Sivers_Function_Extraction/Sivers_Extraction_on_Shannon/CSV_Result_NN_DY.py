import tensorflow as tf
import pandas as pd
import numpy as np
import lhapdf
import matplotlib.pyplot as plt
#import functions_develop
#import natsort
import os

Models_folder = '/project/UVA-Spin/Ishara/Sivers/Extraction_01/DY_Models'
folders_array=os.listdir(Models_folder)
#folders_array=natsort.natsorted(folders_array)
OutputFolder='NN_DY_Fit_Results'
#os.mkdir(OutputFolder)
numreplicas_DY=len(folders_array)
#numreplicas_SIDIS=
print(numreplicas_DY)

DY_Projected_DATA = pd.read_csv('./NN_SIDIS_Fit_Results/Result_DY_from_SIDIS_minus.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
DY_Projected_DATA_df = pd.concat([DY_Projected_DATA])

DY_COMPASS17_DATA = pd.read_csv('./Data/COMPASS_p_DY_2017.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
DY_COMPASS17_DATA_df = pd.concat([DY_COMPASS17_DATA])

DY_SPINQUEST_DATA = pd.read_csv('./Data/SQ_xt_dependence_QT_05_QM_4.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
DY_SPINQUEST_DATA_df = pd.concat([DY_SPINQUEST_DATA])

df_compass = pd.concat([DY_COMPASS17_DATA])
df_spinquest = pd.concat([DY_SPINQUEST_DATA])



def calc_yhat(model, X):
    return model.predict(X)


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


class Quotient(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Quotient, self).__init__()

    def call(self, inputs):
        if len(inputs) != 2 or inputs[0].shape[1] != 1:
            raise Exception('must be two tensors of shape (?, 1)')
        return inputs[0]/inputs[1]


DYmodelsArray = []
for i in range(numreplicas_DY):
    testmodel = tf.keras.models.load_model(str(Models_folder)+'/' + str(folders_array[i]),custom_objects={'B0': B0, 'Quotient': Quotient})
    DYmodelsArray.append(testmodel)
    
DYmodelsArray = np.array(DYmodelsArray)

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

DY_datann = DY_DataANN()


def SiversAsym(df,numReplicas,dependence):
    T_Kins, T_Xplt, T_yplt, T_errplt = DY_datannn.makeData(df, [dependence])
    results = []
    for i in range(numReplicas):
        #folders_array
        #testmodel = tf.keras.models.load_model(str(Models_folder)+'/' + str(folders_array[i]),custom_objects={'A0': A0, 'Quotient': Quotient})
        #testmodel = tf.keras.models.load_model(str(Models_folder)+'/rep' + str(i) + '.h5',custom_objects={'A0': A0, 'Quotient': Quotient})
        testmodel = DYmodelsArray[i]
        results.append(list(calc_yhat(testmodel, T_Xplt)))
    return np.array(results)



# def GenPseudo(datasetdf, numReplicas, numSigma=1):
#     data_dictionary={"Dependence":[],"x1":[],"x2":[],"xF":[],"QT":[],"QM":[],"Siv":[],"tot_err":[]}
#     temp_hads = pd.unique(datasetdf['hadron'])
#     temp_deps = pd.unique(datasetdf['1D_dependence'])
#     SiversA = []
#     SiversA_err = []
#     for i in temp_hads:
#         for j in temp_deps:
#             T_Kins, T_Xplt, T_DEP, T_yplt, T_errplt = SIDISdatann.makeData(datasetdf, [str(i)], [str(j)])
#             #T_Kins, T_Xplt, T_DEP, T_yplt, T_errplt = datann.makeData(datasetdf, ['pi+'], ['x'])
#             results = SiversAsym(datasetdf, numReplicas,i, j)
#             Yhat = np.array(results.mean(axis=0))
#             Yhat = Yhat.flatten()
#             Yerr = np.array(results.std(axis=0))
#             Yerr = Yerr.flatten()
#             SiversA.append(Yhat)
#             SiversA_err.append(Yerr)
#             data_dictionary['hadron']+= list(T_Kins['hadron'])
#             data_dictionary['Q2']+= list(T_Kins['Q2'])
#             data_dictionary['x']+= list(T_Kins['x'])
#             data_dictionary['y']+= list(T_Kins['y'])
#             data_dictionary['z']+= list(T_Kins['z'])
#             data_dictionary['phT']+= list(T_Kins['phT'])
#             data_dictionary['1D_dependence']+= list(T_Kins['1D_dependence'])
#             data_dictionary['Siv']+= list(T_yplt)
#             data_dictionary['tot_err']+= list(T_errplt)
#             #print(Yhat)
#     #print(np.concatenate(t_hads))
#     data_dictionary['Siv'] = np.concatenate(SiversA)
#     data_dictionary['tot_err'] = np.concatenate(SiversA_err)
#     #data_dictionary['Siv_NNFit'] = np.concatenate(SiversA)
#     #data_dictionary['Siv_NNStd'] = np.concatenate(SiversA_err)
#     #print(len(data_dictionary['hadron']))
#     #print(len(data_dictionary['Siv']))
#     #print(data_dictionary)
#     return pd.DataFrame(data_dictionary)
    


# NN_Pseudo_SIDIS_HERMES2009=GenPseudo(HERMES2009, numreplicas_SIDIS, numSigma=1)
# NN_Pseudo_SIDIS_HERMES2020=GenPseudo(HERMES2020, numreplicas_SIDIS, numSigma=1)
# NN_Pseudo_SIDIS_COMPASS2009=GenPseudo(COMPASS2009, numreplicas_SIDIS, numSigma=1)
# NN_Pseudo_SIDIS_COMPASS2015=GenPseudo(COMPASS2015, numreplicas_SIDIS, numSigma=1)



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
    def __init__(self, kperp2avg=.25, pperp2avg=.12, pdfset='cteq61'):
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
    def __init__(self, kperp2avg=.25, pperp2avg=.12, pdfset='cteq61'):
        
        super().__init__(kperp2avg=kperp2avg, pperp2avg=pperp2avg, pdfset=pdfset)
    
        
    def sivers(self, model, kins, SIGN):
        x1 = kins[:, 0]
        x2 = kins[:, 1]
        qT = kins[:, 2]
        QQ = kins[:, 3] 
        m1 = model.get_layer('b0').m1.numpy()
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
    

#GenDYKinematicsFromSIDIS(SIDISdf)

def Project_DY_from_Models(tempdf, numReplicas, sign):
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
        # x1 is beam, x2 is target in the COMPASS Data
        # x1 is target, and x2 is beam in the formalism
        temp=np.array([[data_dictionary["x2"][i],data_dictionary["x1"][i],
                        data_dictionary["QT"][i],data_dictionary["QM"][i]]])        
        # temp=np.array([[data_dictionary["x1"][i],data_dictionary["x2"][i],
        #                 data_dictionary["QT"][i],data_dictionary["QM"][i]]])
        tem_Siv_model_val=[]
        for j in range(numReplicas):
            #print(data_path+'/'+ models_folder[j])
            #t = tf.keras.models.load_model(str(Models_folder)+'/' + str(folders_array[j]),custom_objects={'A0': A0, 'Quotient': Quotient})
            t = DYmodelsArray[j]
            tem_Siv_model_val.append(SivDY.sivers(t,temp, sign)[0])
        temp_mod_array=np.array(tem_Siv_model_val)
        temp_Siv_Mean.append(np.mean(temp_mod_array))
        temp_Siv_Std.append(np.std(temp_mod_array))            
    ############################################
    #data_dictionary["Siv"]=np.array(tempSivTh)
    data_dictionary["Siv"]=np.array(temp_Siv_Mean)
    data_dictionary["tot_err"]=np.array(temp_Siv_Std)
    return pd.DataFrame(data_dictionary)


NN_COMPASS_DY_from_SIDIS_minus=Project_DY_from_Models(df_compass, numreplicas_DY, 1)
NN_E1039_DY_from_SIDIS_minus=Project_DY_from_Models(df_spinquest, numreplicas_DY, 1)

NN_COMPASS_DY_from_SIDIS_minus.to_csv(str(OutputFolder)+'/Result_DY_COMPASS.csv')
NN_E1039_DY_from_SIDIS_minus.to_csv(str(OutputFolder)+'/Result_DY_SPINQUEST.csv')
#testdf = GenDYKinematicsFromSIDIS(SIDISdf)
#testdf.to_csv('testDY.csv')

#######################################################################
########## Sivers Function ############################################
#######################################################################



def h(model, kperp):
    m1 = model.get_layer('b0').m1.numpy()
    e = model.get_layer('b0').e.numpy()
    return np.sqrt(2*e) * (kperp/m1) * np.exp(-kperp**2/m1**2)


def fqp(x, QQ, kperp2avg, kperp, flavor):
    #had = functions_new.Hadron()
    had = DY_Hadron()
    fq = had.pdf_DY(flavor, x, QQ)
    return fq*(1/(np.pi*kperp2avg))*np.exp(-kperp**2/kperp2avg)
    
    
def xsivdist(model, x, QQ, kperp2avg, flavor, kperp):
    refDict = {-3: 'nnsbar',
               -2: 'nnubar',
               -1: 'nndbar',
               1: 'nnd',
               2: 'nnu',
               3: 'nns'}
    nnqval = nnq(model, np.array([x]), refDict[flavor])
    #nnqval = nnq(model , np.array([x]), refDict[flavor])[:,0]
    hval = h(model, kperp)
    fqpval = fqp([x], [QQ], kperp2avg, kperp, flavor)
    return ((2*nnqval*hval*fqpval)[0, :])



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
        t = DYmodelsArray[i]
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
    

fSivCSV = SivDistBandsCSVgen(numreplicas_DY, 0.1, 2.4, 0.25, np.array(list(range(150)))/100)
fSivCSV.to_csv(str(OutputFolder)+'/'+'Sivfuncs.csv')