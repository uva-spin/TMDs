import tensorflow as tf
import pandas as pd
import numpy as np
import lhapdf
import matplotlib.pyplot as plt
import os
import copy
import natsort
import os
import shutil


data_path = 'Models_SIDIS_100'
folders_array=os.listdir(data_path)
folders_array = natsort.natsorted(folders_array)
numreplicas_SIDIS=len(folders_array)

Data_File = 'Data/COMPASS_p_DY_2017.csv'

# def nnq(model, x, hadronstr):
#     if not hadronstr in ['nnu', 'nnd', 'nns', 'nnubar', 'nndbar', 'nnsbar']:
#         raise Exception('hadronstr must be one of nnu, nnd, nns, nnubar, nndbar, nnsbar')
#     mod_out = tf.keras.backend.function(model.get_layer(hadronstr).input,
#                                        model.get_layer(hadronstr).output)
#     return mod_out(x)

def nnq(model, x, hadronstr):
    if not hadronstr in ['nnu', 'nnd', 'nns', 'nnubar', 'nndbar', 'nnsbar']:
        raise Exception('hadronstr must be one of nnu, nnd, nns, nnubar, nndbar, nnsbar')
    lilmod = tf.keras.backend.function(model.get_layer('x').input,
                                       model.get_layer(hadronstr).output)
    return lilmod(x)

##########################################################################

class A0(tf.keras.layers.Layer):
    def __init__(self, kperp2avg=.57, pperp2avg=.12, **kwargs):
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



##########################################################################    
###################### DY Definitions ####################################
##########################################################################

class DY_Hadron(object):
    def __init__(self, kperp2avg=.57, pperp2avg=.12, pdfset='cteq61'):
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
    

#     def NN(self, x,Nq,aq,bq):
#         tempNNq = Nq*(x**aq)*((1-x)**(bq))
#         return tempNNq

#     def NNanti(self, x,Nq,aq,bq):
#         tempNNq = Nq*(x**aq)*((1-x)**(bq))
#         return tempNNq




class Sivers_DY(DY_Hadron):
    def __init__(self, kperp2avg=.57, pperp2avg=.12, pdfset='cteq61'):
        
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
    

def Project_DY_Data_All_Models(datafile, models_folder, sign):
    tempdf=pd.read_csv(datafile)
    tempDEP=np.array(tempdf['Dependence'],dtype=object)
    tempX1=np.array(tempdf['x1'],dtype=object)
    tempX2=np.array(tempdf['x2'],dtype=object)
    tempXF=np.array(tempdf['xF'],dtype=object)
    tempQT=np.array(tempdf['QT'],dtype=object)
    tempQM=np.array(tempdf['QM'],dtype=object)
    tempSivErr=np.array(tempdf['tot_err'],dtype=object)
    tempSivTh=np.array(tempdf['Siv'],dtype=object)
    data_dictionary={"Dependence":[],"x1":[],"x2":[],"xF":[],"QT":[],"QM":[],"Siv":[],"tot_err":[]}
    data_dictionary["Dependence"]=tempDEP
    data_dictionary["x1"]=tempX1
    data_dictionary["x2"]=tempX2
    data_dictionary["xF"]=tempXF
    data_dictionary["QT"]=tempQT
    data_dictionary["QM"]=tempQM
    data_dictionary["tot_err"]=tempSivErr
    SivDY=Sivers_DY()
    ############################################
    temp_Siv_Mean=[]
    temp_Siv_Std=[]
    for i in range(len(tempDEP)):
        temp=np.array([[data_dictionary["x1"][i],data_dictionary["x2"][i],
                        data_dictionary["QT"][i],data_dictionary["QM"][i]]])
        tem_Siv_model_val=[]
        for j in range(len(models_folder)):
            #print(data_path+'/'+ models_folder[j])
            t = tf.keras.models.load_model(data_path+'/'+ models_folder[j], 
                                           custom_objects={'A0': A0, 'Quotient': Quotient})
            tem_Siv_model_val.append(SivDY.sivers(t,temp, sign)[0])
        temp_mod_array=np.array(tem_Siv_model_val)
        temp_Siv_Mean.append(np.mean(temp_mod_array))
        temp_Siv_Std.append(np.std(temp_mod_array))            
    ############################################
    data_dictionary["Siv"]=np.array(tempSivTh)
    data_dictionary["Siv_NNFit"]=np.array(temp_Siv_Mean)
    data_dictionary["Siv_NNStd"]=np.array(temp_Siv_Std)
    return pd.DataFrame(data_dictionary)


Asym_No_Sign_Change=Project_DY_Data_All_Models(Data_File, folders_array,1)
Asym_With_Sign_Change=Project_DY_Data_All_Models(Data_File, folders_array,-1)


def DYDependencePlotSign(dataframe,dep):
    tempdf=dataframe[dataframe["Dependence"]==dep]
    tempx=np.array(tempdf[dep])
    tempNNx=np.array(tempdf[dep]+0.01)
    tempy=np.array(tempdf["Siv"])
    tempyerr=np.array(tempdf["tot_err"])
    tempNNy=np.array(tempdf["Siv_NNFit"])
    tempNNyerr=np.array(tempdf["Siv_NNStd"])
    plt.errorbar(tempx,tempy,tempyerr,fmt='o',color='blue')
    plt.errorbar(tempx,tempNNy,tempNNyerr,fmt='o',color='red')
    plt.title('Asymmetry vs '+str(dep))
    
    
def DYAsymPlots(dataframe,fignum,figname):
    fig1=plt.figure(1,figsize=(15,3))
    plt.subplot(1,5,1)
    DYDependencePlotSign(dataframe,'x1')
    plt.subplot(1,5,2)
    DYDependencePlotSign(dataframe,'x2')
    plt.subplot(1,5,3)
    DYDependencePlotSign(dataframe,'xF')
    plt.subplot(1,5,4)
    DYDependencePlotSign(dataframe,'QT')
    plt.subplot(1,5,5)
    DYDependencePlotSign(dataframe,'QM')
    plt.savefig(figname+'.pdf',format='pdf',bbox_inches='tight')
    

DYAsymPlots(Asym_No_Sign_Change,2,'test2')