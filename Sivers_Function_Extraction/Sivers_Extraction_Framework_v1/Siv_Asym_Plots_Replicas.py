import tensorflow as tf
import pandas as pd
import numpy as np
import lhapdf
import matplotlib.pyplot as plt
import glob
import os
import copy
import natsort

data_path = 'Models_SIDIS_Set_3'
folders_array=os.listdir(data_path)
folders_array = natsort.natsorted(folders_array)
numreplicas_SIDIS=len(folders_array)

data_replica_path = 'Pseudo-Data/Set_3_Replicas'
data_folders_array=os.listdir(data_replica_path)
data_folders_array = natsort.natsorted(data_folders_array)

Results_Plots_Path = 'Asym_Plots'
os.mkdir(Results_Plots_Path)


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


def nnq(model, x, hadronstr):
    if not hadronstr in ['nnu', 'nnd', 'nns', 'nnubar', 'nndbar', 'nnsbar']:
        raise Exception('hadronstr must be one of nnu, nnd, nns, nnubar, nndbar, nnsbar')
    mod_out = tf.keras.backend.function(model.get_layer(hadronstr).input,
                                       model.get_layer(hadronstr).output)
    return mod_out(x)

    
class Sivers_Hadron_Rep():
    def __init__(self, kperp2avg=.57, pperp2avg=.12, pdfset='cteq61',
                 ff_PIp='NNFF10_PIp_nlo', ff_PIm='NNFF10_PIm_nlo', ff_PIsum='NNFF10_PIsum_nlo',
                 ff_KAp='NNFF10_KAp_nlo', ff_KAm='NNFF10_KAm_nlo'):

        self.pdfData = lhapdf.mkPDF(pdfset)
        self.ffDataPIp = lhapdf.mkPDF(ff_PIp, 0)
        self.ffDataPIm = lhapdf.mkPDF(ff_PIm, 0)
        self.ffDataPIsum = lhapdf.mkPDF(ff_PIsum, 0)
        self.ffDataKAp = lhapdf.mkPDF(ff_KAp, 0)
        self.ffDataKAm = lhapdf.mkPDF(ff_KAm, 0)
        
        self.kperp2avg = kperp2avg
        self.pperp2avg = pperp2avg
        self.eu = 2/3
        self.eubar = -2/3
        self.ed = -1/3
        self.edbar = 1/3
        self.es = -1/3
        self.esbar = 1/3
        self.e = 1
    
        self.ffDict = {0: self.ffDataPIp,
               1: self.ffDataPIm,
               2: self.ffDataPIsum,
               3: self.ffDataKAp,
               4: self.ffDataKAm}
        
        #super().__init__(kperp2avg=kperp2avg, pperp2avg=pperp2avg, pdfset=pdfset)

    def pdf(self, flavor, x, QQ):
        return np.array([self.pdfData.xfxQ2(flavor, ax, qq) for ax, qq in zip(x, QQ)])
    
    def ff(self, func, flavor, z, QQ):
        return np.array([func.xfxQ2(flavor, az, qq) for az, qq in zip(z, QQ)])

    
    def A0(self, z, pht, m1):
        ks2avg = (self.kperp2avg*m1**2)/(m1**2 + self.kperp2avg) #correct 
        topfirst = (z**2 * self.kperp2avg + self.pperp2avg) * ks2avg**2 #correct
        bottomfirst = (z**2 * ks2avg + self.pperp2avg)**2 * self.kperp2avg #correct
        exptop = pht**2 * z**2 * (ks2avg - self.kperp2avg) #correct
        expbottom = (z**2 * ks2avg + self.pperp2avg) * (z**2 * self.kperp2avg + self.pperp2avg) #correct
        last = np.sqrt(2*self.e) * z * pht / m1 #correct
        
        return (topfirst/bottomfirst) * np.exp(-exptop/expbottom) * last
    
    
#     def NN(self, x, n, a, b):
#         return n * x**a * (1 - x)**b * (((a + b)**(a + b))/(a**a * b**b))

#     def NNanti(self, x, n):
#         return x*n

    def h(model, kperp):
        m1 = model.get_layer('a0').m1.numpy()
        e = model.get_layer('a0').e.numpy()
        return np.sqrt(2*e) * (kperp/m1) * np.exp(-kperp**2/m1**2)


    def nnq(model, x, hadronstr):
        if not hadronstr in ['nnu', 'nnd', 'nns', 'nnubar', 'nndbar', 'nnsbar']:
            raise Exception('hadronstr must be one of nnu, nnd, nns, nnubar, nndbar, nnsbar')
        mod_out = tf.keras.backend.function(model.get_layer(hadronstr).input,
                                       model.get_layer(hadronstr).output)
        return mod_out(x)

        
    def sivers(self, model, had, kins):
        if had == 'pi+':
            ii = 0
        elif had == 'pi-':
            ii = 1
        elif had == 'pi0':
            ii = 2
        elif had == 'k+':
            ii = 3
        elif had == 'k-':
            ii = 4
        #ii=1           
        x = kins[:, 0]
        z = kins[:, 1]
        pht = kins[:, 2]
        QQ = kins[:, 3]
        m1 = model.get_layer('a0').m1.numpy()
        a0 = self.A0(z, pht, m1)
        NNu = nnq(model , np.array(x), 'nnu')[:,0]
        NNubar = nnq(model , np.array(x), 'nnubar')[:,0]
        NNd = nnq(model , np.array(x), 'nnd')[:,0]
        NNdbar = nnq(model , np.array(x), 'nndbar')[:,0]
        NNs = nnq(model , np.array(x), 'nns')[:,0]
        NNsbar = nnq(model , np.array(x), 'nnsbar')[:,0]
        temp_top = NNu * self.eu**2 * self.pdf(2, x, QQ) * self.ff(self.ffDict[ii],2, z, QQ)
        + NNubar * self.eubar**2 * self.pdf(-2, x, QQ) * self.ff(self.ffDict[ii],-2, z, QQ)
        + NNd * self.ed**2 * self.pdf(1, x, QQ) * self.ff(self.ffDict[ii],1, z, QQ)
        + NNdbar * self.edbar**2 * self.pdf(-1, x, QQ) * self.ff(self.ffDict[ii],-1, z, QQ) 
        + NNs * self.es**2 * self.pdf(3, x, QQ) * self.ff(self.ffDict[ii],3, z, QQ)
        + NNsbar * self.esbar**2 * self.pdf(-3, x, QQ) * self.ff(self.ffDict[ii],-3, z, QQ)
        temp_bottom =  self.eu**2 * self.pdf(2, x, QQ) * self.ff(self.ffDict[ii],2, z, QQ)
        + self.eubar**2 * self.pdf(-2, x, QQ) * self.ff(self.ffDict[ii],-2, z, QQ)
        + self.ed**2 * self.pdf(1, x, QQ) * self.ff(self.ffDict[ii],1, z, QQ)
        + self.edbar**2 * self.pdf(-1, x, QQ) * self.ff(self.ffDict[ii],-1, z, QQ)
        + self.es**2 * self.pdf(3, x, QQ) * self.ff(self.ffDict[ii],3, z, QQ)
        + self.esbar**2 * self.pdf(-3, x, QQ) * self.ff(self.ffDict[ii],-3, z, QQ)
        temp_siv_had = a0*((temp_top)/(temp_bottom))
        return temp_siv_had
    

def Replicas_Result(model,rep_file):
    tempdf=pd.read_csv(rep_file)
    temphad=np.array(tempdf['hadron'],dtype=object)
    tempQ2=np.array(tempdf['Q2'],dtype=object)
    tempX=np.array(tempdf['x'],dtype=object)
    tempY=np.array(tempdf['y'],dtype=object)
    tempZ=np.array(tempdf['z'],dtype=object)
    tempPHT=np.array(tempdf['phT'],dtype=object)
    tempSivErr=np.array(tempdf['Siv_Rep_err'],dtype=object)
    tempSivTh=np.array(tempdf['Siv_Rep'],dtype=object)
    #tempSivResult=tempdf['Siv_test']
    tempDEP=np.array(tempdf['1D_dependence'],dtype=object)
    data_dictionary={"hadron":[],"Q2":[],"x":[],"y":[],"z":[],"phT":[],"Siv":[],"tot_err":[],"1D_dependence":[]}
    data_dictionary["hadron"]=temphad
    data_dictionary["Q2"]=tempQ2
    data_dictionary["x"]=tempX
    data_dictionary["y"]=tempY
    data_dictionary["z"]=tempZ
    data_dictionary["phT"]=tempPHT
    data_dictionary["tot_err"]=tempSivErr
    data_dictionary["1D_dependence"]=tempDEP
    #data_dictionary["Siv_test"]=tempdf["Siv_test"]
    PiP=copy.deepcopy(data_dictionary)
    PiM=copy.deepcopy(data_dictionary)
    Pi0=copy.deepcopy(data_dictionary)
    KP=copy.deepcopy(data_dictionary)
    KM=copy.deepcopy(data_dictionary)
    SivHad = Sivers_Hadron_Rep()
    ############################################
    temp_Siv=[]
    for i in range(len(temphad)):
        temp=np.array([[data_dictionary["x"][i],data_dictionary["z"][i],
                        data_dictionary["phT"][i],data_dictionary["Q2"][i]]])
        temp_had=data_dictionary["hadron"][i]  
        temp_Siv.append(SivHad.sivers(model,temp_had,temp)[0])
    ############################################
    data_dictionary["Siv"]=np.array(tempSivTh)
    data_dictionary["Siv_NNFit"]=np.array(temp_Siv)
    return pd.DataFrame(data_dictionary)


def PlotResults(file_num):
    sample_file = data_replica_path + '/'+ data_folders_array[file_num]
    t = tf.keras.models.load_model(data_path+'/'+ folders_array[file_num], 
        custom_objects={'A0': A0, 'Quotient': Quotient})
    temp_result=Replicas_Result(t,sample_file)
    xx =[i for i in range(len((np.array(temp_result["Siv_NNFit"]))))]
    xxy = [i+float(0.5) for i in range(len((np.array(temp_result["Siv_NNFit"]))))]
    Fit = np.array(temp_result["Siv_NNFit"])
    Th = np.array(temp_result["Siv"])
    err = np.array(temp_result["tot_err"])
    fig = plt.figure(file_num, figsize=(100,10))
    plt.plot(xxy,Fit,'.',color = 'red')
    plt.errorbar(xx,Th,err,fmt='o',color = 'blue') 
    plt.xlabel('Data Point #', fontsize=20)
    plt.ylabel('Sivers Asymmetry', fontsize=20)
    plt.title('Plot for replica #'+str(file_num),fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    fig.savefig(Results_Plots_Path + '/test_' + str(file_num)+'.pdf')

for i in range(5):
    PlotResults(i)