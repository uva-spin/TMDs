import tensorflow as tf
import pandas as pd
import numpy as np
import lhapdf
import matplotlib.pyplot as plt
import os
import functions_develop
from Input_Parameterization import *


#Models_Folder = 'Models_SIDIS'
Models_Folder = '/scratch/cee9hc/Sivers_NN_Models/Train_Pseudo/Models_Eq2_v2'
ModelsArray = os.listdir(Models_Folder)
NNModels = len(ModelsArray)



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

def h(model, kperp):
    m1 = model.get_layer('a0').m1.numpy()
    e = model.get_layer('a0').e.numpy()
    print('**********************')
    print(m1)
    print(e)
    print('**********************')
    return np.sqrt(2*e) * (kperp/m1) * np.exp(-kperp**2/m1**2)


def nnq(model, x, hadronstr):
    if not hadronstr in ['nnu', 'nnd', 'nns', 'nnubar', 'nndbar', 'nnsbar']:
        raise Exception('hadronstr must be one of nnu, nnd, nns, nnubar, nndbar, nnsbar')
    mod_out = tf.keras.backend.function(model.get_layer(hadronstr).input,
                                       model.get_layer(hadronstr).output)
    return mod_out(x)


def fqp(x, QQ, kperp2avg, kperp, flavor):
    #had = functions_new.Hadron()
    had = functions_develop.Hadron()
    fq = had.pdf(flavor, x, QQ)
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
    #print(nnqval)
    #print((2*nnqval)[0, :][0])
    return ((2*nnqval*hval*fqpval)[0, :])
    #return ((2*x*hval*fqpval))
    #return ((2*nnqval)[0, :])
    #return ((2*x*nnqval))



def xsivdistFromReplicas(numReplicas, x, QQ, kperp2avg, flavor, kperp):
    results = []
    for i in range(numReplicas):
        t = tf.keras.models.load_model(Models_Folder+'/'+ModelsArray[i], 
                                         custom_objects={'A0': A0, 'Quotient': Quotient})
        results.append(list(xsivdist(t, x, QQ, kperp2avg, flavor, kperp)))
    return np.array(results)


#################################################################################
############################## From Pseudo-data #################################
#################################################################################


# def NN(x,Nq,aq,bq):
#     tempNNq = Nq*(x**aq)*((1-x)**(bq))*((aq+bq)**(aq+bq))/((aq**aq)*(bq**bq))
#     return tempNNq

# def NNanti(x,Nqbar):
#     tempNNqbar = Nqbar
#     return tempNNqbar


def hTh(m1, kperp):
    ee=1
    return np.sqrt(2*ee) * (kperp/m1) * np.exp(-kperp**2/m1**2)

def Pseudo_SiversQ(x, n, a, b, m1, QQ, kperp2avg, kperp, flavor):
    nnqval = NN(x, n, a, b)
    hval = hTh(m1, kperp)
    fqpval = fqp([x], [QQ], kperp2avg, kperp, flavor)
    #plt.plot(kperp,2*nnqval*hval*fqpval)
    return ((2*nnqval*hval*fqpval))
    #return ((2*x*hval*fqpval))
    #return ((x*nnqval))

def Pseudo_SiversAntiQ(x, n, a, b, m1, QQ, kperp2avg, kperp, flavor):
    nnqval = NNanti(x, n, a, b)
    hval = hTh(m1, kperp)
    fqpval = fqp([x], [QQ], kperp2avg, kperp, flavor)
    #plt.plot(kperp,2*nnqval*hval*fqpval)
    return ((2*nnqval*hval*fqpval))
    #return ((2*x*hval*fqpval))
    #return ((x*nnqval))


def Pseudo_Sivers_Q_Plots(x, m1, Nu, Au, Bu, Nd, Ad, Bd, Ns, As, Bs, QQ, kperp2avg, kperp):
    kp = kperp
    SivU=Pseudo_SiversQ(x, Nu, Au, Bu, m1, QQ, kperp2avg, kperp, 2)
    plt.plot(kperp, SivU,'--', color='lightblue', label='$u_{true}$')
    SivD=Pseudo_SiversQ(x, Nd, Ad, Bd, m1, QQ, kperp2avg, kperp, 1)
    plt.plot(kperp, SivD,'--', color='tomato', label='$d_{true}$')
    SivS=Pseudo_SiversQ(x, Ns, As, Bs, m1, QQ, kperp2avg, kperp, 3)
    plt.plot(kperp, SivS,'--', color='lightgreen', label='$s_{true}$')

def Pseudo_Sivers_AntiQ_Plots(x, Nub, Aub, Bub, Ndb, Adb, Bdb, Nsb, Asb, Bsb, m1, QQ, kperp2avg, kperp):
    kp = kperp
    SivU=Pseudo_SiversAntiQ(x, Nub, Aub, Bub, m1, QQ, kperp2avg, kperp, -2)
    plt.plot(kperp, SivU,'--', color='lightblue', label='$\\bar{u}_{true}$')
    SivD=Pseudo_SiversAntiQ(x, Ndb, Adb, Bdb, m1, QQ, kperp2avg, kperp, -1)
    plt.plot(kperp, SivD,'--', color='tomato', label='$\\bar{d}_{true}$')
    SivS=Pseudo_SiversAntiQ(x, Nsb, Asb, Bsb, m1, QQ, kperp2avg, kperp, -3)
    plt.plot(kperp, SivS,'--', color='lightgreen', label='$\\bar{s}_{true}$')
    

    
    
def plotSivDistBands(numReplicas, x, QQ, kperp2avg, kperp, numSigma=1):
    results = xsivdistFromReplicas(numReplicas, x, QQ, kperp2avg, 2, kperp)
    yhat = results.mean(axis=0)
    yerr = results.std(axis=0)
    #print(yerr)

    plt.fill_between(kperp, yhat-numSigma*yerr, yhat+numSigma*yerr,
                     facecolor='b', alpha=0.3)
    plt.plot(kperp, yhat, 'b', label='$u$')
    
    results = xsivdistFromReplicas(numReplicas, x, QQ, kperp2avg, 1, kperp)
    yhat = results.mean(axis=0)
    yerr = results.std(axis=0)
    
    plt.fill_between(kperp, yhat-numSigma*yerr, yhat+numSigma*yerr,
                     facecolor='r', alpha=0.3)
    plt.plot(kperp, yhat, 'r', label='$d$')
    
    results = xsivdistFromReplicas(numReplicas, x, QQ, kperp2avg, 3, kperp)
    yhat = results.mean(axis=0)
    yerr = results.std(axis=0)
    
    plt.fill_between(kperp, yhat-numSigma*yerr, yhat+numSigma*yerr,
                     facecolor='g', alpha=0.3)
    plt.plot(kperp, yhat, 'g', label='$s$')
    #Pseudo_Sivers_Q_Plots(0.1, 1, 0.2, 0.2, 1, 2.4,  .57,  np.array(list(range(150)))/100)
    Pseudo_Sivers_Q_Plots(0.1, m1v,Nuv,auv,buv,Ndv,adv,bdv,Nsv,asv,bsv, 2.4,  .57,  np.array(list(range(150)))/100)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    #plt.ylim(-0.01,0.01)
    plt.legend(loc=4,fontsize=20,handlelength=3)
    plt.savefig('SiversQ_SIDIS_NN.pdf', format='pdf', bbox_inches='tight')
    
    
def plotSivDistBandsSea(numReplicas, x, QQ, kperp2avg, kperp, numSigma=1):
    #datann = DataANN()
    #X, y, err = datann.makeData(df, [hadron], [dependence])

    results = xsivdistFromReplicas(numReplicas, x, QQ, kperp2avg, -2, kperp)
    yhat = results.mean(axis=0)
    yerr = results.std(axis=0)

    plt.fill_between(kperp, yhat-numSigma*yerr, yhat+numSigma*yerr,
                     facecolor='b', alpha=0.3)
    plt.plot(kperp, yhat, 'b', label='$\\bar{u}$')
    
    results = xsivdistFromReplicas(numReplicas, x, QQ, kperp2avg, -1, kperp)
    yhat = results.mean(axis=0)
    yerr = results.std(axis=0)
    
    plt.fill_between(kperp, yhat-numSigma*yerr, yhat+numSigma*yerr,
                     facecolor='r', alpha=0.3)
    plt.plot(kperp, yhat, 'r', label='$\\bar{d}$')
    
    results = xsivdistFromReplicas(numReplicas, x, QQ, kperp2avg, -3, kperp)
    yhat = results.mean(axis=0)
    yerr = results.std(axis=0)
    
    plt.fill_between(kperp, yhat-numSigma*yerr, yhat+numSigma*yerr,
                     facecolor='g', alpha=0.3)
    plt.plot(kperp, yhat, 'g', label='$\\bar{s}$')
    #Pseudo_Sivers_AntiQ_Plots(0.1, 0.5, 1, 2.4,  .57,  np.array(list(range(150)))/100)
    Pseudo_Sivers_AntiQ_Plots(0.1, Nubv, aubv, bubv, Ndbv, adbv, bdbv, Nsbv, asbv, bsbv, m1v, 2.4,  .57,  np.array(list(range(150)))/100)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    #plt.ylim(-0.001,0.001)
    plt.legend(loc=1,fontsize=20,handlelength=3)
    plt.savefig('SiversAntiQ_SIDIS_NN.pdf', format='pdf', bbox_inches='tight')     
    

plt.figure(1)    
plotSivDistBands(NNModels, 0.1, 2.4, .57, np.array(list(range(150)))/100)
plt.figure(2)
plotSivDistBandsSea(NNModels, 0.1, 2.4, .57, np.array(list(range(150)))/100)