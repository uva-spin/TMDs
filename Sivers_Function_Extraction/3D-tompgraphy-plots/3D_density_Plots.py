import tensorflow as tf
import pandas as pd
import numpy as np
import lhapdf
import matplotlib.pyplot as plt
import os


Models_folder = './SIDIS_Models'
folders_array=os.listdir(Models_folder)
OutputFolder='NN_SIDIS_Plots'
#numreplicas_SIDIS=len(folders_array)
numreplicas_SIDIS=3
print(numreplicas_SIDIS)

pdfset = 'cteq61'
Mp = 0.938

class A0(tf.keras.layers.Layer):
    def __init__(self, kperp2avg=.57, pperp2avg=.12, **kwargs):
        super(A0, self).__init__(name='a0')
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
    
    
def nnq(model, x, hadronstr):
    if not hadronstr in ['nnu', 'nnd', 'nns', 'nnubar', 'nndbar', 'nnsbar']:
        raise Exception('hadronstr must be one of nnu, nnd, nns, nnubar, nndbar, nnsbar')
    mod_out = tf.keras.backend.function(model.get_layer(hadronstr).input,
                                       model.get_layer(hadronstr).output)
    return mod_out(x)

def pdf_vals(flavor, x, QQ):
    pdfData = lhapdf.mkPDF(pdfset)
    return np.array([pdfData.xfxQ2(flavor, ax, qq) for ax, qq in zip(x, QQ)])

def h(model, kperp):
    m1 = model.get_layer('a0').m1.numpy()
    e = model.get_layer('a0').e.numpy()
    return np.sqrt(2*e) * (kperp/m1) * np.exp(-kperp**2/m1**2)

def fqp(x, QQ, kperp2avg, kperp, flavor):
    fq = pdf_vals(flavor, x, QQ)
    return fq*(1/(np.pi*kperp2avg))*np.exp(-kperp**2/kperp2avg)


def xSivFunction(model, x, QQ, kperp2avg, flavor, kperp):
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


def kval(kx,ky):
    return np.sqrt(kx**2 + ky**2)


def density_func(model, x, QQ, kperp2avg, flavor, kxx, kyy):
    kTval = kval(kxx,kyy)
    return fqp(x, QQ, kperp2avg, kTval, flavor) - (kxx/Mp)*xSivFunction(model, x, QQ, kperp2avg, flavor, kTval)


SIDISmodelsArray = []
for i in range(numreplicas_SIDIS):
    testmodel = tf.keras.models.load_model(str(Models_folder)+'/' + str(folders_array[i]),custom_objects={'A0': A0, 'Quotient': Quotient})
    SIDISmodelsArray.append(testmodel)
    
SIDISmodelsArray = np.array(SIDISmodelsArray)


kxv = np.array(np.linspace(-1,1,100))
kyv = np.array(np.linspace(-1,1,100))
kTv = kval(kxv,kyv)


def 3D_plot_FromReplicas(numReplicas, x, QQ, kperp2avg, kperp):
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
        tempfu.append(list(density_func(tt, [x], [2], 0.57, 2, kxv, kyv)))
        tempfd.append(list(density_func(tt, [x], [2], 0.57, 1, kxv, kyv)))
        tempfs.append(list(density_func(tt, [x], [2], 0.57, 3, kxv, kyv)))
        tempfubar.append(list(density_func(tt, [x], [2], 0.57, -2, kxv, kyv)))
        tempfdbar.append(list(density_func(tt, [x], [2], 0.57, -1, kxv, kyv)))
        tempfsbar.append(list(density_func(tt, [x], [2], 0.57, -3, kxv, kyv)))
    return np.array(tempfu),np.array(tempfubar),np.array(tempfd),np.array(tempfdbar),np.array(tempfs),np.array(tempfsbar)

fig1=plt.figure(1)
side = np.linspace(-1,1,100)
kxv, kyv = np.meshgrid(side,side)
Zu = density_func(tt, [0.1], [2], 0.57, 2, kxv, kyv)
# Plot the density map using nearest-neighbor interpolation
#plt.pcolormesh(kxv,kyv,Zu)
plt.pcolormesh(kxv,kyv,Zu)
plt.savefig(str(OutputFolder)+'/'+'test_3D_plot')




