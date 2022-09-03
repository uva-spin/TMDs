import lhapdf
import numpy as np
from Input_Parameterization import *
# Quark Flavor correspondence in LHAPDF:
# 1-d; 2-u; 3-s; 4-t; 5-t; 6-b; 21-gluon
# negative for corresponding anti-quark

class Hadron(object):
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
    

    # def NN(self, x,Nq,aq,bq):
    #     tempNNq = Nq*(x**aq)*((1-x)**(bq))
    #     return tempNNq

    # def NNanti(self, x,Nq,aq,bq):
    #     tempNNq = Nq*(x**aq)*((1-x)**(bq))
    #     return tempNNq
    # def NN(self, x, n, a, b):
    #     return n * x**a * (1 - x)**b * (((a + b)**(a + b))/(a**a * b**b))

    # def NNanti(self, x, n):
    #     return x*n

#     def NN(self, x, n, a, b):
#         return n * x**a * (1 - x)**b 

#     def NNanti(self, x, n):
#         return n*x
    
    
class Sivers_Hadron(Hadron):
    def __init__(self, kperp2avg=.57, pperp2avg=.12, pdfset='cteq61',
                 ff_PIp='NNFF10_PIp_nlo', ff_PIm='NNFF10_PIm_nlo', ff_PIsum='NNFF10_PIsum_nlo',
                 ff_KAp='NNFF10_KAp_nlo', ff_KAm='NNFF10_KAm_nlo'):
        
        super().__init__(kperp2avg=kperp2avg, pperp2avg=pperp2avg, pdfset=pdfset)

        
    def sivers(self, had, kins, m1, Nu, au, bu, Nub, Nd, ad, bd, Ndb):
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
        #print(ii)           
        x = kins[:, 0]
        z = kins[:, 1]
        pht = kins[:, 2]
        QQ = kins[:, 3]
        a0 = self.A0(z, pht, m1)
        temp_top = NNq(x, Nu, au, bu) * self.eu**2 * self.pdf(2, x, QQ) * self.ff(self.ffDict[ii],2, z, QQ)
        + NNqbar(x, Nub) * self.eubar**2 * self.pdf(-2, x, QQ) * self.ff(self.ffDict[ii],-2, z, QQ)
        + NNq(x, Nd, ad, bd) * self.ed**2 * self.pdf(1, x, QQ) * self.ff(self.ffDict[ii],1, z, QQ)
        + NNqbar(x, Ndb) * self.edbar**2 * self.pdf(-1, x, QQ) * self.ff(self.ffDict[ii],-1, z, QQ) 
        #+ NNq(x, NS, aS, bS) * self.es**2 * self.pdf(3, x, QQ) * self.ff(self.ffDict[ii],3, z, QQ)
        #+ NNqbar(x, NSb) * self.esbar**2 * self.pdf(-3, x, QQ) * self.ff(self.ffDict[ii],-3, z, QQ)
        temp_bottom =  self.eu**2 * self.pdf(2, x, QQ) * self.ff(self.ffDict[ii],2, z, QQ)
        + self.eubar**2 * self.pdf(-2, x, QQ) * self.ff(self.ffDict[ii],-2, z, QQ)
        + self.ed**2 * self.pdf(1, x, QQ) * self.ff(self.ffDict[ii],1, z, QQ)
        + self.edbar**2 * self.pdf(-1, x, QQ) * self.ff(self.ffDict[ii],-1, z, QQ)
        #+ self.es**2 * self.pdf(3, x, QQ) * self.ff(self.ffDict[ii],3, z, QQ)
        #+ self.esbar**2 * self.pdf(-3, x, QQ) * self.ff(self.ffDict[ii],-3, z, QQ)
        temp_siv_had = a0*((temp_top)/(temp_bottom))
        #print(temp_siv_had)
        return temp_siv_had
 

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
    


class Sivers_DY(DY_Hadron):
    def __init__(self, kperp2avg=.57, pperp2avg=.12, pdfset='cteq61'):
        
        super().__init__(kperp2avg=kperp2avg, pperp2avg=pperp2avg, pdfset=pdfset)
        
    def sivers(self, kins, m1, Nu, au, bu, Nub, Nd, ad, bd, Ndb):
        x1 = kins[:, 0]
        x2 = kins[:, 1]
        qT = kins[:, 2]
        QQ = kins[:, 3] 
        b0 = self.B0(qT, m1)
        temp_top = NNq(x1, Nu, au, bu) * self.eu**2 * self.pdf_DY(2, x1, QQ) * self.pdf_DY(-2, x2, QQ)
        + NNq(x2, Nu, au, bu) * self.eu**2 * self.pdf_DY(2, x2, QQ) * self.pdf_DY(-2, x1, QQ)
        + NNqbar(x1, Nub) * self.eubar**2 * self.pdf_DY(-2, x1, QQ) * self.pdf_DY(2, x2, QQ)
        + NNqbar(x2, Nub) * self.eubar**2 * self.pdf_DY(-2, x2, QQ) * self.pdf_DY(2, x1, QQ)
        + NNq(x1, Nd, ad, bd) * self.ed**2 * self.pdf_DY(1, x1, QQ) * self.pdf_DY(-1, x2, QQ)
        + NNq(x2, Nd, ad, bd) * self.ed**2 * self.pdf_DY(1, x2, QQ) * self.pdf_DY(-1, x1, QQ)
        + NNqbar(x1, Ndb) * self.edbar**2 * self.pdf_DY(-1, x1, QQ) * self.pdf_DY(1, x2, QQ)
        + NNqbar(x2, Ndb) * self.edbar**2 * self.pdf_DY(-1, x2, QQ) * self.pdf_DY(1, x1, QQ)
        #+ NNq(x1, NS, aS, bS) * self.es**2 * self.pdf_DY(3, x1, QQ) * self.pdf_DY(-3, x2, QQ)
        #+ NNq(x2, NS, aS, bS) * self.es**2 * self.pdf_DY(3, x2, QQ) * self.pdf_DY(-3, x1, QQ)
        #+ NNqbar(x1, NSb, aSb, bSb) * self.esbar**2 * self.pdf_DY(-3, x1, QQ) * self.pdf_DY(3, x2, QQ)
        #+ NNqbar(x2, NSb, aSb, bSb ) * self.esbar**2 * self.pdf_DY(-3, x2, QQ) * self.pdf_DY(3, x1, QQ)
        temp_bottom = self.eu**2 * self.pdf_DY(2, x1, QQ) * self.pdf_DY(-2, x2, QQ)
        + self.eu**2 * self.pdf_DY(2, x2, QQ) * self.pdf_DY(-2, x1, QQ)
        + self.eubar**2 * self.pdf_DY(-2, x1, QQ) * self.pdf_DY(2, x2, QQ)
        + self.eubar**2 * self.pdf_DY(-2, x2, QQ) * self.pdf_DY(2, x1, QQ)
        + self.ed**2 * self.pdf_DY(1, x1, QQ) * self.pdf_DY(-1, x2, QQ)
        + self.ed**2 * self.pdf_DY(1, x2, QQ) * self.pdf_DY(-1, x1, QQ)
        + self.edbar**2 * self.pdf_DY(-1, x1, QQ) * self.pdf_DY(1, x2, QQ)
        + self.edbar**2 * self.pdf_DY(-1, x2, QQ) * self.pdf_DY(1, x1, QQ)
        #+ self.es**2 * self.pdf_DY(3, x1, QQ) * self.pdf_DY(-3, x2, QQ)
        #+ self.es**2 * self.pdf_DY(3, x2, QQ) * self.pdf_DY(-3, x1, QQ)
        #+ self.esbar**2 * self.pdf_DY(-3, x1, QQ) * self.pdf_DY(3, x2, QQ)
        #+ self.esbar**2 * self.pdf_DY(-3, x2, QQ) * self.pdf_DY(3, x1, QQ)
        temp_siv_had = b0*((temp_top)/(temp_bottom))
        return temp_siv_had
 