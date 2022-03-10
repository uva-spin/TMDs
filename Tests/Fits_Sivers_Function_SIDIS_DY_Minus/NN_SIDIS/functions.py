import lhapdf
import numpy as np

# Quark Flavor correspondence in LHAPDF:
# 1-d; 2-u; 3-s; 4-t; 5-t; 6-b; 21-gluon
# negative for corresponding anti-quark

class Hadron(object):
    def __init__(self, kperp2avg=.57, pperp2avg=.12, pdfset='JAM19PDF_proton_nlo',
                 ff_pion='JAM19FF_pion_nlo', ff_kaon='JAM19FF_kaon_nlo'):
        '''
        Parent class of individual hadron functions as defined in Sivers Extraction with Neural Network (2021)
        '''
        self.pdfData = lhapdf.mkPDF(pdfset)
        self.ffDataPion = lhapdf.mkPDF(ff_pion, 0)
        self.ffDataKaon = lhapdf.mkPDF(ff_kaon, 0)
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
    
    def ffPion(self, flavor, z, QQ):
        return np.array([self.ffDataPion.xfxQ2(flavor, az, qq) for az, qq in zip(z, QQ)])
    
    def ffKaon(self, flavor, z, QQ):
        return np.array([self.ffDataKaon.xfxQ2(flavor, az, qq) for az, qq in zip(z, QQ)])
    
    
    def A0(self, z, pht, m1):
        ks2avg = (self.kperp2avg*m1**2)/(m1**2 + self.kperp2avg) #correct 
        topfirst = (z**2 * self.kperp2avg + self.pperp2avg) * ks2avg**2 #correct
        bottomfirst = (z**2 * ks2avg + self.pperp2avg)**2 * self.kperp2avg #correct
        exptop = pht**2 * z**2 * (ks2avg - self.kperp2avg) #correct
        expbottom = (z**2 * ks2avg + self.pperp2avg) * (z**2 * self.kperp2avg + self.pperp2avg) #correct
        last = np.sqrt(2*self.e) * z * pht / m1 #correct
        
        return (topfirst/bottomfirst) * np.exp(-exptop/expbottom) * last
    
    
    def NN(self, x, n, a, b):
        return n * x**a * (1 - x)**b * (((a + b)**(a + b))/(a**a * b**b))

    def NNanti(self, n):
        return n
            
    
class PiPlus(Hadron):

    def __init__(self, kperp2avg=.57, pperp2avg=.12, pdfset='JAM19PDF_proton_nlo',
                 ff_pion='JAM19FF_pion_nlo'):
        '''
        The PiPlus reaction occurs between "u" and "dbar" quarks
        
        :param kperp2avg: average kperp^2
        :param pperp2avg: average pperp^2
        :param pdfset: the name of the pdf grid downloaded from lhapdf that you would like to use
        :param ff_pion: the name of the fragmentation function grid (for pions) downloaded from lhapdf
        '''        
        super().__init__(kperp2avg=kperp2avg, pperp2avg=pperp2avg, pdfset=pdfset,
                 ff_pion=ff_pion)
        
    
    def sivers(self, kins, Nu, au, bu, Ndbar, m1):  # adbar, bdbar): are not necessary
        '''
        Calculate sivers assymetry for specified variables
        
        :param kins: numpy array w shape (n, 4) of kinematics in order of x, z, pht, QQ (kins[:, 0] = xs)
        :param Nu: free parameter of NN function for u quark (corresponds to N)
        :param au: free parameter of NN function for u quark (corresponds to alpha)
        :param bu: free parameter of NN function for u quark (corresponds to beat)
        :param Ndbar: free parameter of NN function for dbar quark (corresponds to N)
        :param m1: free parameter of A0 function
        
        :returns: length n array of sivers assymetries
        '''
        
        x = kins[:, 0]
        z = kins[:, 1]
        pht = kins[:, 2]
        QQ = kins[:, 3]
        a0 = self.A0(z, pht, m1)
        topleft = self.NN(x, Nu, au, bu) * self.eu**2 * self.pdf(2, x, QQ) * self.ffPion(2, z, QQ)
        topright = self.NNanti(Ndbar) * self.edbar**2 * self.pdf(-1, x, QQ) * self.ffPion(-1, z, QQ)
        bottomleft = self.eu**2 * self.pdf(2, x, QQ) * self.ffPion(2, z, QQ)
        bottomright = self.edbar**2 * self.pdf(-1, x, QQ) * self.ffPion(-1, z, QQ)
        return a0*((topleft + topright)/(bottomleft + bottomright))
    
    
class PiMinus(Hadron):
    def __init__(self, kperp2avg=.57, pperp2avg=.12, pdfset='JAM19PDF_proton_nlo',
                 ff_pion='JAM19FF_pion_nlo'):
        '''
        The PiMinus reaction occurs between "d" and "ubar" quarks
        
        :param kperp2avg: average kperp^2
        :param pperp2avg: average pperp^2
        :param pdfset: the name of the pdf grid downloaded from lhapdf that you would like to use
        :param ff_pion: the name of the fragmentation function grid (for pions) downloaded from lhapdf
        '''   
        super().__init__(kperp2avg=kperp2avg, pperp2avg=pperp2avg, pdfset=pdfset,
                 ff_pion=ff_pion)
        
    
    def sivers(self, kins, Nd, ad, bd, Nubar, m1): # aubar, bubar
        '''
        Calculate sivers assymetry for specified variables
        
        :param kins: numpy array w shape (n, 4) of kinematics in order of x, z, pht, QQ (kins[:, 0] = xs)
        :param Nd: free parameter of NN function for u quark (corresponds to N)
        :param ad: free parameter of NN function for u quark (corresponds to alpha)
        :param bd: free parameter of NN function for u quark (corresponds to beat)
        :param Nubar: free parameter of NN function for dbar quark (corresponds to N)
        :param m1: free parameter of A0 function
        
        :returns: length n array of sivers assymetries
        '''
        
        x = kins[:, 0]
        z = kins[:, 1]
        pht = kins[:, 2]
        QQ = kins[:, 3]
        a0 = self.A0(z, pht, m1)
        topleft = self.NN(x, Nd, ad, bd) * self.ed**2 * self.pdf(1, x, QQ) * self.ffPion(1, z, QQ)
        topright = self.NNanti(Nubar) * self.eubar**2 * self.pdf(-2, x, QQ) * self.ffPion(-2, z, QQ)
        bottomleft = self.ed**2 * self.pdf(1, x, QQ) * self.ffPion(1, z, QQ)
        bottomright = self.eubar**2 * self.pdf(-2, x, QQ) * self.ffPion(-2, z, QQ)
        return a0*((topleft + topright)/(bottomleft + bottomright))
    
    
class PiZero(Hadron):
    def __init__(self, kperp2avg=.57, pperp2avg=.12, pdfset='JAM19PDF_proton_nlo',
                 ff_pion='JAM19FF_pion_nlo'):
        '''
        The PiZero reaction occurs between "u" and "ubar" quarks
        
        :param kperp2avg: average kperp^2
        :param pperp2avg: average pperp^2
        :param pdfset: the name of the pdf grid downloaded from lhapdf that you would like to use
        :param ff_pion: the name of the fragmentation function grid (for pions) downloaded from lhapdf
        '''   
        super().__init__(kperp2avg=kperp2avg, pperp2avg=pperp2avg, pdfset=pdfset,
                 ff_pion=ff_pion)
        
    
    def sivers(self, kins, Nu, au, bu, Nubar, m1): # aubar, bubar
        x = kins[:, 0]
        z = kins[:, 1]
        pht = kins[:, 2]
        QQ = kins[:, 3]
        a0 = self.A0(z, pht, m1)
        # N_u * e_u^2 * f_u(x) * (D_pi+/u(z) + D_pi-/u(z))
        topleft = self.NN(x, Nu, au, bu) * self.eu**2 * self.pdf(2, x, QQ) * (self.ffPion(2, z, QQ) + self.ffPion(2, z, QQ))
        # N_ubar * e_ubar^2 * f_ubar(x) * (D_pi+/ubar(z) + D_pi-/ubar(z))
        topright = self.NNanti(Nubar) * self.eubar**2 * self.pdf(-2, x, QQ) * (self.ffPion(-2, z, QQ) + self.ffPion(-2, z, QQ))
        # e_u^2 * f_u(x) * (D_pi+/u(z) + D_pi-/u(z))
        bottomleft = self.eu**2 * self.pdf(2, x, QQ) * (self.ffPion(2, z, QQ) + self.ffPion(2, z, QQ))
        # e_ubar^2 * f_ubar(x) * (D_pi+/ubar(z) + D_pi-/ubar(z))
        bottomright = self.eubar**2 * self.pdf(-2, x, QQ) * (self.ffPion(-2, z, QQ) + self.ffPion(-2, z, QQ))
        return a0*((topleft + topright)/(bottomleft + bottomright))
    
    
class KPlus(Hadron):
    def __init__(self, kperp2avg=.57, pperp2avg=.12, pdfset='JAM19PDF_proton_nlo',
                 ff_kaon='JAM19FF_kaon_nlo'):
        '''
        The KZero reaction occurs between "u" and "sbar" quarks
        
        :param kperp2avg: average kperp^2
        :param pperp2avg: average pperp^2
        :param pdfset: the name of the pdf grid downloaded from lhapdf that you would like to use
        :param ff_kaon: the name of the fragmentation function grid (for kaons) downloaded from lhapdf
        '''   
        super().__init__(kperp2avg=kperp2avg, pperp2avg=pperp2avg, pdfset=pdfset,
                 ff_kaon=ff_kaon)
        
    
    def sivers(self, kins, Nu, au, bu, Nsbar, m1): # asbar, bsbar
        x = kins[:, 0]
        z = kins[:, 1]
        pht = kins[:, 2]
        QQ = kins[:, 3]
        a0 = self.A0(z, pht, m1)
        # N_u * e_u^2 * f_u(x) * (D_k+/u(z) + D_k+/u(z))
        topleft = self.NN(x, Nu, au, bu) * self.eu**2 * self.pdf(2, x, QQ) * self.ffPion(2, z, QQ)
        # N_sbar * e_sbar^2 * f_sbar(x) * (D_k+/sbar(z) + D_k-/sbar(z))
        topright = self.NNanti(Nsbar) * self.esbar**2 * self.pdf(-3, x, QQ) * self.ffKaon(-3, z, QQ)
        # e_u^2 * f_u(x) * (D_pi+/u(z) + D_pi-/u(z))
        bottomleft = self.eu**2 * self.pdf(2, x, QQ) * self.ffPion(2, z, QQ)
        # e_sbar^2 * f_sbar(x) * (D_pi+/sbar(z) + D_pi-/sbar(z))
        bottomright = self.esbar**2 * self.pdf(-3, x, QQ) * self.ffKaon(-3, z, QQ)
        return a0*((topleft + topright)/(bottomleft + bottomright))
    
    
class KMinus(Hadron):
    def __init__(self, kperp2avg=.57, pperp2avg=.12, pdfset='JAM19PDF_proton_nlo',
                 ff_kaon='JAM19FF_kaon_nlo'):
        '''
        The KMinus reaction occurs between "ubar" and "s" quarks
        
        :param kperp2avg: average kperp^2
        :param pperp2avg: average pperp^2
        :param pdfset: the name of the pdf grid downloaded from lhapdf that you would like to use
        :param ff_kaon: the name of the fragmentation function grid (for kaons) downloaded from lhapdf
        '''   
        super().__init__(kperp2avg=kperp2avg, pperp2avg=pperp2avg, pdfset=pdfset,
                 ff_kaon=ff_kaon)
        
    
    def sivers(self, kins, Ns, as0, bs, Nubar, m1): # asbar, bsbar
        x = kins[:, 0]
        z = kins[:, 1]
        pht = kins[:, 2]
        QQ = kins[:, 3]
        a0 = self.A0(z, pht, m1)
        # N_s * e_s^2 * f_s(x) * (D_k+/s(z) + D_k+/s(z))
        topleft = self.NN(x, Ns, as0, bs) * self.es**2 * self.pdf(3, x, QQ) * self.ffKaon(3, z, QQ)
        # N_ubar * e_ubar^2 * f_ubar(x) * (D_k+/ubar(z) + D_k-/ubar(z))
        topright = self.NNanti(Nubar) * self.eubar**2 * self.pdf(-2, x, QQ) * self.ffPion(-2, z, QQ)
        # e_s^2 * f_s(x) * (D_pi+/s(z) + D_pi-/s(z))
        bottomleft = self.es**2 * self.pdf(3, x, QQ) * self.ffKaon(3, z, QQ)
        # e_ubar^2 * f_ubar(x) * (D_pi+/ubar(z) + D_pi-/ubar(z))
        bottomright = self.eubar**2 * self.pdf(-2, x, QQ) * self.ffPion(-2, z, QQ)
        return a0*((topleft + topright)/(bottomleft + bottomright))
    
    
class CombinedHadrons(object):
    def __init__(self, kperp2avg=.57, pperp2avg=.12, pdfset='JAM19PDF_proton_nlo',
             ff_kaon='JAM19FF_kaon_nlo', ff_pion='JAM19FF_pion_nlo'):
        
        self.funcDict = {'pi+': PiPlus(kperp2avg=kperp2avg, pperp2avg=pperp2avg, pdfset=pdfset,
                     ff_pion=ff_pion),
                         'pi-': PiMinus(kperp2avg=kperp2avg, pperp2avg=pperp2avg, pdfset=pdfset,
                     ff_pion=ff_pion), 
                         'pi0': PiZero(kperp2avg=kperp2avg, pperp2avg=pperp2avg, pdfset=pdfset,
                     ff_pion=ff_pion), 
                         'k+': KPlus(kperp2avg=kperp2avg, pperp2avg=pperp2avg, pdfset=pdfset,
                     ff_kaon=ff_kaon), 
                         'k-': KMinus(kperp2avg=kperp2avg, pperp2avg=pperp2avg, pdfset=pdfset,
                     ff_kaon=ff_kaon)
                     }
        
        self.argDict = {'pi+': ['Nu', 'au', 'bu', 'Ndbar', 'm1'],
                        'pi-': ['Nd', 'ad', 'bd', 'Nubar', 'm1'],
                        'pi0': ['Nu', 'au', 'bu', 'Nubar', 'm1'],
                        'k+': ['Nu', 'au', 'bu', 'Nsbar', 'm1'],
                        'k-': ['Ns', 'as0', 'bs', 'Nubar', 'm1']
                       }
        
        
    def siversAll(self, kinsandflag, **params):
        '''
        Calculate sivers assymetry for specified variables
        
        :param kins: numpy array w shape (n, 5) of kinematics in order of x, z, pht, QQ (kins[:, 0] = xs) and then a flag variable which contains ('pi+', 'pi-', 'pi0', 'k+', 'k-')
        :param params: a dictionary of some combination of the 13 free parameters of sivers functions for the various hadron functions (should be labeled as: Nu, Nd, Ns, Nubar, Ndbar, Nsbar, au, ad, as0, bu, bd, bs, m1)
        
        :returns: length n array of sivers assymetries
        '''
        
        #funcs = [self.pp, self.pm, self.pz, self.kp, self.km]
        
        results = []
        for hadrn in np.unique(kinsandflag[:, 4]):
            kins = kinsandflag[kinsandflag[:, 4] == hadrn, :4].astype('float')
            args = [params[x] for x in self.argDict[hadrn]]
            results += list(self.funcDict[hadrn].sivers(kins, *args))
        
        return np.array(results)

   
#      def siversAllCurveFitFactory(self, kinsandflag, **kwargs):
        
#          args = kwargs.keys()
        
#          l1 = 'def siversAllCurveFit(kins, ' + ', '.join(args) + '):\r'
#          l2 = '    '
        
#          exec()


    def siversAllCurveFit(self, kinsandflag, Nu, Nd, Ns, Nubar, Ndbar, Nsbar, au, ad, as0, bu, bd, bs, m1):
        res = []
        kinspp = kinsandflag[kinsandflag[:, 4] == 'pi+', :4].astype('float') 
        res += list(self.funcDict['pi+'].sivers(kinspp, Nu, au, bu, Ndbar, m1))
        kinspm = kinsandflag[kinsandflag[:, 4] == 'pi-', :4].astype('float') 
        res += list(self.funcDict['pi-'].sivers(kinspm, Nd, ad, bd, Nubar, m1))
        kinsp0 = kinsandflag[kinsandflag[:, 4] == 'pi0', :4].astype('float') 
        res += list(self.funcDict['pi0'].sivers(kinsp0, Nu, au, bu, Nubar, m1))
        kinskp = kinsandflag[kinsandflag[:, 4] == 'k+', :4].astype('float') 
        res += list(self.funcDict['k+'].sivers(kinskp, Nu, au, bu, Nsbar, m1))
        kinskm = kinsandflag[kinsandflag[:, 4] == 'k-', :4].astype('float') 
        res += list(self.funcDict['k-'].sivers(kinskm, Ns, as0, bs, Nubar, m1))
        
        return np.array(res)
    
    
    def siversPionCurveFit(self, kinsandflag, Nu, Nd, Nubar, Ndbar, au, ad, bu, bd, m1):
    
        res = []
        kinspp = kinsandflag[kinsandflag[4] == 'pi+', :4].astype('float') 
        res.append(self.funcDict['pi+'].sivers(kinspp, Nu, au, bu, Ndbar, m1))
        kinspm = kinsandflag[kinsandflag[4] == 'pi-', :4].astype('float') 
        res.append(self.funcDict['pi-'].sivers(kinspm, Nd, ad, bd, Nubar, m1))
        kinsp0 = kinsandflag[kinsandflag[4] == 'pi0', :4].astype('float') 
        res.append(self.funcDict['pi0'].sivers(kinsp0, Nu, au, bu, Nubar, m1))
        
        return np.array(res)
    
    def siversAllCurveFitFactorized(self, kinsandflag, Nu, Nd, Ns, Nubar, Ndbar, Nsbar, au, ad, as0, bu, bd, bs, m1):

        res = []
        kinspp = kinsandflag[kinsandflag[:, 4] == 0, :4]
        res += list(self.funcDict['pi+'].sivers(kinspp, Nu, au, bu, Ndbar, m1))
        kinspm = kinsandflag[kinsandflag[:, 4] == 1, :4] 
        res += list(self.funcDict['pi-'].sivers(kinspm, Nd, ad, bd, Nubar, m1))
        kinsp0 = kinsandflag[kinsandflag[:, 4] == 2, :4] 
        res += list(self.funcDict['pi0'].sivers(kinsp0, Nu, au, bu, Nubar, m1))
        kinskp = kinsandflag[kinsandflag[:, 4] == 3, :4] 
        res += list(self.funcDict['k+'].sivers(kinskp, Nu, au, bu, Nsbar, m1))
        kinskm = kinsandflag[kinsandflag[:, 4] == 4, :4] 
        res += list(self.funcDict['k-'].sivers(kinskm, Ns, as0, bs, Nubar, m1))
        
        return np.array(res)
    
    def siversAllCurveFitFactorizeds0(self, kinsandflag, Nu, Nd, Nubar, Ndbar, au, ad, bu, bd, m1):

        res = []
        kinspp = kinsandflag[kinsandflag[:, 4] == 0, :4]
        res += list(self.funcDict['pi+'].sivers(kinspp, Nu, au, bu, Ndbar, m1))
        kinspm = kinsandflag[kinsandflag[:, 4] == 1, :4] 
        res += list(self.funcDict['pi-'].sivers(kinspm, Nd, ad, bd, Nubar, m1))
        kinsp0 = kinsandflag[kinsandflag[:, 4] == 2, :4] 
        res += list(self.funcDict['pi0'].sivers(kinsp0, Nu, au, bu, Nubar, m1))
        kinskp = kinsandflag[kinsandflag[:, 4] == 3, :4] 
        res += list(self.funcDict['k+'].sivers(kinskp, Nu, au, bu, 0, m1))
        kinskm = kinsandflag[kinsandflag[:, 4] == 4, :4] 
        res += list(self.funcDict['k-'].sivers(kinskm, 0, 0, 0, Nubar, m1))
        
        return np.array(res)