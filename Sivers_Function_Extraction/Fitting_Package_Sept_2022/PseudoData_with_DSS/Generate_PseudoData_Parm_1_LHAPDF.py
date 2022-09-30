import numpy as np
import pandas as pd
import lhapdf
import os


Mp=0.938272
alpha_s=1/(137.0359998)

Kp2A=0.57
Pp2A=0.12
p2unp=0.25

ee=1
eU=2/3
eUbar=-2/3
eD=-1/3
eDbar=1/3
eS=-1/3
eSbar=1/3

qCharge=np.array([eSbar,eUbar,eDbar,eU,eD,eS])
qFlavor=np.array([-3,-2,-1,1,2,3])

SIGN = 1

### NNq parameterization ####

def NNq(x,Nq,aq,bq):
    tempNNq = Nq*(x**aq)*((1-x)**(bq))*((aq+bq)**(aq+bq))/((aq**aq)*(bq**bq))
    return tempNNq

def NNqbar(x,Nqbar):
    tempNNqbar = Nqbar
    return tempNNqbar



## DY ###
DY_DataFilesArray=np.array(['./Data/COMPASS_p_DY_2017.csv'])

## SIDIS ###
Dat1='./Data/HERMES_p_2009.csv'
Dat2='./Data/HERMES_p_2020.csv'
Dat3='./Data/COMPASS_d_2009.csv'
Dat4='./Data/COMPASS_p_2015.csv'
SIDIS_DataFilesArray=[Dat1,Dat2,Dat3,Dat4]




test_pars=([7.0, 0.89, 2.78, 19.4, -0.07, -2.33, 2.5, 15.8, -0.29, -14, 4.9, 3, 0])
test_errs=([0.6, 0.05, 0.17, 1.6, 0.06, 0.31, 0.4, 3.2, 0.27, 10, 3.3, 2, 0.18])


m1v = test_pars[0]

Nuv = test_pars[1]
auv = test_pars[2]
buv = test_pars[3]

Nubv = test_pars[4]
# aubv = 0.01
# bubv = 0.01

Ndv = test_pars[5]
adv = test_pars[6]
bdv = test_pars[7]

Ndbv = test_pars[8]
# adbv = 0.01
# bdbv = 0.01

Nsv = test_pars[9]
asv = test_pars[10]
bsv = test_pars[11]

Nsbv = test_pars[12]
# asbv = 0.01
# bsbv = 0.01


class Hadron(object):
    # def __init__(self, kperp2avg=.57, pperp2avg=.12, pdfset='cteq61',
    #              ff_PIp='NNFF10_PIp_nlo', ff_PIm='NNFF10_PIm_nlo', ff_PIsum='NNFF10_PIsum_nlo',
    #              ff_KAp='NNFF10_KAp_nlo', ff_KAm='NNFF10_KAm_nlo'):
    def __init__(self, kperp2avg=.57, pperp2avg=.12, pdfset='cteq61',
                 ff_PIp='DSS14_NLO_Pip', ff_PIm='DSS14_NLO_Pim', ff_PIsum='DSS14_NLO_PiSum',
                 ff_KAp='DSS17_NLO_KaonPlus', ff_KAm='DSS17_NLO_KaonMinus'):

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
    
    
    
class Sivers_Hadron(Hadron):
    # def __init__(self, kperp2avg=.57, pperp2avg=.12, pdfset='cteq61',
    #              ff_PIp='NNFF10_PIp_nlo', ff_PIm='NNFF10_PIm_nlo', ff_PIsum='NNFF10_PIsum_nlo',
    #              ff_KAp='NNFF10_KAp_nlo', ff_KAm='NNFF10_KAm_nlo'):
    def __init__(self, kperp2avg=.57, pperp2avg=.12, pdfset='cteq61',
                 ff_PIp='DSS14_NLO_Pip', ff_PIm='DSS14_NLO_Pim', ff_PIsum='DSS14_NLO_PiSum',
                 ff_KAp='DSS17_NLO_KaonPlus', ff_KAm='DSS17_NLO_KaonMinus'):
                     
        super().__init__(kperp2avg=kperp2avg, pperp2avg=pperp2avg, pdfset=pdfset)

        
    def sivers(self, had, kins, m1, Nu, au, bu, Nub, Nd, ad, bd, Ndb, NS, aS, bS, NSb):
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
        + NNq(x, NS, aS, bS) * self.es**2 * self.pdf(3, x, QQ) * self.ff(self.ffDict[ii],3, z, QQ)
        + NNqbar(x, NSb) * self.esbar**2 * self.pdf(-3, x, QQ) * self.ff(self.ffDict[ii],-3, z, QQ)
        temp_bottom =  self.eu**2 * self.pdf(2, x, QQ) * self.ff(self.ffDict[ii],2, z, QQ)
        + self.eubar**2 * self.pdf(-2, x, QQ) * self.ff(self.ffDict[ii],-2, z, QQ)
        + self.ed**2 * self.pdf(1, x, QQ) * self.ff(self.ffDict[ii],1, z, QQ)
        + self.edbar**2 * self.pdf(-1, x, QQ) * self.ff(self.ffDict[ii],-1, z, QQ)
        + self.es**2 * self.pdf(3, x, QQ) * self.ff(self.ffDict[ii],3, z, QQ)
        + self.esbar**2 * self.pdf(-3, x, QQ) * self.ff(self.ffDict[ii],-3, z, QQ)
        temp_siv_had = a0*((temp_top)/(temp_bottom))
        #print(temp_siv_had)
        return temp_siv_had
 


import copy
def Create_SIDIS_P_Data(datafile, m1, Nu, au, bu, Nub, Nd, ad, bd, Ndb, NS, aS, bS, NSb):
    tempdf=pd.read_csv(datafile)
    temphad=np.array(tempdf['hadron'],dtype=object)
    tempQ2=np.array(tempdf['Q2'],dtype=object)
    tempX=np.array(tempdf['x'],dtype=object)
    tempY=np.array(tempdf['y'],dtype=object)
    tempZ=np.array(tempdf['z'],dtype=object)
    tempPHT=np.array(tempdf['phT'],dtype=object)
    tempSivErr=np.array(tempdf['tot_err'],dtype=object)
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
    PiP=copy.deepcopy(data_dictionary)
    PiM=copy.deepcopy(data_dictionary)
    Pi0=copy.deepcopy(data_dictionary)
    KP=copy.deepcopy(data_dictionary)
    KM=copy.deepcopy(data_dictionary)
    SivHad=functions_develop.Sivers_Hadron()
    ############################################
    temp_Siv=[]
    for i in range(len(temphad)):
        temp=np.array([[data_dictionary["x"][i],data_dictionary["z"][i],
                        data_dictionary["phT"][i],data_dictionary["Q2"][i]]])
        temp_had=data_dictionary["hadron"][i]
        #print(temp_had)
        temp_Siv.append(SivHad.sivers(temp_had,temp, m1, Nu, au, bu, Nub, Nd, ad, bd, Ndb, NS, aS, bS, NSb)[0])
    ############################################
    data_dictionary["Siv"]=np.array(temp_Siv)
    return pd.DataFrame(data_dictionary)


OutputFolder='PseudoData_Parm1_LHAPDF'
os.mkdir(OutputFolder)

Pseudo_SIDIS_HERMES2009=Create_SIDIS_P_Data(Dat1,m1v,Nuv,auv,buv,Nubv,Ndv,adv,bdv,Ndbv,Nsv,asv,bsv,Nsbv)
Pseudo_SIDIS_HERMES2009.to_csv(str(OutputFolder)+'/HERMES2009_Pseudo.csv')

Pseudo_SIDIS_HERMES2020=Create_SIDIS_P_Data(Dat2,m1v,Nuv,auv,buv,Nubv,Ndv,adv,bdv,Ndbv,Nsv,asv,bsv,Nsbv)
Pseudo_SIDIS_HERMES2020.to_csv(str(OutputFolder)+'/HERMES2020_Pseudo.csv')

Pseudo_SIDIS_COMPASS2009=Create_SIDIS_P_Data(Dat3,m1v,Nuv,auv,buv,Nubv,Ndv,adv,bdv,Ndbv,Nsv,asv,bsv,Nsbv)
Pseudo_SIDIS_COMPASS2009.to_csv(str(OutputFolder)+'/COMPASS2009_Pseudo.csv')

Pseudo_SIDIS_COMPASS2015.to_csv(str(OutputFolder)+'/COMPASS2015_Pseudo.csv')
Pseudo_SIDIS_COMPASS2015=Create_SIDIS_P_Data(Dat4,m1v,Nuv,auv,buv,Nubv,Ndv,adv,bdv,Ndbv,Nsv,asv,bsv,Nsbv)