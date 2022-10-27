import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lhapdf
import os

Plots_Folder = 'NN_SIDIS_Plots'
os.mkdir(str(Plots_Folder))

CSVs_Folder = 'NN_SIDIS_Fit_Results'

##############################################################################

class DataANN(object):
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

        y = []
        err = []

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
            data['uexpr'] += list(self.eu**2 * self.pdf(2, x, QQ) * self.ff(self.ffDict[i], 2, z, QQ))
            data['ubarexpr'] += list(self.eubar**2 * self.pdf(-2, x, QQ) * self.ff(self.ffDict[i], -2, z, QQ))
            data['dexpr'] += list(self.ed**2 * self.pdf(1, x, QQ) * self.ff(self.ffDict[i], 1, z, QQ))
            data['dbarexpr'] += list(self.edbar**2 * self.pdf(-1, x, QQ) * self.ff(self.ffDict[i], -1, z, QQ))
            data['sexpr'] += list(self.es**2 * self.pdf(3, x, QQ) * self.ff(self.ffDict[i], 3, z, QQ))
            data['sbarexpr'] += list(self.esbar**2 * self.pdf(-3, x, QQ) * self.ff(self.ffDict[i], -3, z, QQ))

            data['x'] += list(x)
            data['z'] += list(z)
            data['phT'] += list(sliced['phT'])

        for key in data.keys():
            data[key] = np.array(data[key])

        return data, data[dependencies[0]], np.array(y), np.array(err)


datann = DataANN()



#################################################################################

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

def NNq(x,Nq,aq,bq):
    tempNNq = Nq*(x**aq)*((1-x)**(bq))*((aq+bq)**(aq+bq))/((aq**aq)*(bq**bq))
    return tempNNq

def NNqbar(x,Nq,aq,bq):
    tempNNqbar = Nq*(x**aq)*((1-x)**(bq))*((aq+bq)**(aq+bq))/((aq**aq)*(bq**bq))
    return tempNNqbar


#parms_df = pd.read_csv('Parameters.csv')
test_pars=([7.0, 0.89, 2.50, 20, -0.12, 0.4,  20, -2.4, 2.7, 17, -0.7, 1.5, 20, -20, 4.7, 2.3, 20, 9.5, 20])
test_errs=([  4, 0.06, 0.11,  2,  0.06, 0.35, 16,  0.4, 0.6,  4,  0.5, 0.6, 17,  40, 3.0, 2.2,  5, 1.4, 14])


#######################################################################
########## Sivers Function ############################################
#######################################################################

PDFdataset = lhapdf.mkPDF("cteq61")

def xFxQ2(dataset,flavor,x,QQ):
    temp_parton_dist_x=np.array(dataset.xfxQ2(flavor, x, QQ))
    return temp_parton_dist_x
    
def hfunc(kp,m1):
    temphfunc=np.sqrt(2*ee)*(kp/m1)*(np.exp((-kp**2)/(m1**2)))
    return temphfunc

def SiversFuncQ(dataset,flavor,x,QQ,kp,fitresult):
    tempM1=fitresult[0]
    if(flavor==2):
        Nq=fitresult[1]
        aq=fitresult[2]
        bq=fitresult[3]
        tempsiv=2*NNq(x,Nq,aq,bq)*hfunc(kp,tempM1)*(np.exp((-kp**2)/(Kp2A)))*(1/((np.pi)*(Kp2A)))*(xFxQ2(dataset,flavor,x,QQ))
    if(flavor==1):
        Nq=fitresult[7]
        aq=fitresult[8]
        bq=fitresult[9]
        tempsiv=2*NNq(x,Nq,aq,bq)*hfunc(kp,tempM1)*(np.exp((-kp**2)/(Kp2A)))*(1/((np.pi)*(Kp2A)))*(xFxQ2(dataset,flavor,x,QQ))
    if(flavor==3):
        Nq=fitresult[13]
        aq=fitresult[14]
        bq=fitresult[15]
        tempsiv=2*NNq(x,Nq,aq,bq)*hfunc(kp,tempM1)*(np.exp((-kp**2)/(Kp2A)))*(1/((np.pi)*(Kp2A)))*(xFxQ2(dataset,flavor,x,QQ))
    return tempsiv
    
    
def SiversFuncAntiQ(dataset,flavor,x,QQ,kp,fitresult):
    tempM1=fitresult[0]
    if(flavor==-2):
        tempM1=fitresult[0]
        Nq=fitresult[4]
        aq=fitresult[5]
        bq=fitresult[6]
        tempsiv=2*NNqbar(x,Nq,aq,bq)*hfunc(kp,tempM1)*(np.exp((-kp)/(Kp2A**2)))*(1/((np.pi)*(Kp2A)))*(xFxQ2(dataset,flavor,x,QQ))
    if(flavor==-1):
        tempM1=fitresult[0]
        Nq=fitresult[10]
        aq=fitresult[11]
        bq=fitresult[12]
        tempsiv=2*NNqbar(x,Nq,aq,bq)*hfunc(kp,tempM1)*(np.exp((-kp**2)/(Kp2A)))*(1/((np.pi)*(Kp2A)))*(xFxQ2(dataset,flavor,x,QQ))
    if(flavor==-3):
        tempM1=fitresult[0]
        Nq=fitresult[16]
        aq=fitresult[17]
        bq=fitresult[18]
        tempsiv=2*NNqbar(x,Nq,aq,bq)*hfunc(kp,tempM1)*(np.exp((-kp**2)/(Kp2A)))*(1/((np.pi)*(Kp2A)))*(xFxQ2(dataset,flavor,x,QQ))
    return tempsiv
    

def plotSiversQ(flavor,ParmResults,col,lbl):
    tempkT=np.linspace(0, 1.5)
    tempSiv=[SiversFuncQ(PDFdataset,flavor,0.1,2.4,tempkT[i],ParmResults) for i in range(0,len(tempkT))]
    plt.plot(tempkT,tempSiv, '--', color = col, label = lbl)
    #return tempSiv

def plotSiversAntiQ(flavor,ParmResults, col,lbl):
    tempkT=np.linspace(0, 1.5)
    tempSiv=[SiversFuncAntiQ(PDFdataset,flavor,0.1,2.4,tempkT[i],ParmResults) for i in range(0,len(tempkT))]
    plt.plot(tempkT,tempSiv, '--',color = col, label = lbl)
    


Sivers_CSV_file = pd.read_csv(str(CSVs_Folder)+'/'+'Sivfuncs.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
Sivers_CSV_df = pd.concat([Sivers_CSV_file])


def QSiversPlots(tempdf):
    tempKT = np.array(tempdf['kperp'])
    tempfu = np.array(tempdf['fu'])
    tempfuErr = np.array(tempdf['fuErr'])
    tempfd = np.array(tempdf['fd'])
    tempfdErr = np.array(tempdf['fdErr'])
    tempfs = np.array(tempdf['fs'])
    tempfsErr = np.array(tempdf['fsErr'])
    plt.plot(tempKT, tempfu, 'b', label='$u$')
    plt.fill_between(tempKT, tempfu-tempfuErr, tempfu+tempfuErr, facecolor='b', alpha=0.3)
    plotSiversQ(2,test_pars, 'b', '$u_{pseudo}$')
    plt.plot(tempKT, tempfd, 'r', label='$d$')
    plt.fill_between(tempKT, tempfd-tempfdErr, tempfd+tempfdErr, facecolor='r', alpha=0.3)
    plotSiversQ(1,test_pars, 'r', '$d_{pseudo}$')
    plt.plot(tempKT, tempfs, 'g', label='$s$')
    plt.fill_between(tempKT, tempfs-tempfsErr, tempfs+tempfsErr, facecolor='g', alpha=0.3)
    plotSiversQ(3,test_pars, 'g', '$s_{pseudo}$')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(-0.12,0.12)
    plt.legend(loc=4,fontsize=20,handlelength=3)
    plt.savefig(str(Plots_Folder)+'/'+'SiversQ_SIDIS_NN.pdf', format='pdf', bbox_inches='tight')
    

def AntiQSiversPlots(tempdf):
    tempKT = np.array(tempdf['kperp'])
    tempfu = np.array(tempdf['fubar'])
    tempfuErr = np.array(tempdf['fubarErr'])
    tempfd = np.array(tempdf['fdbar'])
    tempfdErr = np.array(tempdf['fdbarErr'])
    tempfs = np.array(tempdf['fsbar'])
    tempfsErr = np.array(tempdf['fsbarErr'])
    plt.plot(tempKT, tempfu, 'b', label='$\\bar{u}$')
    plt.fill_between(tempKT, tempfu-tempfuErr, tempfu+tempfuErr, facecolor='b', alpha=0.3)
    plotSiversAntiQ(-2,test_pars, 'b', '$\\bar{u}_{pseudo}$')
    plt.plot(tempKT, tempfd, 'r', label='$\\bar{d}$')
    plt.fill_between(tempKT, tempfd-tempfdErr, tempfd+tempfdErr, facecolor='r', alpha=0.3)
    plotSiversAntiQ(-1,test_pars, 'r', '$\\bar{d}_{pseudo}$')
    plt.plot(tempKT, tempfs, 'g', label='$\\bar{s}$')
    plt.fill_between(tempKT, tempfs-tempfsErr, tempfs+tempfsErr, facecolor='g', alpha=0.3)
    plotSiversAntiQ(-3,test_pars, 'g', '$\\bar{s}_{pseudo}$')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(-0.012,0.012)
    plt.legend(loc=4,fontsize=20,handlelength=3)
    plt.savefig(str(Plots_Folder)+'/'+'Sivers_AntiQ_SIDIS_NN.pdf', format='pdf', bbox_inches='tight')
    
    
fig7=plt.figure(7)    
QSiversPlots(Sivers_CSV_df)
fig8=plt.figure(8)
AntiQSiversPlots(Sivers_CSV_df)