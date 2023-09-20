import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lhapdf
import os

Plots_Folder = './NN_SIDIS_Plots'
#os.mkdir(str(Plots_Folder))
######################################################################
######################    Results files ##############################
######################################################################
CSVs_Folder = './NN_SIDIS_Fit_Results'
folders_array=os.listdir(CSVs_Folder)
# HERMES09NN = pd.read_csv(str(CSVs_Folder)+'/'+'Result_SIDIS_HERMES2009.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# HERMES09NN_df = pd.concat([HERMES09NN])
# HERMES20NN = pd.read_csv(str(CSVs_Folder)+'/'+'Result_SIDIS_HERMES2020.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# HERMES20NN_df = pd.concat([HERMES20NN])
# COMPASS09NN = pd.read_csv(str(CSVs_Folder)+'/'+'Result_SIDIS_COMPASS2009.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# COMPASS09NN_df = pd.concat([COMPASS09NN])
# COMPASS15NN = pd.read_csv(str(CSVs_Folder)+'/'+'Result_SIDIS_COMPASS2015.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# COMPASS15NN_df = pd.concat([COMPASS15NN])

# ######################################################################
# #########################    Data files ##############################
# ######################################################################

# Data_Folder = '../Data'
# HERMES09ex = pd.read_csv(str(Data_Folder)+'/'+'HERMES_p_2009.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# HERMES09ex_df = pd.concat([HERMES09ex])
# HERMES20ex = pd.read_csv(str(Data_Folder)+'/'+'HERMES_p_2020.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# HERMES20ex_df = pd.concat([HERMES20ex])
# COMPASS09ex = pd.read_csv(str(Data_Folder)+'/'+'COMPASS_d_2009.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# COMPASS09ex_df = pd.concat([COMPASS09ex])
# COMPASS15ex = pd.read_csv(str(Data_Folder)+'/'+'COMPASS_p_2015.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# COMPASS15ex_df = pd.concat([COMPASS15ex])
# COMPASS17ex = pd.read_csv('../Data/COMPASS_p_DY_2017.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# COMPASS17ex_df = pd.concat([COMPASS17ex])

# Data_Folder = '../PseudoData_from_NN_01/'
# HERMES09ex = pd.read_csv(str(Data_Folder)+'/'+'HERMES2009_3DPseudo.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# HERMES09ex_df = pd.concat([HERMES09ex])
# HERMES20ex = pd.read_csv(str(Data_Folder)+'/'+'HERMES2020_3DPseudo.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# HERMES20ex_df = pd.concat([HERMES20ex])
# COMPASS09ex = pd.read_csv(str(Data_Folder)+'/'+'COMPASS2009_3DPseudo.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# COMPASS09ex_df = pd.concat([COMPASS09ex])
# COMPASS15ex = pd.read_csv(str(Data_Folder)+'/'+'COMPASS2015_3DPseudo.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# COMPASS15ex_df = pd.concat([COMPASS15ex])


# COMPASS17NN_minus = pd.read_csv(str(CSVs_Folder)+'/'+'Result_COMPASS_DY_from_SIDIS_minus.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# COMPASS17NN_minus_df = pd.concat([COMPASS17NN_minus])

# COMPASS17NN_plus = pd.read_csv(str(CSVs_Folder)+'/'+'Result_COMPASS_DY_from_SIDIS_plus.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
# COMPASS17NN_plus_df = pd.concat([COMPASS17NN_plus])


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

##############################################################################

def chisquare(y, yhat, err):
    return np.sum(((y - yhat)/err)**2)
    
    
# def plotSivAsymmBands(resultdf, datasetdf, hadron, dependence):
#     D_Xplt, D_DEP, D_yplt, D_errplt = datann.makeData(datasetdf, [hadron], [dependence])
#     R_Xplt, R_DEP, R_yplt, R_errplt = datann.makeData(resultdf, [hadron], [dependence])
#     chi2val=chisquare(D_yplt, R_yplt, D_errplt)
#     plt.errorbar(D_DEP, D_yplt,yerr=D_errplt, fmt='bo',label='Data')
#     plt.errorbar(R_DEP, R_yplt,yerr=R_errplt, fmt='ro',label='NN')
#     #### Here the user needs to define the plot title based on the data set ####
#     plt.title('SIDIS Sivers '+str(hadron)+' $\chi2$='+str('%.2f'%chi2val),fontsize=15)
#     #plt.ylim([-0.001,0.001])
#     plt.xlabel(str(dependence),fontsize=15)
#     plt.legend(loc=2,fontsize=15,handlelength=3)

def plotSivAsymmBands(resultdf, datasetdf, hadron, dependence):
    D_Xplt, D_DEP, D_yplt, D_errplt = datann.makeData(datasetdf, [hadron], [dependence])
    R_Xplt, R_DEP, R_yplt, R_errplt = datann.makeData(resultdf, [hadron], [dependence])
    chi2val=chisquare(D_yplt, R_yplt, D_errplt)
    plt.errorbar(D_DEP, D_yplt,yerr=D_errplt, fmt='bo',label='Data')
    plt.errorbar(R_DEP, R_yplt,yerr=R_errplt, fmt='ro',label='NN')
    #### Here the user needs to define the plot title based on the data set ####
    #plt.title('SIDIS Sivers '+str(hadron)+' $\chi2$='+str('%.2f'%chi2val),fontsize=15)
    if(str(hadron)=='pi+'):
        plt.text(np.min(D_DEP),0.13,'$\chi^2$('+'$\pi^+$'+')='+str('%.2f'%chi2val),fontsize=16)
    if(str(hadron)=='pi-'):
        plt.text(np.min(D_DEP),0.13,'$\chi^2$('+'$\pi^-$'+')='+str('%.2f'%chi2val),fontsize=16)
    if(str(hadron)=='pi0'):
        plt.text(np.min(D_DEP),0.13,'$\chi^2$('+'$\pi^0$'+')='+str('%.2f'%chi2val),fontsize=16)
    if(str(hadron)=='k+'):
        plt.text(np.min(D_DEP),0.13,'$\chi^2$('+'$k^+$'+')='+str('%.2f'%chi2val),fontsize=16)
    if(str(hadron)=='k-'):
        plt.text(np.min(D_DEP),0.13,'$\chi^2$('+'$k^-$'+')='+str('%.2f'%chi2val),fontsize=16)
    #plt.text(np.min(D_DEP),-0.07,'$\chi2$('+str(hadron)+')='+str('%.2f'%chi2val),fontsize=16)
    plt.ylim([-0.1,0.15])
    if(str(dependence)=='x'):
        plt.xlabel('$x$',fontsize=16)
    if(str(dependence)=='z'):
        plt.xlabel('$z$',fontsize=16)
    if(str(dependence)=='phT'):
        plt.xlabel('$p_{hT}$',fontsize=16)
    plt.legend(loc=4,fontsize=15,handlelength=3)
    


# fig1=plt.figure(1,figsize=(15,30))
# plt.subplot(5,3,1)
# plotSivAsymmBands(HERMES09NN_df,HERMES09ex_df,'pi+','x')
# plt.subplot(5,3,2)
# plotSivAsymmBands(HERMES09NN_df,HERMES09ex_df,'pi+','z')
# plt.subplot(5,3,3)
# plotSivAsymmBands(HERMES09NN_df,HERMES09ex_df,'pi+','phT')
# plt.subplot(5,3,4)
# plotSivAsymmBands(HERMES09NN_df,HERMES09ex_df,'pi-','x')
# plt.subplot(5,3,5)
# plotSivAsymmBands(HERMES09NN_df,HERMES09ex_df,'pi-','z')
# plt.subplot(5,3,6)
# plotSivAsymmBands(HERMES09NN_df,HERMES09ex_df,'pi-','phT')
# plt.subplot(5,3,7)
# plotSivAsymmBands(HERMES09NN_df,HERMES09ex_df,'pi0','x')
# plt.subplot(5,3,8)
# plotSivAsymmBands(HERMES09NN_df,HERMES09ex_df,'pi0','z')
# plt.subplot(5,3,9)
# plotSivAsymmBands(HERMES09NN_df,HERMES09ex_df,'pi0','phT')
# plt.subplot(5,3,10)
# plotSivAsymmBands(HERMES09NN_df,HERMES09ex_df,'k+','x')
# plt.subplot(5,3,11)
# plotSivAsymmBands(HERMES09NN_df,HERMES09ex_df,'k+','z')
# plt.subplot(5,3,12)
# plotSivAsymmBands(HERMES09NN_df,HERMES09ex_df,'k+','phT')
# plt.subplot(5,3,13)
# plotSivAsymmBands(HERMES09NN_df,HERMES09ex_df,'k-','x')
# plt.subplot(5,3,14)
# plotSivAsymmBands(HERMES09NN_df,HERMES09ex_df,'k-','z')
# plt.subplot(5,3,15)
# plotSivAsymmBands(HERMES09NN_df,HERMES09ex_df,'k-','phT')
# #### Here the user needs to define the plot title based on the data set ####
# #plt.savefig('HERMES09_'+str(LOSSFN)+'_'+str(EPOCHS)+'Ep_'+str(HL)+'H_'+str(NODES)+'N_'+str(LR)+'LR.pdf', format='pdf', bbox_inches='tight')
# plt.savefig(str(Plots_Folder)+'/'+'HERMES09.pdf', format='pdf', bbox_inches='tight')

# fig2=plt.figure(2,figsize=(15,30))
# plt.subplot(5,3,1)
# plotSivAsymmBands(HERMES20NN_df,HERMES20ex_df,'pi+','x')
# plt.subplot(5,3,2)
# plotSivAsymmBands(HERMES20NN_df,HERMES20ex_df,'pi+','z')
# plt.subplot(5,3,3)
# plotSivAsymmBands(HERMES20NN_df,HERMES20ex_df,'pi+','phT')
# plt.subplot(5,3,4)
# plotSivAsymmBands(HERMES20NN_df,HERMES20ex_df,'pi-','x')
# plt.subplot(5,3,5)
# plotSivAsymmBands(HERMES20NN_df,HERMES20ex_df,'pi-','z')
# plt.subplot(5,3,6)
# plotSivAsymmBands(HERMES20NN_df,HERMES20ex_df,'pi-','phT')
# plt.subplot(5,3,7)
# plotSivAsymmBands(HERMES20NN_df,HERMES20ex_df,'pi0','x')
# plt.subplot(5,3,8)
# plotSivAsymmBands(HERMES20NN_df,HERMES20ex_df,'pi0','z')
# plt.subplot(5,3,9)
# plotSivAsymmBands(HERMES20NN_df,HERMES20ex_df,'pi0','phT')
# plt.subplot(5,3,10)
# plotSivAsymmBands(HERMES20NN_df,HERMES20ex_df,'k+','x')
# plt.subplot(5,3,11)
# plotSivAsymmBands(HERMES20NN_df,HERMES20ex_df,'k+','z')
# plt.subplot(5,3,12)
# plotSivAsymmBands(HERMES20NN_df,HERMES20ex_df,'k+','phT')
# plt.subplot(5,3,13)
# plotSivAsymmBands(HERMES20NN_df,HERMES20ex_df,'k-','x')
# plt.subplot(5,3,14)
# plotSivAsymmBands(HERMES20NN_df,HERMES20ex_df,'k-','z')
# plt.subplot(5,3,15)
# plotSivAsymmBands(HERMES20NN_df,HERMES20ex_df,'k-','phT')
# plt.savefig(str(Plots_Folder)+'/'+'HERMES20.pdf', format='pdf', bbox_inches='tight')


# fig3=plt.figure(3,figsize=(15,30))
# plt.subplot(4,3,1)
# plotSivAsymmBands(COMPASS09NN_df,COMPASS09ex_df,'pi+','x')
# plt.subplot(4,3,2)
# plotSivAsymmBands(COMPASS09NN_df,COMPASS09ex_df,'pi+','z')
# plt.subplot(4,3,3)
# plotSivAsymmBands(COMPASS09NN_df,COMPASS09ex_df,'pi+','phT')
# plt.subplot(4,3,4)
# plotSivAsymmBands(COMPASS09NN_df,COMPASS09ex_df,'pi-','x')
# plt.subplot(4,3,5)
# plotSivAsymmBands(COMPASS09NN_df,COMPASS09ex_df,'pi-','z')
# plt.subplot(4,3,6)
# plotSivAsymmBands(COMPASS09NN_df,COMPASS09ex_df,'pi-','phT')
# plt.subplot(4,3,7)
# plotSivAsymmBands(COMPASS09NN_df,COMPASS09ex_df,'k+','x')
# plt.subplot(4,3,8)
# plotSivAsymmBands(COMPASS09NN_df,COMPASS09ex_df,'k+','z')
# plt.subplot(4,3,9)
# plotSivAsymmBands(COMPASS09NN_df,COMPASS09ex_df,'k+','phT')
# plt.subplot(4,3,10)
# plotSivAsymmBands(COMPASS09NN_df,COMPASS09ex_df,'k-','x')
# plt.subplot(4,3,11)
# plotSivAsymmBands(COMPASS09NN_df,COMPASS09ex_df,'k-','z')
# plt.subplot(4,3,12)
# plotSivAsymmBands(COMPASS09NN_df,COMPASS09ex_df,'k-','phT')
# #### Here the user needs to define the plot title based on the data set ####
# plt.savefig(str(Plots_Folder)+'/'+'COMPASS09.pdf', format='pdf', bbox_inches='tight')

# fig4=plt.figure(4,figsize=(15,30))
# plt.subplot(4,3,1)
# plotSivAsymmBands(COMPASS15NN_df,COMPASS15ex_df,'pi+','x')
# plt.subplot(4,3,2)
# plotSivAsymmBands(COMPASS15NN_df,COMPASS15ex_df,'pi+','z')
# plt.subplot(4,3,3)
# plotSivAsymmBands(COMPASS15NN_df,COMPASS15ex_df,'pi+','phT')
# plt.subplot(4,3,4)
# plotSivAsymmBands(COMPASS15NN_df,COMPASS15ex_df,'pi-','x')
# plt.subplot(4,3,5)
# plotSivAsymmBands(COMPASS15NN_df,COMPASS15ex_df,'pi-','z')
# plt.subplot(4,3,6)
# plotSivAsymmBands(COMPASS15NN_df,COMPASS15ex_df,'pi-','phT')
# plt.subplot(4,3,7)
# plotSivAsymmBands(COMPASS15NN_df,COMPASS15ex_df,'k+','x')
# plt.subplot(4,3,8)
# plotSivAsymmBands(COMPASS15NN_df,COMPASS15ex_df,'k+','z')
# plt.subplot(4,3,9)
# plotSivAsymmBands(COMPASS15NN_df,COMPASS15ex_df,'k+','phT')
# plt.subplot(4,3,10)
# plotSivAsymmBands(COMPASS15NN_df,COMPASS15ex_df,'k-','x')
# plt.subplot(4,3,11)
# plotSivAsymmBands(COMPASS15NN_df,COMPASS15ex_df,'k-','z')
# plt.subplot(4,3,12)
# plotSivAsymmBands(COMPASS15NN_df,COMPASS15ex_df,'k-','phT')
# #### Here the user needs to define the plot title based on the data set ####
# plt.savefig(str(Plots_Folder)+'/'+'COMPASS15.pdf', format='pdf', bbox_inches='tight')

############## COMPASS DY Projections ####################

def DYDependencePlotSign(RealDF,ProjDF,dep):
    tempRdf=RealDF[RealDF["Dependence"]==dep]
    tempPdf=ProjDF[ProjDF["Dependence"]==dep]
    tempx=np.array(tempRdf[dep])
    tempNNx=np.array(tempPdf[dep])
    tempy=np.array(tempRdf["Siv"])
    tempyerr=np.array(tempRdf["tot_err"])
    tempNNy=np.array(tempPdf["Siv"])
    tempNNyerr=np.array(tempPdf["tot_err"])
    plt.errorbar(tempx,tempy,tempyerr,fmt='o',color='blue')
    plt.errorbar(tempNNx,tempNNy,tempNNyerr,fmt='o',color='red')
    plt.title('Asymmetry vs '+str(dep))
    

def DYAsymPlots(RealDF,ProjDF,figname):
    plt.subplot(1,5,1)
    DYDependencePlotSign(RealDF,ProjDF,'x1')
    plt.subplot(1,5,2)
    DYDependencePlotSign(RealDF,ProjDF,'x2')
    plt.subplot(1,5,3)
    DYDependencePlotSign(RealDF,ProjDF,'xF')
    plt.subplot(1,5,4)
    DYDependencePlotSign(RealDF,ProjDF,'QT')
    plt.subplot(1,5,5)
    DYDependencePlotSign(RealDF,ProjDF,'QM')
    plt.savefig(str(Plots_Folder)+'/'+str(figname)+'.pdf',format='pdf',bbox_inches='tight')


# fig5=plt.figure(5,figsize=(15,5))
# DYAsymPlots(COMPASS17ex_df,COMPASS17NN_minus_df,'Projected_DY_minus')
# fig6=plt.figure(6,figsize=(15,5))
# DYAsymPlots(COMPASS17ex_df,COMPASS17NN_plus_df,'Projected_DY_plus')

# def chi2plot(arr):
#     fig2=plt.figure(2)
#     plt.hist(arr, bins='auto', color='#0504aa',
#                             alpha=0.7, rwidth=0.85)
#     plt.grid(axis='y', alpha=0.75)
#     plt.xlabel('Chi2')
#     plt.ylabel('Frequency')
#     plt.savefig('Chi2_DY.pdf',format='pdf',bbox_inches='tight')
    
    
# chi2plot(Projected_Siv_Chi2)


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
#test_pars=([7.0, 0.89, 2.50, 20, -0.12, 0.4,  20, -2.4, 2.7, 17, -0.7, 1.5, 20, -20, 4.7, 2.3, 20, 9.5, 20])
# test_pars=([7.0, 0.89, 2.75, 20, -0.12, 0.4,  20, -2.2, 3.0, 15.5, -0.7, 1.5, 15, -20, 4.7, 2.3, 20, 9.5, 20])
# test_errs=([  4, 0.06, 0.11,  2,  0.06, 0.35, 16,  0.4, 0.6,  4,  0.5, 0.6, 17,  40, 3.0, 2.2,  5, 1.4, 14])
# test_pars=([7.0, 0.773, 1.77, 12.71, -0.635, 1.22, 1.1, -1.54, 0.24, 0.19, -0.19, 1.9, 19.2, -0.6, 2.28, 2.39, 14, 9.7, 12])
# test_errs=([  0.005, 0.006, 0.04, 0.29, 0.023, 0.13, 0.4, 0.11, 0.04, 0.15, 0.11, 0.6, 2.6, 0.4, 0.30, 0.31, 6, 2, 5])
# test_pars=  ([ 3.69, 1.32, 3.37, 16.86, 16.91, 1.83, 0.38, -1.74, 2.03, 11.48, -31.62, 5.27, 18.52, -13.25, 4.82, 7.24, -19.97, 6.04, 19.99])
# test_errs=  ([  0.005, 0.006, 0.04, 0.29, 0.023, 0.13, 0.4, 0.11, 0.04, 0.15, 0.11, 0.6, 2.6, 0.4, 0.30, 0.31, 6, 2, 5])
# test_pars=  ([ 3.15, 0.93, 3.02, 14.32, 3.44, 1.51, 0.76, -1.4, 1.8, 9.98, -19.99, 5.78, 20, -4.15, 8.02, 19.99, -15.1, 6.92, 19.79 ])
# test_errs=  ([  0.005, 0.006, 0.04, 0.29, 0.023, 0.13, 0.4, 0.11, 0.04, 0.15, 0.11, 0.6, 2.6, 0.4, 0.30, 0.31, 6, 2, 5])

test_pars=  ([ 3.15, 1.3, 3.37, 14.32, 6.44, 1.51, 0.76, -1.7, 1.3, 13.98, -19, 5.84, 20, -4.15, 8.02, 19.99, -15.1, 6.92, 19 ])
test_errs=  ([  0.005, 0.006, 0.04, 0.29, 0.023, 0.13, 0.4, 0.11, 0.04, 0.15, 0.11, 0.6, 2.6, 0.4, 0.30, 0.31, 6, 2, 5])




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
        tempsiv=2*NNqbar(x,Nq,aq,bq)*hfunc(kp,tempM1)*(np.exp((-kp**2)/(Kp2A)))*(1/((np.pi)*(Kp2A)))*(xFxQ2(dataset,flavor,x,QQ))
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
    

# def plotSiversQ(flavor,ParmResults,col):
#     tempkT=np.linspace(0, 1.5)
#     tempSiv=[SiversFuncQ(PDFdataset,flavor,0.1,2.4,tempkT[i],ParmResults) for i in range(0,len(tempkT))]
#     plt.plot(tempkT,tempSiv, '--', color = col)
#     #return tempSiv

# def plotSiversAntiQ(flavor,ParmResults, col):
#     tempkT=np.linspace(0, 1.5)
#     tempSiv=[SiversFuncAntiQ(PDFdataset,flavor,0.1,2.4,tempkT[i],ParmResults) for i in range(0,len(tempkT))]
#     plt.plot(tempkT,tempSiv, '--',color = col)

def plotSiversQ(flavor,ParmResults,col, lbl):
    tempkT=np.linspace(0, 1.5)
    tempSiv=[SiversFuncQ(PDFdataset,flavor,0.1,2.4,tempkT[i],ParmResults) for i in range(0,len(tempkT))]
    plt.plot(tempkT,tempSiv, '--', color = col, label=lbl)
    #return tempSiv

def plotSiversAntiQ(flavor,ParmResults, col, lbl):
    tempkT=np.linspace(0, 1.5)
    tempSiv=[SiversFuncAntiQ(PDFdataset,flavor,0.1,2.4,tempkT[i],ParmResults) for i in range(0,len(tempkT))]
    plt.plot(tempkT,tempSiv, '--',color = col, label =lbl)
    


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
    plotSiversQ(2,test_pars, 'b')
    plt.plot(tempKT, tempfd, 'r', label='$d$')
    plt.fill_between(tempKT, tempfd-tempfdErr, tempfd+tempfdErr, facecolor='r', alpha=0.3)
    plotSiversQ(1,test_pars, 'r')
    plt.plot(tempKT, tempfs, 'g', label='$s$')
    plt.fill_between(tempKT, tempfs-tempfsErr, tempfs+tempfsErr, facecolor='g', alpha=0.3)
    plotSiversQ(3,test_pars, 'g')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(-0.125,0.125)
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
    plotSiversAntiQ(-2,test_pars, 'b')
    plt.plot(tempKT, tempfd, 'r', label='$\\bar{d}$')
    plt.fill_between(tempKT, tempfd-tempfdErr, tempfd+tempfdErr, facecolor='r', alpha=0.3)
    plotSiversAntiQ(-1,test_pars, 'r')
    plt.plot(tempKT, tempfs, 'g', label='$\\bar{s}$')
    plt.fill_between(tempKT, tempfs-tempfsErr, tempfs+tempfsErr, facecolor='g', alpha=0.3)
    plotSiversAntiQ(-3,test_pars, 'g')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(-0.125,0.125)
    plt.legend(loc=4,fontsize=20,handlelength=3)
    plt.savefig(str(Plots_Folder)+'/'+'Sivers_AntiQ_SIDIS_NN.pdf', format='pdf', bbox_inches='tight')
    

def UPlots(tempdf):
    tempKT = np.array(tempdf['kperp'])
    tempfu = np.array(tempdf['fu'])
    tempfubar = np.array(tempdf['fubar'])
    tempfubarErr = np.array(tempdf['fubarErr'])
    plt.plot(tempKT, tempfu, 'b', label='$u$')
    plt.fill_between(tempKT, tempfu-tempfuErr, tempfu+tempfuErr, facecolor='b', alpha=0.3)
    plt.plot(tempKT, tempfubar, 'b', '--', label='$\\bar{u}$')
    plt.fill_between(tempKT, tempfubar-tempfubarErr, tempfubar + tempfubarErr, facecolor='b', alpha=0.15)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(-0.1,0.1)
    plt.legend(loc=4,fontsize=20,handlelength=3)
    #plt.savefig(str(Plots_Folder)+'/'+'SiversQ_SIDIS_NN.pdf', format='pdf', bbox_inches='tight')

def DPlots(tempdf):
    tempKT = np.array(tempdf['kperp'])
    tempfu = np.array(tempdf['fu'])
    tempfubar = np.array(tempdf['fubar'])
    tempfubarErr = np.array(tempdf['fubarErr'])
    plt.plot(tempKT, tempfu, 'b', label='$u$')
    plt.fill_between(tempKT, tempfu-tempfuErr, tempfu+tempfuErr, facecolor='b', alpha=0.3)
    plt.plot(tempKT, tempfubar, 'b', '--', label='$\\bar{u}$')
    plt.fill_between(tempKT, tempfubar-tempfubarErr, tempfubar + tempfubarErr, facecolor='b', alpha=0.15)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(-0.1,0.1)
    plt.legend(loc=4,fontsize=20,handlelength=3)
    

# fig7=plt.figure(7)    
# QSiversPlots(Sivers_CSV_df)
# fig8=plt.figure(8)
# AntiQSiversPlots(Sivers_CSV_df)


# def QSiversPlotsCombined(tempdf):
#     tempKT = np.array(tempdf['kperp'])
#     tempfu = np.array(tempdf['fu'])
#     tempfuErr = np.array(tempdf['fuErr'])
#     tempfd = np.array(tempdf['fd'])
#     tempfdErr = np.array(tempdf['fdErr'])
#     tempfs = np.array(tempdf['fs'])
#     tempfsErr = np.array(tempdf['fsErr'])
#     plt.plot(tempKT, tempfu, 'b', label='$u$')
#     plt.fill_between(tempKT, tempfu-tempfuErr, tempfu+tempfuErr, facecolor='b', alpha=0.3)
#     plotSiversQ(2,test_pars, 'b')
#     plt.plot(tempKT, tempfd, 'r', label='$d$')
#     plt.fill_between(tempKT, tempfd-tempfdErr, tempfd+tempfdErr, facecolor='r', alpha=0.3)
#     plotSiversQ(1,test_pars, 'r')
#     plt.plot(tempKT, tempfs, 'g', label='$s$')
#     plt.fill_between(tempKT, tempfs-tempfsErr, tempfs+tempfsErr, facecolor='g', alpha=0.3)
#     plotSiversQ(3,test_pars, 'g')
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.ylim(-0.125,0.125)
#     plt.legend(loc=1,fontsize=12,handlelength=3)
#     plt.savefig(str(Plots_Folder)+'/'+'SiversQ_SIDIS_NN.pdf', format='pdf', bbox_inches='tight')
    

# def AntiQSiversPlotsCombined(tempdf):
#     tempKT = np.array(tempdf['kperp'])
#     tempfu = np.array(tempdf['fubar'])
#     tempfuErr = np.array(tempdf['fubarErr'])
#     tempfd = np.array(tempdf['fdbar'])
#     tempfdErr = np.array(tempdf['fdbarErr'])
#     tempfs = np.array(tempdf['fsbar'])
#     tempfsErr = np.array(tempdf['fsbarErr'])
#     plt.plot(tempKT, tempfu, 'b', label='$\\bar{u}$')
#     plt.fill_between(tempKT, tempfu-tempfuErr, tempfu+tempfuErr, facecolor='b', alpha=0.3)
#     plotSiversAntiQ(-2,test_pars, 'b')
#     plt.plot(tempKT, tempfd, 'r', label='$\\bar{d}$')
#     plt.fill_between(tempKT, tempfd-tempfdErr, tempfd+tempfdErr, facecolor='r', alpha=0.3)
#     plotSiversAntiQ(-1,test_pars, 'r')
#     plt.plot(tempKT, tempfs, 'g', label='$\\bar{s}$')
#     plt.fill_between(tempKT, tempfs-tempfsErr, tempfs+tempfsErr, facecolor='g', alpha=0.3)
#     plotSiversAntiQ(-3,test_pars, 'g')
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.ylim(-0.125,0.125)
#     plt.legend(loc=1,fontsize=12,handlelength=3)
#     plt.savefig(str(Plots_Folder)+'/'+'Sivers_AntiQ_SIDIS_NN.pdf', format='pdf', bbox_inches='tight')

# fig9 = plt.figure(9,figsize=(10,8))
# plt.subplot(1,2,1)
# QSiversPlotsCombined(Sivers_CSV_df)
# plt.subplot(1,2,2)
# AntiQSiversPlotsCombined(Sivers_CSV_df)
# plt.savefig(str(Plots_Folder)+'/'+'Sivers_SIDIS.pdf', format='pdf', bbox_inches='tight')

fontsiz = 26


def QSiversPlots_Combined(tempdf):
    tempKT = np.array(tempdf['kperp'])
    tempfu = np.array(tempdf['fu'])
    tempfuErr = np.array(tempdf['fuErr'])
    tempfd = np.array(tempdf['fd'])
    tempfdErr = np.array(tempdf['fdErr'])
    tempfs = np.array(tempdf['fs'])
    tempfsErr = np.array(tempdf['fsErr'])
    plt.plot(tempKT, tempfu, 'b', label='$u_{NN}$')
    plt.fill_between(tempKT, tempfu-tempfuErr, tempfu+tempfuErr, facecolor='b', alpha=0.3)
    #plotSiversQ(2,test_pars, 'b', '$u_{MINUIT}$')
    plt.plot(tempKT, tempfd, 'r', label='$d_{NN}$')
    plt.fill_between(tempKT, tempfd-tempfdErr, tempfd+tempfdErr, facecolor='r', alpha=0.3)
    #plotSiversQ(1,test_pars, 'r', '$d_{MINUIT}$')
    plt.plot(tempKT, tempfs, 'g', label='$s_{NN}$')
    plt.fill_between(tempKT, tempfs-tempfsErr, tempfs+tempfsErr, facecolor='g', alpha=0.3)
    #plotSiversQ(3,test_pars, 'g', '$s_{MINUIT}$')
    plt.xticks(fontsize=fontsiz)
    plt.yticks(fontsize=fontsiz)
    plt.text(0.01, 0.12,'$Q^2=2.4$ GeV$^2$', fontsize=fontsiz, fontname = 'Times New Roman')
    plt.text(0.01, 0.10,'$x=0.1$', fontsize=fontsiz, fontname = 'Times New Roman')
    plt.xlabel('$k_{\perp}$ (GeV)', fontsize=fontsiz, fontname = 'Times New Roman')
    plt.ylabel('$x \Delta f^N (x,k_{\perp})$', fontsize=fontsiz+2, fontname = 'Times New Roman')
    plt.ylim(-0.15,0.15)
    plt.legend(loc=1,fontsize=fontsiz,handlelength=3)
    

def AntiQSiversPlots_Combined(tempdf):
    tempKT = np.array(tempdf['kperp'])
    tempfu = np.array(tempdf['fubar'])
    tempfuErr = np.array(tempdf['fubarErr'])
    tempfd = np.array(tempdf['fdbar'])
    tempfdErr = np.array(tempdf['fdbarErr'])
    tempfs = np.array(tempdf['fsbar'])
    tempfsErr = np.array(tempdf['fsbarErr'])
    plt.plot(tempKT, tempfu, 'b', label='$\\bar{u}_{NN}$')
    plt.fill_between(tempKT, tempfu-tempfuErr, tempfu+tempfuErr, facecolor='b', alpha=0.3)
    #plotSiversAntiQ(-2,test_pars, 'b', '$\\bar{u}_{MINUIT}$')
    plt.plot(tempKT, tempfd, 'r', label='$\\bar{d}_{NN}$')
    plt.fill_between(tempKT, tempfd-tempfdErr, tempfd+tempfdErr, facecolor='r', alpha=0.3)
    #plotSiversAntiQ(-1,test_pars, 'r', '$\\bar{d}_{MINUIT}$')
    plt.plot(tempKT, tempfs, 'g', label='$\\bar{s}_{NN}$')
    plt.fill_between(tempKT, tempfs-tempfsErr, tempfs+tempfsErr, facecolor='g', alpha=0.3)
    #plotSiversAntiQ(-3,test_pars, 'g', '$\\bar{s}_{MINUIT}$')
    plt.xticks(fontsize=fontsiz)
    plt.yticks(fontsize=fontsiz)
    plt.text(0.01, 0.12,'$Q^2=2.4$ GeV$^2$', fontsize=fontsiz, fontname = 'Times New Roman')
    plt.text(0.01, 0.10,'$x=0.1$', fontsize=fontsiz, fontname = 'Times New Roman')
    plt.xlabel('$k_{\perp}$  (GeV)', fontsize=fontsiz, fontname = 'Times New Roman')
    #plt.ylabel('$x \Delta f^N (x,k_{\perp})$', fontsize=fontsiz)
    plt.ylim(-0.15,0.15)
    plt.legend(loc=1,fontsize=fontsiz,handlelength=3)
    
fig9=plt.figure(9,figsize=(20,8))
plt.subplot(1,2,1)    
QSiversPlots_Combined(Sivers_CSV_df)
plt.subplot(1,2,2)
AntiQSiversPlots_Combined(Sivers_CSV_df)
plt.savefig(str(Plots_Folder)+'/'+'Sivers_SIDIS_NN.pdf', format='pdf', bbox_inches='tight')