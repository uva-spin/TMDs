import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lhapdf
import os

Plots_Folder = './NN_DY_Proj_Plots'
######################################################################
######################    Results files ##############################
######################################################################
CSVs_Folder = './NN_SIDIS_Fit_Results'
folders_array=os.listdir(CSVs_Folder)
COMPASS17NN_minus = pd.read_csv(str(CSVs_Folder)+'/'+'Result_COMPASS_DY_from_SIDIS_minus.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
COMPASS17NN_minus_df = pd.concat([COMPASS17NN_minus])
COMPASS17NN_plus = pd.read_csv(str(CSVs_Folder)+'/'+'Result_COMPASS_DY_from_SIDIS_plus.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
COMPASS17NN_plus_df = pd.concat([COMPASS17NN_plus])

SQ_NN_minus = pd.read_csv(str(CSVs_Folder)+'/'+'Result_SpinQuest_DY_from_SIDIS_minus.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
SQ_NN_minus_df = pd.concat([SQ_NN_minus])
SQ_NN_plus = pd.read_csv(str(CSVs_Folder)+'/'+'Result_SpinQuest_DY_from_SIDIS_plus.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
SQ_NN_plus_df = pd.concat([SQ_NN_plus])

######################################################################
#########################    Data files ##############################
######################################################################

Data_Folder = '../Data'
COMPASS17ex = pd.read_csv(str(Data_Folder)+'/'+'COMPASS_p_DY_2017.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
COMPASS17ex_df = pd.concat([COMPASS17ex])
SpinQuest = pd.read_csv(str(Data_Folder)+'/'+'SpinQuest.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
SpinQuest_df = pd.concat([SpinQuest])


##############################################################################

def DY_Points(tempdf):
    tempSivErr = np.array(tempdf['tot_err'])
    tempSiv = np.array(tempdf['Siv'])
    return np.array(tempSiv), np.array(tempSivErr)

def chisquare(y, yhat, err):
    return np.sum(((y - yhat)/err)**2)
    
    
# def DYDependencePlotSign(RealDF,ProjDF,dep):
#     tempRdf=RealDF[RealDF["Dependence"]==dep]
#     tempPdf=ProjDF[ProjDF["Dependence"]==dep]
#     tempx=np.array(tempRdf[dep])
#     tempNNx=np.array(tempPdf[dep])
#     tempy=np.array(tempRdf["Siv"])
#     tempyerr=np.array(tempRdf["tot_err"])
#     tempNNy=np.array(tempPdf["Siv"])
#     tempNNyerr=np.array(tempPdf["tot_err"])
#     plt.errorbar(tempx,tempy,tempyerr,fmt='o',color='blue')
#     plt.errorbar(tempNNx,tempNNy,tempNNyerr,fmt='o',color='red')
#     plt.title('Asymmetry vs '+str(dep))
    

def COMPASS_DYDependencePlotSign(RealDF,ProjDFminus,ProjDFplus,dep):
    tempRdf=RealDF[RealDF["Dependence"]==dep]
    tempPdfm=ProjDFminus[ProjDFminus["Dependence"]==dep]
    tempPdfp=ProjDFplus[ProjDFplus["Dependence"]==dep]
    tempx=np.array(tempRdf[dep])
    tempNNx=np.array(tempPdfm[dep])
    tempy=np.array(tempRdf["Siv"])
    tempyerr=np.array(tempRdf["tot_err"])
    tempNNym=np.array(tempPdfm["Siv"])
    tempNNymerr=np.array(tempPdfm["tot_err"])
    tempNNyp=np.array(tempPdfp["Siv"])
    tempNNyperr=np.array(tempPdfp["tot_err"])
    plt.errorbar(tempx,tempy,tempyerr,fmt='.',color='blue', label='COMPASS data')
    plt.errorbar(tempNNx,tempNNym,tempNNymerr,fmt='.',color='red', label='with sign-change')
    plt.errorbar(tempNNx,tempNNyp,tempNNyperr,fmt='+',color='red', label='without sign-change')
    plt.legend(loc='lower right')
    plt.ylim(-0.3,0.4)
    if str(dep)=='x1':
        plt.title('$x_1$ vs $A_N$')
    elif str(dep)=='x2':
        plt.title('$x_2$ vs $A_N$')
    elif str(dep)=='xF':
        plt.title('$x_F$ vs $A_N$')
    elif str(dep)=='QT':
        plt.title('$Q_T$ vs $A_N$')
    elif str(dep)=='QM':
        plt.title('$Q_M$ vs $A_N$')
    #plt.title(str(dep) + ' vs $A_N$')
    #plt.title('$A_N$ vs'+str(dep))

# def COMPASS_DYAsymPlots(RealDF,ProjDF):
#     fig1=plt.figure(1,figsize=(15,3))
#     plt.subplot(1,5,1)
#     DYDependencePlotSign(RealDF,ProjDF,'x1')
#     plt.subplot(1,5,2)
#     DYDependencePlotSign(RealDF,ProjDF,'x2')
#     plt.subplot(1,5,3)
#     DYDependencePlotSign(RealDF,ProjDF,'xF')
#     plt.subplot(1,5,4)
#     DYDependencePlotSign(RealDF,ProjDF,'QT')
#     plt.subplot(1,5,5)
#     DYDependencePlotSign(RealDF,ProjDF,'QM')
#     #plt.savefig(str(Plots_Folder)+'/'+str(figname)+'.pdf',format='pdf',bbox_inches='tight')

    
# def COMPASS_DYAsymPlots(RealDF,ProjDFminus,ProjDFplus):
#     #fig1=plt.figure(1,figsize=(15,3))
#     plt.rcParams["font.family"] = ["Times New Roman"]
#     plt.subplot(1,5,1)
#     plt.grid()
#     COMPASS_DYDependencePlotSign(RealDF,ProjDFminus,ProjDFplus,'x1')
#     plt.subplot(1,5,2)
#     plt.grid()
#     COMPASS_DYDependencePlotSign(RealDF,ProjDFminus,ProjDFplus,'x2')
#     plt.subplot(1,5,3)
#     plt.grid()
#     COMPASS_DYDependencePlotSign(RealDF,ProjDFminus,ProjDFplus,'xF')
#     plt.subplot(1,5,4)
#     plt.grid()
#     COMPASS_DYDependencePlotSign(RealDF,ProjDFminus,ProjDFplus,'QT')
#     plt.subplot(1,5,5)
#     plt.grid()
#     COMPASS_DYDependencePlotSign(RealDF,ProjDFminus,ProjDFplus,'QM')
    #plt.savefig(str(Plots_Folder)+'/'+str(figname)+'.pdf',format='pdf',bbox_inches='tight')
    
def COMPASS_DYAsymPlots(RealDF,ProjDFminus,ProjDFplus):
    #fig1=plt.figure(1,figsize=(15,3))
    plt.rcParams["font.family"] = ["Times New Roman"]
    plt.subplot(3,2,1)
    plt.grid()
    COMPASS_DYDependencePlotSign(RealDF,ProjDFminus,ProjDFplus,'x1')
    plt.subplot(3,2,2)
    plt.grid()
    COMPASS_DYDependencePlotSign(RealDF,ProjDFminus,ProjDFplus,'x2')
    plt.subplot(3,2,3)
    plt.grid()
    COMPASS_DYDependencePlotSign(RealDF,ProjDFminus,ProjDFplus,'xF')
    plt.subplot(3,2,4)
    plt.grid()
    COMPASS_DYDependencePlotSign(RealDF,ProjDFminus,ProjDFplus,'QT')
    plt.subplot(3,2,5)
    plt.grid()
    COMPASS_DYDependencePlotSign(RealDF,ProjDFminus,ProjDFplus,'QM')
    

def SQ_DYDependencePlotSign(ProjDFminus,ProjDFplus,dep):
    plt.rcParams["font.family"] = ["Times New Roman"]
    tempPdfm=ProjDFminus[ProjDFminus["Dependence"]==dep]
    tempPdfp=ProjDFplus[ProjDFplus["Dependence"]==dep]
    tempNNx=np.array(tempPdfm[dep])
    tempNNym=np.array(tempPdfm["Siv"])
    tempNNymerr=np.array(tempPdfm["tot_err"])
    tempNNyp=np.array(tempPdfp["Siv"])
    tempNNyperr=np.array(tempPdfp["tot_err"])
    # plt.plot(tempNNx,tempNNym,'-',color='red', label='with sign-change')
    plt.plot(tempNNx,tempNNym,'-',color='red')
    plt.fill_between(tempNNx,tempNNym-tempNNymerr, tempNNym+tempNNymerr, facecolor='r', alpha=0.3)
    #plt.plot(tempNNx,tempNNyp,'-',color='blue', label='without sign-change')
    #plt.fill_between(tempNNx,tempNNyp-tempNNyperr, tempNNyp+tempNNyperr, facecolor='b', alpha=0.3)    
    #plt.errorbar(tempNNx,tempNNyp,tempNNyperr,fmt='+',color='blue',alpha = 0.5, label='without sign-change')
    plt.legend(loc='upper left')
    plt.ylim(-0.1,0.1)
    if str(dep)=='x1':
        plt.title('$x_1$ vs $A_N$')
    elif str(dep)=='x2':
        plt.title('$x_2$ vs $A_N$')
    elif str(dep)=='xF':
        plt.title('$x_F$ vs $A_N$')
    elif str(dep)=='QT':
        plt.title('$Q_T$ vs $A_N$')
    elif str(dep)=='QM':
        plt.title('$Q_M$ vs $A_N$')
 
# def SQ_DYDependencePlotSign(ProjDFminus,ProjDFplus,dep):
#     tempPdfm=ProjDFminus[ProjDFminus["Dependence"]==dep]
#     tempPdfp=ProjDFplus[ProjDFplus["Dependence"]==dep]
#     tempNNx=np.array(tempPdfm[dep])
#     tempNNym=np.array(tempPdfm["Siv"])
#     tempNNymerr=np.array(tempPdfm["tot_err"])
#     tempNNyp=np.array(tempPdfp["Siv"])
#     tempNNyperr=np.array(tempPdfp["tot_err"])
#     plt.errorbar(tempNNx,tempNNym,tempNNymerr,fmt='.',color='red', label='with sign-change')
#     plt.errorbar(tempNNx,tempNNyp,tempNNyperr,fmt='+',color='blue',alpha = 0.5, label='without sign-change')
#     plt.legend(loc='center right')
#     if str(dep)=='x1':
#         plt.title('$x_1$ vs $A_N$')
#     elif str(dep)=='x2':
#         plt.title('$x_2$ vs $A_N$')
#     elif str(dep)=='xF':
#         plt.title('$x_F$ vs $A_N$')
#     elif str(dep)=='QT':
#         plt.title('$Q_T$ vs $A_N$')
#     elif str(dep)=='QM':
#         plt.title('$Q_M$ vs $A_N$')

# def SQ_DYAsymPlots(RealDF,ProjDF):
#     #fig1=plt.figure(1,figsize=(15,3))
#     plt.rcParams["font.family"] = ["Times New Roman"]
#     plt.subplot(1,4,1)
#     plt.grid()
#     SQ_DYDependencePlotSign(RealDF,ProjDF,'x1')
#     plt.subplot(1,4,2)
#     plt.grid()
#     SQ_DYDependencePlotSign(RealDF,ProjDF,'x2')
#     plt.subplot(1,4,3)
#     plt.grid()
#     SQ_DYDependencePlotSign(RealDF,ProjDF,'xF')
#     plt.subplot(1,4,4)
#     plt.grid()
#     SQ_DYDependencePlotSign(RealDF,ProjDF,'QT')
    #plt.savefig(str(Plots_Folder)+'/'+str(figname)+'.pdf',format='pdf',bbox_inches='tight')
    
def SQ_DYAsymPlots(RealDF,ProjDF):
    #fig1=plt.figure(1,figsize=(15,3))
    plt.rcParams["font.family"] = ["Times New Roman"]
    plt.subplot(2,2,1)
    plt.grid()
    SQ_DYDependencePlotSign(RealDF,ProjDF,'x1')
    plt.subplot(2,2,2)
    plt.grid()
    SQ_DYDependencePlotSign(RealDF,ProjDF,'x2')
    plt.subplot(2,2,3)
    plt.grid()
    SQ_DYDependencePlotSign(RealDF,ProjDF,'xF')
    plt.subplot(2,2,4)
    plt.grid('--')
    SQ_DYDependencePlotSign(RealDF,ProjDF,'QT')


# fig1=plt.figure(1,figsize=(20,3))  
fig1=plt.figure(1,figsize=(10,10))  
COMPASS_DYAsymPlots(COMPASS17ex_df,COMPASS17NN_minus_df, COMPASS17NN_plus_df)
plt.savefig(str(Plots_Folder)+'/'+'Projected_COMPASS_DY.pdf',format='pdf',bbox_inches='tight')

fig2=plt.figure(2,figsize=(10,10))  
SQ_DYAsymPlots(SQ_NN_minus_df, SQ_NN_plus_df)
plt.savefig(str(Plots_Folder)+'/'+'Projected_SpinQuest_DY.pdf',format='pdf',bbox_inches='tight')