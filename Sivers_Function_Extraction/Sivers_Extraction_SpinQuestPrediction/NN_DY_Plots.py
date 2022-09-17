import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lhapdf
import os

Plots_Folder = './NN_DY_Plots'
######################################################################
######################    Results files ##############################
######################################################################
CSVs_Folder = './NN_DY_Fit_Results'
folders_array=os.listdir(CSVs_Folder)
COMPASS17NN = pd.read_csv(str(CSVs_Folder)+'/'+'Result_DY_from_SIDIS_minus.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
COMPASS17NN_df = pd.concat([COMPASS17NN])


######################################################################
#########################    Data files ##############################
######################################################################

Data_Folder = './Data'
COMPASS17ex = pd.read_csv(str(Data_Folder)+'/'+'COMPASS_p_DY_2017.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
COMPASS17ex_df = pd.concat([COMPASS17ex])


##############################################################################

def DY_Points(tempdf):
    tempSivErr = np.array(tempdf['tot_err'])
    tempSiv = np.array(tempdf['Siv'])
    return np.array(tempSiv), np.array(tempSivErr)

def chisquare(y, yhat, err):
    return np.sum(((y - yhat)/err)**2)
    
    
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
    fig1=plt.figure(1,figsize=(15,3))
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


   
DYAsymPlots(COMPASS17ex_df,COMPASS17NN_df,'Projected_DY_minus')



####################################################################################

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
    plt.plot(tempKT, tempfd, 'r', label='$d$')
    plt.fill_between(tempKT, tempfd-tempfdErr, tempfd+tempfdErr, facecolor='r', alpha=0.3)
    plt.plot(tempKT, tempfs, 'g', label='$s$')
    plt.fill_between(tempKT, tempfs-tempfsErr, tempfs+tempfsErr, facecolor='g', alpha=0.3)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc=4,fontsize=20,handlelength=3)
    plt.savefig(str(Plots_Folder)+'/'+'SiversQ_DY_NN.pdf', format='pdf', bbox_inches='tight')
    

def AntiQSiversPlots(tempdf):
    tempKT = np.array(tempdf['kperp'])
    tempfu = np.array(tempdf['fubar'])
    tempfuErr = np.array(tempdf['fubarErr'])
    tempfd = np.array(tempdf['fdbar'])
    tempfdErr = 0.5*np.array(tempdf['fdbarErr'])
    tempfs = np.array(tempdf['fsbar'])
    tempfsErr = np.array(tempdf['fsbarErr'])
    plt.plot(tempKT, tempfu, 'b', label='$\\bar{u}$')
    plt.fill_between(tempKT, tempfu-tempfuErr, tempfu+tempfuErr, facecolor='b', alpha=0.3)
    plt.plot(tempKT, tempfd, 'r', label='$\\bar{d}$')
    plt.fill_between(tempKT, tempfd-tempfdErr, tempfd+tempfdErr, facecolor='r', alpha=0.3)
    plt.plot(tempKT, tempfs, 'g', label='$\\bar{s}$')
    plt.fill_between(tempKT, tempfs-tempfsErr, tempfs+tempfsErr, facecolor='g', alpha=0.3)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc=4,fontsize=20,handlelength=3)
    plt.savefig(str(Plots_Folder)+'/'+'Sivers_AntiQ_DY_NN.pdf', format='pdf', bbox_inches='tight')
    

fig2=plt.figure(2)    
QSiversPlots(Sivers_CSV_df)
fig3=plt.figure(3)
AntiQSiversPlots(Sivers_CSV_df)