import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lhapdf
import os

Plots_Folder = './NN_SIDIS_Plots'
######################################################################
######################    Results files ##############################
######################################################################
CSVs_Folder = './NN_SIDIS_Fit_Results'
folders_array=os.listdir(CSVs_Folder)
HERMES09NN = pd.read_csv(str(CSVs_Folder)+'/'+'Result_SIDIS_HERMES2009.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
HERMES09NN_df = pd.concat([HERMES09NN])
HERMES20NN = pd.read_csv(str(CSVs_Folder)+'/'+'Result_SIDIS_HERMES2020.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
HERMES20NN_df = pd.concat([HERMES20NN])
COMPASS09NN = pd.read_csv(str(CSVs_Folder)+'/'+'Result_SIDIS_COMPASS2009.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
COMPASS09NN_df = pd.concat([COMPASS09NN])
COMPASS15NN = pd.read_csv(str(CSVs_Folder)+'/'+'Result_SIDIS_COMPASS2015.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
COMPASS15NN_df = pd.concat([COMPASS15NN])

######################################################################
#########################    Data files ##############################
######################################################################

Data_Folder = './Data'
HERMES09ex = pd.read_csv(str(Data_Folder)+'/'+'HERMES_p_2009.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
HERMES09ex_df = pd.concat([HERMES09ex])
HERMES20ex = pd.read_csv(str(Data_Folder)+'/'+'HERMES_p_2020.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
HERMES20ex_df = pd.concat([HERMES20ex])
COMPASS09ex = pd.read_csv(str(Data_Folder)+'/'+'COMPASS_d_2009.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
COMPASS09ex_df = pd.concat([COMPASS09ex])
COMPASS15ex = pd.read_csv(str(Data_Folder)+'/'+'COMPASS_p_2015.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
COMPASS15ex_df = pd.concat([COMPASS15ex])


##############################################################################

class DataANN(object):
    def __init__(self, pdfset='cteq61',
                 ff_PIp='NNFF10_PIp_nlo', ff_PIm='NNFF10_PIm_nlo', ff_PIsum='NNFF10_PIsum_nlo',
                 ff_KAp='NNFF10_KAp_nlo', ff_KAm='NNFF10_KAm_nlo'):
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
    
    
def plotSivAsymmBands(resultdf, datasetdf, hadron, dependence):
    D_Xplt, D_DEP, D_yplt, D_errplt = datann.makeData(datasetdf, [hadron], [dependence])
    R_Xplt, R_DEP, R_yplt, R_errplt = datann.makeData(resultdf, [hadron], [dependence])
    chi2val=chisquare(D_yplt, R_yplt, D_errplt)
    plt.errorbar(D_DEP, D_yplt,yerr=D_errplt, fmt='bo',label='Data')
    plt.errorbar(R_DEP, R_yplt,yerr=R_errplt, fmt='ro',label='NN')
    #### Here the user needs to define the plot title based on the data set ####
    plt.title('SIDIS Sivers '+str(hadron)+' $\chi2$='+str('%.2f'%chi2val),fontsize=15)
    #plt.ylim([-0.001,0.001])
    plt.xlabel(str(dependence),fontsize=15)
    plt.legend(loc=2,fontsize=15,handlelength=3)


fig1=plt.figure(1,figsize=(15,30))
plt.subplot(5,3,1)
plotSivAsymmBands(HERMES09NN_df,HERMES09ex_df,'pi+','x')
plt.subplot(5,3,2)
plotSivAsymmBands(HERMES09NN_df,HERMES09ex_df,'pi+','z')
plt.subplot(5,3,3)
plotSivAsymmBands(HERMES09NN_df,HERMES09ex_df,'pi+','phT')
plt.subplot(5,3,4)
plotSivAsymmBands(HERMES09NN_df,HERMES09ex_df,'pi-','x')
plt.subplot(5,3,5)
plotSivAsymmBands(HERMES09NN_df,HERMES09ex_df,'pi-','z')
plt.subplot(5,3,6)
plotSivAsymmBands(HERMES09NN_df,HERMES09ex_df,'pi-','phT')
plt.subplot(5,3,7)
plotSivAsymmBands(HERMES09NN_df,HERMES09ex_df,'pi0','x')
plt.subplot(5,3,8)
plotSivAsymmBands(HERMES09NN_df,HERMES09ex_df,'pi0','z')
plt.subplot(5,3,9)
plotSivAsymmBands(HERMES09NN_df,HERMES09ex_df,'pi0','phT')
plt.subplot(5,3,10)
plotSivAsymmBands(HERMES09NN_df,HERMES09ex_df,'k+','x')
plt.subplot(5,3,11)
plotSivAsymmBands(HERMES09NN_df,HERMES09ex_df,'k+','z')
plt.subplot(5,3,12)
plotSivAsymmBands(HERMES09NN_df,HERMES09ex_df,'k+','phT')
plt.subplot(5,3,13)
plotSivAsymmBands(HERMES09NN_df,HERMES09ex_df,'k-','x')
plt.subplot(5,3,14)
plotSivAsymmBands(HERMES09NN_df,HERMES09ex_df,'k-','z')
plt.subplot(5,3,15)
plotSivAsymmBands(HERMES09NN_df,HERMES09ex_df,'k-','phT')
#### Here the user needs to define the plot title based on the data set ####
#plt.savefig('HERMES09_'+str(LOSSFN)+'_'+str(EPOCHS)+'Ep_'+str(HL)+'H_'+str(NODES)+'N_'+str(LR)+'LR.pdf', format='pdf', bbox_inches='tight')
plt.savefig(str(Plots_Folder)+'/'+'HERMES09.pdf', format='pdf', bbox_inches='tight')

fig2=plt.figure(2,figsize=(15,30))
plt.subplot(5,3,1)
plotSivAsymmBands(HERMES20NN_df,HERMES20ex_df,'pi+','x')
plt.subplot(5,3,2)
plotSivAsymmBands(HERMES20NN_df,HERMES20ex_df,'pi+','z')
plt.subplot(5,3,3)
plotSivAsymmBands(HERMES20NN_df,HERMES20ex_df,'pi+','phT')
plt.subplot(5,3,4)
plotSivAsymmBands(HERMES20NN_df,HERMES20ex_df,'pi-','x')
plt.subplot(5,3,5)
plotSivAsymmBands(HERMES20NN_df,HERMES20ex_df,'pi-','z')
plt.subplot(5,3,6)
plotSivAsymmBands(HERMES20NN_df,HERMES20ex_df,'pi-','phT')
plt.subplot(5,3,7)
plotSivAsymmBands(HERMES20NN_df,HERMES20ex_df,'pi0','x')
plt.subplot(5,3,8)
plotSivAsymmBands(HERMES20NN_df,HERMES20ex_df,'pi0','z')
plt.subplot(5,3,9)
plotSivAsymmBands(HERMES20NN_df,HERMES20ex_df,'pi0','phT')
plt.subplot(5,3,10)
plotSivAsymmBands(HERMES20NN_df,HERMES20ex_df,'k+','x')
plt.subplot(5,3,11)
plotSivAsymmBands(HERMES20NN_df,HERMES20ex_df,'k+','z')
plt.subplot(5,3,12)
plotSivAsymmBands(HERMES20NN_df,HERMES20ex_df,'k+','phT')
plt.subplot(5,3,13)
plotSivAsymmBands(HERMES20NN_df,HERMES20ex_df,'k-','x')
plt.subplot(5,3,14)
plotSivAsymmBands(HERMES20NN_df,HERMES20ex_df,'k-','z')
plt.subplot(5,3,15)
plotSivAsymmBands(HERMES20NN_df,HERMES20ex_df,'k-','phT')
plt.savefig(str(Plots_Folder)+'/'+'HERMES20.pdf', format='pdf', bbox_inches='tight')


fig3=plt.figure(3,figsize=(15,30))
plt.subplot(4,3,1)
plotSivAsymmBands(COMPASS09NN_df,COMPASS09ex_df,'pi+','x')
plt.subplot(4,3,2)
plotSivAsymmBands(COMPASS09NN_df,COMPASS09ex_df,'pi+','z')
plt.subplot(4,3,3)
plotSivAsymmBands(COMPASS09NN_df,COMPASS09ex_df,'pi+','phT')
plt.subplot(4,3,4)
plotSivAsymmBands(COMPASS09NN_df,COMPASS09ex_df,'pi-','x')
plt.subplot(4,3,5)
plotSivAsymmBands(COMPASS09NN_df,COMPASS09ex_df,'pi-','z')
plt.subplot(4,3,6)
plotSivAsymmBands(COMPASS09NN_df,COMPASS09ex_df,'pi-','phT')
plt.subplot(4,3,7)
plotSivAsymmBands(COMPASS09NN_df,COMPASS09ex_df,'k+','x')
plt.subplot(4,3,8)
plotSivAsymmBands(COMPASS09NN_df,COMPASS09ex_df,'k+','z')
plt.subplot(4,3,9)
plotSivAsymmBands(COMPASS09NN_df,COMPASS09ex_df,'k+','phT')
plt.subplot(4,3,10)
plotSivAsymmBands(COMPASS09NN_df,COMPASS09ex_df,'k-','x')
plt.subplot(4,3,11)
plotSivAsymmBands(COMPASS09NN_df,COMPASS09ex_df,'k-','z')
plt.subplot(4,3,12)
plotSivAsymmBands(COMPASS09NN_df,COMPASS09ex_df,'k-','phT')
#### Here the user needs to define the plot title based on the data set ####
plt.savefig(str(Plots_Folder)+'/'+'COMPASS09.pdf', format='pdf', bbox_inches='tight')

fig4=plt.figure(4,figsize=(15,30))
plt.subplot(4,3,1)
plotSivAsymmBands(COMPASS15NN_df,COMPASS15ex_df,'pi+','x')
plt.subplot(4,3,2)
plotSivAsymmBands(COMPASS15NN_df,COMPASS15ex_df,'pi+','z')
plt.subplot(4,3,3)
plotSivAsymmBands(COMPASS15NN_df,COMPASS15ex_df,'pi+','phT')
plt.subplot(4,3,4)
plotSivAsymmBands(COMPASS15NN_df,COMPASS15ex_df,'pi-','x')
plt.subplot(4,3,5)
plotSivAsymmBands(COMPASS15NN_df,COMPASS15ex_df,'pi-','z')
plt.subplot(4,3,6)
plotSivAsymmBands(COMPASS15NN_df,COMPASS15ex_df,'pi-','phT')
plt.subplot(4,3,7)
plotSivAsymmBands(COMPASS15NN_df,COMPASS15ex_df,'k+','x')
plt.subplot(4,3,8)
plotSivAsymmBands(COMPASS15NN_df,COMPASS15ex_df,'k+','z')
plt.subplot(4,3,9)
plotSivAsymmBands(COMPASS15NN_df,COMPASS15ex_df,'k+','phT')
plt.subplot(4,3,10)
plotSivAsymmBands(COMPASS15NN_df,COMPASS15ex_df,'k-','x')
plt.subplot(4,3,11)
plotSivAsymmBands(COMPASS15NN_df,COMPASS15ex_df,'k-','z')
plt.subplot(4,3,12)
plotSivAsymmBands(COMPASS15NN_df,COMPASS15ex_df,'k-','phT')
#### Here the user needs to define the plot title based on the data set ####
plt.savefig(str(Plots_Folder)+'/'+'COMPASS15.pdf', format='pdf', bbox_inches='tight')


# def Sivers_Asym_vals(datadf,resultdf):
#     tempSivData = np.array(datadf['Siv'])
#     tempSivErrData = np.array(datadf['tot_err'])
#     tempSivResult = np.array(resultdf['Siv'])
#     tempSivErrResult = np.array(resultdf['tot_err'])
#     return np.array(tempSiv), np.array(tempSivErr)

# def SIDIS_NNFit_Results(f_array):
#     tempSiv = []
#     tempSivErr = []
#     chi2 = []
#     for i in range(0,len(folders_array)):
#         tempdf=pd.read_csv(str(Replicas_Folder) + '/' + str(f_array[i]))
#         tempyerr = np.array(tempdf['tot_err'])
#         tempSivErr.append(tempyerr)
#         tempyhat = np.array(tempdf['Siv'])
#         tempSiv.append(tempyhat)
#         tempy = np.array(COMPASS_DY2017['Siv'])
#         chi2.append(chisquare(tempy, tempyhat, tempyerr))
#     TempSiv = np.array(tempSiv)
#     TempSivErr = np.array(tempSivErr)
#     ChiSqr = np.array(chi2)
#     return np.mean(TempSiv,axis=0),np.std(TempSiv,axis=0), ChiSqr

# Projected_Siv = Projected_DY_Pseudo_Data(folders_array)[0]
# Projected_Siv_Err = Projected_DY_Pseudo_Data(folders_array)[1]
# Projected_Siv_Chi2 = Projected_DY_Pseudo_Data(folders_array)[2]

# COMPASS_DY_Siv = DY_Real_Data(COMPASS_DY2017)[0]
# COMPASS_DY_Siv_Err = DY_Real_Data(COMPASS_DY2017)[1]
    
# def Projected_DY_Pseudo_DataFrame(tempdf):
#     #tempdf=pd.read_csv(datafile)
#     tempMod=np.array(tempdf['Siv_mod'],dtype=object)
#     tempDEP=np.array(tempdf['Dependence'],dtype=object)
#     tempX1=np.array(tempdf['x1'],dtype=object)
#     tempX2=np.array(tempdf['x2'],dtype=object)
#     tempXF=np.array(tempdf['xF'],dtype=object)
#     tempQT=np.array(tempdf['QT'],dtype=object)
#     tempQM=np.array(tempdf['QM'],dtype=object)
#     tempSivErr=np.array(Projected_Siv_Err)
#     tempSiv=np.array(Projected_Siv)
#     data_dictionary={"Siv_mod":[],"Dependence":[],"x1":[],"x2":[],"xF":[],"QT":[],"QM":[],"Siv":[],"tot_err":[]}
#     data_dictionary["Siv_mod"]=tempMod
#     data_dictionary["Dependence"]=tempDEP
#     data_dictionary["x1"]=tempX1
#     data_dictionary["x2"]=tempX2
#     data_dictionary["xF"]=tempXF
#     data_dictionary["QT"]=tempQT
#     data_dictionary["QM"]=tempQM
#     data_dictionary["tot_err"]=tempSivErr
#     data_dictionary["Siv"]=tempSiv
#     return pd.DataFrame(data_dictionary)
    
# COMPASS_DY_ProjectedDF = Projected_DY_Pseudo_DataFrame(COMPASS_DY2017)

# def DYDependencePlotSign(RealDF,ProjDF,dep):
#     tempRdf=RealDF[RealDF["Dependence"]==dep]
#     tempPdf=ProjDF[ProjDF["Dependence"]==dep]
#     tempx=np.array(tempRdf[dep])
#     tempNNx=np.array(tempPdf[dep]+0.01)
#     tempy=np.array(tempRdf["Siv"])
#     tempyerr=np.array(tempRdf["tot_err"])
#     tempNNy=np.array(tempPdf["Siv"])
#     tempNNyerr=np.array(tempPdf["tot_err"])
#     plt.errorbar(tempx,tempy,tempyerr,fmt='o',color='blue')
#     plt.errorbar(tempNNx,tempNNy,tempNNyerr,fmt='o',color='red')
#     plt.title('Asymmetry vs '+str(dep))
    

# def DYAsymPlots(RealDF,ProjDF,figname):
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
#     plt.savefig(figname+'.pdf',format='pdf',bbox_inches='tight')
    
# DYAsymPlots(COMPASS_DY2017,COMPASS_DY_ProjectedDF,'Projected_DY')

# def chi2plot(arr):
#     fig2=plt.figure(2)
#     plt.hist(arr, bins='auto', color='#0504aa',
#                             alpha=0.7, rwidth=0.85)
#     plt.grid(axis='y', alpha=0.75)
#     plt.xlabel('Chi2')
#     plt.ylabel('Frequency')
#     plt.savefig('Chi2_DY.pdf',format='pdf',bbox_inches='tight')
    
    
# chi2plot(Projected_Siv_Chi2)




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
    plt.plot(tempKT, tempfd, 'r', label='$\\bar{d}$')
    plt.fill_between(tempKT, tempfd-tempfdErr, tempfd+tempfdErr, facecolor='r', alpha=0.3)
    plt.plot(tempKT, tempfs, 'g', label='$\\bar{s}$')
    plt.fill_between(tempKT, tempfs-tempfsErr, tempfs+tempfsErr, facecolor='g', alpha=0.3)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc=4,fontsize=20,handlelength=3)
    plt.savefig(str(Plots_Folder)+'/'+'Sivers_AntiQ_SIDIS_NN.pdf', format='pdf', bbox_inches='tight')
    

fig5=plt.figure(5)    
QSiversPlots(Sivers_CSV_df)
fig6=plt.figure(6)
AntiQSiversPlots(Sivers_CSV_df)