#import tensorflow as tf
import pandas as pd
import numpy as np
#import lhapdf
import matplotlib.pyplot as plt
#import functions_develop
#import natsort
#import os


herm9_DATA = pd.read_csv('./Data/HERMES_p_2009.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
herm20_DATA = pd.read_csv('./Data/HERMES_p_2020.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
comp9_DATA = pd.read_csv('./Data/COMPASS_d_2009.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
comp15_DATA = pd.read_csv('./Data/COMPASS_p_2015.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')

herm9 = pd.read_csv('./PseudoData_Parm1/HERMES2009_Pseudo.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
herm20 = pd.read_csv('./PseudoData_Parm1/HERMES2020_Pseudo.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
comp9 = pd.read_csv('./PseudoData_Parm1/COMPASS2009_Pseudo.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
comp15 = pd.read_csv('./PseudoData_Parm1/COMPASS2015_Pseudo.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')


OutputFolder = 'PseudoData_Parm1'

######################################################################################
############### HERE the User need to define: plots for which data set (kinematics)###
######################################################################################

#dfsingle = pd.concat([herm20])
#DataSet = 'HERMES2020'

HERMES2009_DATA = pd.concat([herm9_DATA])
HERMES2020_DATA = pd.concat([herm20_DATA])
COMPASS2009_DATA = pd.concat([comp9_DATA])
COMPASS2015_DATA = pd.concat([comp15_DATA])

HERMES2009 = pd.concat([herm9])
HERMES2020 = pd.concat([herm20])
COMPASS2009 = pd.concat([comp9])
COMPASS2015 = pd.concat([comp15])


def calc_yhat(model, X):
    return model.predict(X)



class DataANN(object):
    def __init__(self):

        self.eu = 2/3
        self.eubar = -2/3
        self.ed = -1/3
        self.edbar = 1/3
        self.es = -1/3
        self.esbar = 1/3


    def makeData(self, df, hadrons, dependencies):

        data = {'x': [],
             'z': [],
             'phT': []}

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

            data['x'] += list(x)
            data['z'] += list(z)
            data['phT'] += list(sliced['phT'])

        for key in data.keys():
            data[key] = np.array(data[key])

        #print(np.array(y))
        return data, data[dependencies[0]], np.array(y), np.array(err)

datann = DataANN()

def chisquare(y, yhat, err):
    return np.sum(((y - yhat)/err)**2)
    
    
def plotSivAsymm(datasetdf, resultdf, hadron, dependence):
    D_Xplt, D_DEP, D_yplt, D_errplt = datann.makeData(datasetdf, [hadron], [dependence])
    R_Xplt, R_DEP, R_yplt, R_errplt = datann.makeData(resultdf, [hadron], [dependence])
    chi2val=chisquare(D_yplt, R_yplt, D_errplt)
    plt.errorbar(D_DEP, D_yplt,yerr=D_errplt, fmt='bo',label='Data')
    plt.errorbar(R_DEP, R_yplt,yerr=R_errplt, fmt='ro',label='PseudoData')
    #### Here the user needs to define the plot title based on the data set ####
    plt.title('        '+ str(hadron)+' $\chi2$='+str('%.2f'%chi2val),fontsize=15)
    #plt.ylim([-0.001,0.001])
    plt.xlabel(str(dependence),fontsize=15)
    plt.legend(loc=2,fontsize=10,handlelength=3)


# fig1=plt.figure(1,figsize=(15,10))
# plotSivAsymm(HERMES2009_DATA,HERMES2009,'pi+','x')
# plt.savefig('HERMES09_comparison.pdf', format='pdf', bbox_inches='tight')

fig1=plt.figure(1,figsize=(15,20))
plt.subplot(5,3,1)
plotSivAsymm(HERMES2009_DATA,HERMES2009,'pi+','x')
plt.subplot(5,3,2)
plotSivAsymm(HERMES2009_DATA,HERMES2009,'pi+','z')
plt.subplot(5,3,3)
plotSivAsymm(HERMES2009_DATA,HERMES2009,'pi+','phT')
plt.subplot(5,3,4)
plotSivAsymm(HERMES2009_DATA,HERMES2009,'pi-','x')
plt.subplot(5,3,5)
plotSivAsymm(HERMES2009_DATA,HERMES2009,'pi-','z')
plt.subplot(5,3,6)
plotSivAsymm(HERMES2009_DATA,HERMES2009,'pi-','phT')
plt.subplot(5,3,7)
plotSivAsymm(HERMES2009_DATA,HERMES2009,'pi0','x')
plt.subplot(5,3,8)
plotSivAsymm(HERMES2009_DATA,HERMES2009,'pi0','z')
plt.subplot(5,3,9)
plotSivAsymm(HERMES2009_DATA,HERMES2009,'pi0','phT')
plt.subplot(5,3,10)
plotSivAsymm(HERMES2009_DATA,HERMES2009,'k+','x')
plt.subplot(5,3,11)
plotSivAsymm(HERMES2009_DATA,HERMES2009,'k+','z')
plt.subplot(5,3,12)
plotSivAsymm(HERMES2009_DATA,HERMES2009,'k+','phT')
plt.subplot(5,3,13)
plotSivAsymm(HERMES2009_DATA,HERMES2009,'k-','x')
plt.subplot(5,3,14)
plotSivAsymm(HERMES2009_DATA,HERMES2009,'k-','z')
plt.subplot(5,3,15)
plotSivAsymm(HERMES2009_DATA,HERMES2009,'k-','phT')
#### Here the user needs to define the plot title based on the data set ####
plt.savefig(str(OutputFolder)+'/'+'HERMES09_comparison.pdf', format='pdf', bbox_inches='tight')


fig2=plt.figure(2,figsize=(15,20))
plt.subplot(5,3,1)
plotSivAsymm(HERMES2020_DATA,HERMES2020,'pi+','x')
plt.subplot(5,3,2)
plotSivAsymm(HERMES2020_DATA,HERMES2020,'pi+','z')
plt.subplot(5,3,3)
plotSivAsymm(HERMES2020_DATA,HERMES2020,'pi+','phT')
plt.subplot(5,3,4)
plotSivAsymm(HERMES2020_DATA,HERMES2020,'pi-','x')
plt.subplot(5,3,5)
plotSivAsymm(HERMES2020_DATA,HERMES2020,'pi-','z')
plt.subplot(5,3,6)
plotSivAsymm(HERMES2020_DATA,HERMES2020,'pi-','phT')
plt.subplot(5,3,7)
plotSivAsymm(HERMES2020_DATA,HERMES2020,'pi0','x')
plt.subplot(5,3,8)
plotSivAsymm(HERMES2020_DATA,HERMES2020,'pi0','z')
plt.subplot(5,3,9)
plotSivAsymm(HERMES2020_DATA,HERMES2020,'pi0','phT')
plt.subplot(5,3,10)
plotSivAsymm(HERMES2020_DATA,HERMES2020,'k+','x')
plt.subplot(5,3,11)
plotSivAsymm(HERMES2020_DATA,HERMES2020,'k+','z')
plt.subplot(5,3,12)
plotSivAsymm(HERMES2020_DATA,HERMES2020,'k+','phT')
plt.subplot(5,3,13)
plotSivAsymm(HERMES2020_DATA,HERMES2020,'k-','x')
plt.subplot(5,3,14)
plotSivAsymm(HERMES2020_DATA,HERMES2020,'k-','z')
plt.subplot(5,3,15)
plotSivAsymm(HERMES2020_DATA,HERMES2020,'k-','phT')
#### Here the user needs to define the plot title based on the data set ####
plt.savefig(str(OutputFolder)+'/'+'HERMES20_comparison.pdf', format='pdf', bbox_inches='tight')


fig3=plt.figure(3,figsize=(15,20))
plt.subplot(4,3,1)
plotSivAsymm(COMPASS2009_DATA,COMPASS2009,'pi+','x')
plt.subplot(4,3,2)
plotSivAsymm(COMPASS2009_DATA,COMPASS2009,'pi+','z')
plt.subplot(4,3,3)
plotSivAsymm(COMPASS2009_DATA,COMPASS2009,'pi+','phT')
plt.subplot(4,3,4)
plotSivAsymm(COMPASS2009_DATA,COMPASS2009,'pi-','x')
plt.subplot(4,3,5)
plotSivAsymm(COMPASS2009_DATA,COMPASS2009,'pi-','z')
plt.subplot(4,3,6)
plotSivAsymm(COMPASS2009_DATA,COMPASS2009,'pi-','phT')
plt.subplot(4,3,7)
plotSivAsymm(COMPASS2009_DATA,COMPASS2009,'k+','x')
plt.subplot(4,3,8)
plotSivAsymm(COMPASS2009_DATA,COMPASS2009,'k+','z')
plt.subplot(4,3,9)
plotSivAsymm(COMPASS2009_DATA,COMPASS2009,'k+','phT')
plt.subplot(4,3,10)
plotSivAsymm(COMPASS2009_DATA,COMPASS2009,'k-','x')
plt.subplot(4,3,11)
plotSivAsymm(COMPASS2009_DATA,COMPASS2009,'k-','z')
plt.subplot(4,3,12)
plotSivAsymm(COMPASS2009_DATA,COMPASS2009,'k-','phT')
#### Here the user needs to define the plot title based on the data set ####
plt.savefig(str(OutputFolder)+'/'+'COMPASS09_comparison.pdf', format='pdf', bbox_inches='tight')


fig4=plt.figure(4,figsize=(15,20))
plt.subplot(4,3,1)
plotSivAsymm(COMPASS2015_DATA,COMPASS2015,'pi+','x')
plt.subplot(4,3,2)
plotSivAsymm(COMPASS2015_DATA,COMPASS2015,'pi+','z')
plt.subplot(4,3,3)
plotSivAsymm(COMPASS2015_DATA,COMPASS2015,'pi+','phT')
plt.subplot(4,3,4)
plotSivAsymm(COMPASS2015_DATA,COMPASS2015,'pi-','x')
plt.subplot(4,3,5)
plotSivAsymm(COMPASS2015_DATA,COMPASS2015,'pi-','z')
plt.subplot(4,3,6)
plotSivAsymm(COMPASS2015_DATA,COMPASS2015,'pi-','phT')
plt.subplot(4,3,7)
plotSivAsymm(COMPASS2015_DATA,COMPASS2015,'k+','x')
plt.subplot(4,3,8)
plotSivAsymm(COMPASS2015_DATA,COMPASS2015,'k+','z')
plt.subplot(4,3,9)
plotSivAsymm(COMPASS2015_DATA,COMPASS2015,'k+','phT')
plt.subplot(4,3,10)
plotSivAsymm(COMPASS2015_DATA,COMPASS2015,'k-','x')
plt.subplot(4,3,11)
plotSivAsymm(COMPASS2015_DATA,COMPASS2015,'k-','z')
plt.subplot(4,3,12)
plotSivAsymm(COMPASS2015_DATA,COMPASS2015,'k-','phT')
#### Here the user needs to define the plot title based on the data set ####
plt.savefig(str(OutputFolder)+'/'+'COMPASS15_comparison.pdf', format='pdf', bbox_inches='tight')












