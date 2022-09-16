import lhapdf
import numpy as np
import pandas as pd
from Input_Parameterization import *
from Sivers_SIDIS_Definitions_for_Pseudo import *

#parms_df = pd.read_csv('Parameters.csv')
test_pars=([7.0, 0.89, 2.78, 19.4, -0.07, -2.33, 2.5, 15.8, -0.29, -14, 4.9, 3, 0])
test_errs=([0.6, 0.05, 0.17, 1.6, 0.06, 0.31, 0.4, 3.2, 0.27, 10, 3.3, 2, 0.18])
# These parameters gave chi2/N = 531.2/313 = 1.69 for MINUIT fit with 
# HERMES2009, COMPASS2009, COMPASS2015 data (HERMES2020 wasn't included)
parms_df = pd.read_csv('./PseudoData/Parameters.csv')
Parameters_array = parms_df.to_numpy()

HERMES2009 = './Data/HERMES_p_2009.csv'
HERMES2020 = './Data/HERMES_p_2020.csv'
COMPASS2009 = './Data/COMPASS_d_2009.csv'
COMPASS2015 = './Data/COMPASS_p_2015.csv'

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
        tempsiv=2*NNq(x,Nq,aq,bq)*hfunc(kp,tempM1)*(np.exp((-kp**2)/(Kp2A**2)))*(1/((np.pi)*(Kp2A**2)))*(xFxQ2(dataset,flavor,x,QQ))
    if(flavor==1):
        Nq=fitresult[5]
        aq=fitresult[6]
        bq=fitresult[7]
        tempsiv=2*NNq(x,Nq,aq,bq)*hfunc(kp,tempM1)*(np.exp((-kp**2)/(Kp2A**2)))*(1/((np.pi)*(Kp2A**2)))*(xFxQ2(dataset,flavor,x,QQ))
    if(flavor==3):
        Nq=fitresult[9]
        aq=fitresult[10]
        bq=fitresult[11]
        tempsiv=2*NNq(x,Nq,aq,bq)*hfunc(kp,tempM1)*(np.exp((-kp**2)/(Kp2A**2)))*(1/((np.pi)*(Kp2A**2)))*(xFxQ2(dataset,flavor,x,QQ))
    return tempsiv
    
    
def SiversFuncAntiQ(dataset,flavor,x,QQ,kp,fitresult):
    tempM1=fitresult[0]
    if(flavor==-2):
        tempM1=fitresult[0]
        Nq=fitresult[4]
        tempsiv=2*NNqbar(x,Nq)*hfunc(kp,tempM1)*(np.exp((-kp**2)/(Kp2A**2)))*(1/((np.pi)*(Kp2A**2)))*(xFxQ2(dataset,flavor,x,QQ))
    if(flavor==-1):
        tempM1=fitresult[0]
        Nq=fitresult[8]
        tempsiv=2*NNqbar(x,Nq)*hfunc(kp,tempM1)*(np.exp((-kp**2)/(Kp2A**2)))*(1/((np.pi)*(Kp2A**2)))*(xFxQ2(dataset,flavor,x,QQ))
    if(flavor==-3):
        tempM1=fitresult[0]
        Nq=fitresult[12]
        tempsiv=2*NNqbar(x,Nq)*hfunc(kp,tempM1)*(np.exp((-kp**2)/(Kp2A**2)))*(1/((np.pi)*(Kp2A**2)))*(xFxQ2(dataset,flavor,x,QQ))
    return tempsiv
    

def plotSiversQ(flavor,ParmResults,col):
    tempkT=np.linspace(0, 1.5)
    tempSiv=[SiversFuncQ(PDFdataset,flavor,0.1,2.4,tempkT[i],ParmResults) for i in range(0,len(tempkT))]
    plt.plot(tempkT,tempSiv, color = col)
    #return tempSiv

def plotSiversAntiQ(flavor,ParmResults):
    tempkT=np.linspace(0, 1.5)
    tempSiv=[SiversFuncAntiQ(PDFdataset,flavor,0.1,2.4,tempkT[i],ParmResults) for i in range(0,len(tempkT))]
    plt.plot(tempkT,tempSiv, color = col)

# plt.figure(1)
# plotSiversQ(2,test_pars, 'b')
# plotSiversQ(1,test_pars, 'r')
# plotSiversQ(3,test_pars, 'g')
# plt.savefig('PseudoData/testplot.pdf')

    
# def plotSiversQBand(flavor,array,col,lbl,ParmResults):
#     tempkT=np.linspace(0, 1.5)
#     lenarray=len(array)
#     tempASivVal=[]
#     for j in range(0,lenarray):
#         ttt=[SiversFuncQ(PDFdataset,flavor,0.1,2.4,tempkT[i],array[j]) for i in range(0,len(tempkT))]
#         plt.plot(tempkT,ttt,color=col,alpha=0.1) 
#     tempSiv=[SiversFuncQ(PDFdataset,flavor,0.1,2.4,tempkT[i],ParmResults) for i in range(0,len(tempkT))]
#     plt.plot(tempkT,tempSiv,col,label=lbl)
#     #return tempSiv
    
    
# def plotSiversAntiQBand(flavor,array,col,lbl,ParmResults):
#     tempkT=np.linspace(0, 1.5)
#     lenarray=len(array)
#     tempASivVal=[]
#     for j in range(0,lenarray):
#         ttt=[SiversFuncAntiQ(PDFdataset,flavor,0.1,2.4,tempkT[i],array[j]) for i in range(0,len(tempkT))]
#         plt.plot(tempkT,ttt,color=col,alpha=0.1) 
#     tempSiv=[SiversFuncAntiQ(PDFdataset,flavor,0.1,2.4,tempkT[i],ParmResults) for i in range(0,len(tempkT))]
#     plt.plot(tempkT,tempSiv,col,label=lbl)


    
# ### Different style band

def plotSiversQBandFill(flavor,array,col,lbl,ParmResults):
    tempkT=np.linspace(0, 1.5)
    lenarray=len(array)
    tempASivVal=[]
    tempSiv=[SiversFuncQ(PDFdataset,flavor,0.1,2.4,tempkT[i],ParmResults) for i in range(0,len(tempkT))]
    Smax=[]
    Smin=[]
    for i in range(0,len(tempkT)):
        tempmax = SiversFuncQ(PDFdataset,flavor,0.1,2.4,tempkT[i],ParmResults)
        tempmin = SiversFuncQ(PDFdataset,flavor,0.1,2.4,tempkT[i],ParmResults)
        for j in range(0,lenarray):
            modified_array = np.delete(np.array(array[j]), 0)
            ttt=SiversFuncQ(PDFdataset,flavor,0.1,2.4,tempkT[i],modified_array)
            if ttt > tempmax:
                tempmax = ttt
            elif ttt < tempmin:
                tempmin = ttt
        Smax.append(tempmax)
        Smin.append(tempmin)
    plt.fill_between(tempkT,Smin,Smax,alpha=0.4,color=col,linewidth=0.01)    
    plt.plot(tempkT,tempSiv,col,label=lbl)

    
def plotSiversAntiQBandFill(flavor,array,col,lbl,ParmResults):
    tempkT=np.linspace(0, 1.5)
    lenarray=len(array)
    tempASivVal=[]
    tempSiv=[SiversFuncAntiQ(PDFdataset,flavor,0.1,2.4,tempkT[i],ParmResults) for i in range(0,len(tempkT))]
    #print(tempSiv)
    Smax=[]
    Smin=[]
    for i in range(0,len(tempkT)):
        tempmax = SiversFuncAntiQ(PDFdataset,flavor,0.1,2.4,tempkT[i],ParmResults)
        tempmin = SiversFuncAntiQ(PDFdataset,flavor,0.1,2.4,tempkT[i],ParmResults)
        for j in range(0,lenarray):
            modified_array = np.delete(np.array(array[j]), 0)
            ttt=SiversFuncAntiQ(PDFdataset,flavor,0.1,2.4,tempkT[i],modified_array)
            if ttt > tempmax:
                tempmax = ttt
            elif ttt < tempmin:
                tempmin = ttt
        Smax.append(tempmax)
        Smin.append(tempmin)
    plt.fill_between(tempkT,Smin,Smax,alpha=0.4,color=col,linewidth=0.01)    
    plt.plot(tempkT,tempSiv,col,label=lbl)
    
plt.figure(1)
plotSiversQBandFill(2,Parameters_array,'b','$u$',test_pars) 
plotSiversQBandFill(1,Parameters_array,'r','$d$',test_pars)
plotSiversQBandFill(3,Parameters_array,'g','$s$',test_pars)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylim(-0.12,0.12)
plt.legend(loc=4,fontsize=20,handlelength=3)
plt.savefig('PseudoData/testplotQ.pdf')

plt.figure(2)
plotSiversAntiQBandFill(-2,Parameters_array,'b','$u$',test_pars) 
plotSiversAntiQBandFill(-1,Parameters_array,'r','$d$',test_pars)
plotSiversAntiQBandFill(-3,Parameters_array,'g','$s$',test_pars)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylim(-0.012,0.012)
plt.legend(loc=4,fontsize=20,handlelength=3)
plt.savefig('PseudoData/testplotAntiQ.pdf')
    
# def h(m1, kperp):
#     return np.sqrt(2*ee) * (kperp/m1) * np.exp(-kperp**2/m1**2)


# def fqp(x, QQ, kperp2avg, kperp, flavor):
#     fq = pdf(flavor, x, QQ)
#     return fq*(1/(np.pi*kperp2avg))*np.exp(-kperp**2/kperp2avg)
   
### need to be modified using plot_definitions ###
# def Sivers_Single(x, QQ, kperp2avg, flavor, kperp):
#     refDict = {-3: 'nnsbar',
#               -2: 'nnubar',
#               -1: 'nndbar',
#               1: 'nnd',
#               2: 'nnu',
#               3: 'nns'}
#     nnqval = nnq(model, np.array([x]), refDict[flavor])
#     #nnqval = nnq(model , np.array([x]), refDict[flavor])[:,0]
#     hval = h(model, kperp)
#     fqpval = fqp([x], [QQ], kperp2avg, kperp, flavor)
#     return ((2*nnqval*hval*fqpval)[0, :])



# def xsivdistFromReplicas(numReplicas, x, QQ, kperp2avg, kperp):
#     tempfu = []
#     tempfd = []
#     tempfs = []
#     tempfubar = []
#     tempfdbar = []
#     tempfsbar = []
#     for i in range(numReplicas):
#         #t = tf.keras.models.load_model(Models_folder+'/'+ str(folders_array[i]), 
#         #                                 custom_objects={'A0': A0, 'Quotient': Quotient})
#         t = SIDISmodelsArray[i]
#         tempfu.append(list(xsivdist(t, x, QQ, kperp2avg, 2, kperp)))
#         tempfd.append(list(xsivdist(t, x, QQ, kperp2avg, 1, kperp)))
#         tempfs.append(list(xsivdist(t, x, QQ, kperp2avg, 3, kperp)))
#         tempfubar.append(list(xsivdist(t, x, QQ, kperp2avg, -2, kperp)))
#         tempfdbar.append(list(xsivdist(t, x, QQ, kperp2avg, -1, kperp)))
#         tempfsbar.append(list(xsivdist(t, x, QQ, kperp2avg, -3, kperp)))
#     return np.array(tempfu),np.array(tempfubar),np.array(tempfd),np.array(tempfdbar),np.array(tempfs),np.array(tempfsbar)

    
    
# def SivDistBandsCSVgen(numReplicas, x, QQ, kperp2avg, kperp, numSigma=1):
#     data_dictionary={"kperp":[],"fu":[],"fuErr":[],"fubar":[],"fubarErr":[],"fd":[],"fdErr":[],"fdbar":[],"fdbarErr":[],"fs":[],"fsErr":[],"fsbar":[],"fsbarErr":[]}
#     results = xsivdistFromReplicas(numReplicas, x, QQ, kperp2avg, kperp)
#     tempfu = results[0].mean(axis=0)
#     tempfuErr = results[0].std(axis=0)
#     tempfubar = results[1].mean(axis=0)
#     tempfubarErr = results[1].std(axis=0)
#     tempfd = results[2].mean(axis=0)
#     tempfdErr = results[2].std(axis=0)
#     tempfdbar = results[3].mean(axis=0)
#     tempfdbarErr = results[3].std(axis=0)
#     tempfs = results[4].mean(axis=0)
#     tempfsErr = results[4].std(axis=0)
#     tempfsbar = results[5].mean(axis=0)
#     tempfsbarErr = results[5].std(axis=0)
#     kp = np.array(kperp)
#     data_dictionary["kperp"]=kp
#     data_dictionary["fu"]=tempfu
#     data_dictionary["fuErr"]=tempfuErr
#     data_dictionary["fubar"]=tempfubar
#     data_dictionary["fubarErr"]=tempfubarErr
#     data_dictionary["fd"]=tempfd
#     data_dictionary["fdErr"]=tempfdErr
#     data_dictionary["fdbar"]=tempfdbar
#     data_dictionary["fdbarErr"]=tempfdbarErr
#     data_dictionary["fs"]=tempfs
#     data_dictionary["fsErr"]=tempfsErr
#     data_dictionary["fsbar"]=tempfsbar
#     data_dictionary["fsbarErr"]=tempfsbarErr
#     return pd.DataFrame(data_dictionary)
    

# fSivCSV = SivDistBandsCSVgen(numreplicas_SIDIS, 0.1, 2.4, 0.25, np.array(list(range(150)))/100)
# fSivCSV.to_csv(str(OutputFolder)+'/'+'Sivfuncs.csv')

