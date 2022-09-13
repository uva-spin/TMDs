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
#parms_df = pd.read_csv('./PseudoData/Parameters.csv')
#Parameters_array = parms_df.to_numpy()

HERMES2009 = './Data/HERMES_p_2009.csv'
HERMES2020 = './Data/HERMES_p_2020.csv'
COMPASS2009 = './Data/COMPASS_d_2009.csv'
COMPASS2015 = './Data/COMPASS_p_2015.csv'

def Create_SIDIS_Asym_Data(datafile, m1, Nu, au, bu, Nub, Nd, ad, bd, Ndb, Ns, aS, bS, Nsb):
    tempdf=pd.read_csv(datafile)
    temphad=np.array(tempdf['hadron'],dtype=object)
    tempQ2=np.array(tempdf['Q2'],dtype=object)
    tempX=np.array(tempdf['x'],dtype=object)
    tempY=np.array(tempdf['y'],dtype=object)
    tempZ=np.array(tempdf['z'],dtype=object)
    tempPHT=np.array(tempdf['phT'],dtype=object)
    #tempSivData=np.array(tempdf['Siv'],dtype=object)
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
    temp_Siv = totalfitDataSet(datafile, m1=m1, Nu=Nu, au=au, bu=bu, Nub=Nub, 
              Nd=Nd, ad=ad, bd=bd, Ndb=Ndb,
              Ns=Ns,aS=aS,bS=bS,Nsb=Nsb)
    ############################################
    data_dictionary["Siv"]=temp_Siv
    return pd.DataFrame(data_dictionary)
    
R_SIDIS_HERMES2009=Create_SIDIS_Asym_Data(HERMES2009, *test_pars)
R_SIDIS_HERMES2020=Create_SIDIS_Asym_Data(HERMES2020, *test_pars)
R_SIDIS_COMPASS2009=Create_SIDIS_Asym_Data(COMPASS2009, *test_pars)
R_SIDIS_COMPASS2015=Create_SIDIS_Asym_Data(COMPASS2015, *test_pars)

OutputFolder = 'PseudoData'

R_SIDIS_HERMES2009.to_csv(str(OutputFolder)+'/'+'HERMES2009_Pseudo.csv')
R_SIDIS_HERMES2020.to_csv(str(OutputFolder)+'/'+'HERMES2020_Pseudo.csv')
R_SIDIS_COMPASS2009.to_csv(str(OutputFolder)+'/'+'COMPASS2009_Pseudo.csv')
R_SIDIS_COMPASS2015.to_csv(str(OutputFolder)+'/'+'COMPASS2015_Pseudo.csv')



