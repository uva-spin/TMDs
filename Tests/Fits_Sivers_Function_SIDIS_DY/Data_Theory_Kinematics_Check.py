import pandas as pd
import numpy as np
from Global_Constants import *
from Sivers_SIDIS_Definitions import *
from Sivers_DY_Definitions import *

def dataslice_Comp(filename,Had,Var):
    tempdf=pd.read_csv(filename)
    temp_slice=tempdf[(tempdf["hadron"]==Had)&(tempdf["1D_dependence"]==Var)]
    tempHad=np.array(temp_slice["hadron"],dtype=object)
    tempDep=np.array(temp_slice["1D_dependence"],dtype=object)
    tempQ2=np.array(temp_slice["Q2"],dtype=object)
    tempX=np.array(temp_slice["x"],dtype=object)
    tempZ=np.array(temp_slice["z"],dtype=object)
    tempPHT=np.array(temp_slice["phT"],dtype=object)
    tempSiv=np.array(temp_slice["Siv"],dtype=object)
    #temperrSiv=np.array(temp_slice["tot_err"],dtype=object)
    return tempHad,tempDep,tempQ2,tempX,tempZ,tempPHT,tempSiv



def ASiv_data_Comp(datfile,hadron):
    tempdf_dat=pd.DataFrame({'Hadron':[],'Dependence':[],'QQ':[],'x':[],'z':[],'phT':[],'A_SIDIS':[]})
    tempXfile=dataslice_Comp(datfile,hadron,"x")
    tempZfile=dataslice_Comp(datfile,hadron,"z")
    tempPhTfile=dataslice_Comp(datfile,hadron,"phT")   
    ##### Hadron ################
    tempHad_x=np.array(tempXfile[0],dtype=object)
    tempHad_z=np.array(tempZfile[0],dtype=object)
    tempHad_phT=np.array(tempPhTfile[0],dtype=object)
    tempHad=np.concatenate((tempHad_x,tempHad_z,tempHad_phT))
    ##### Dependence ################
    tempDep_x=np.array(tempXfile[1],dtype=object)
    tempDep_z=np.array(tempZfile[1],dtype=object)
    tempDep_phT=np.array(tempPhTfile[1],dtype=object)
    tempDep=np.concatenate((tempDep_x,tempDep_z,tempDep_phT))
    ##### Q2 ################
    tempQ2_x=np.array(tempXfile[2],dtype=object)
    tempQ2_z=np.array(tempZfile[2],dtype=object)
    tempQ2_phT=np.array(tempPhTfile[2],dtype=object)
    tempQ2=np.concatenate((tempQ2_x,tempQ2_z,tempQ2_phT))
    ##### X ################
    tempX_x=np.array(tempXfile[3],dtype=object)
    tempX_z=np.array(tempZfile[3],dtype=object)
    tempX_phT=np.array(tempPhTfile[3],dtype=object)
    tempX=np.concatenate((tempX_x,tempX_z,tempX_phT))
    ##### Z ################
    tempZ_x=np.array(tempXfile[4],dtype=object)
    tempZ_z=np.array(tempZfile[4],dtype=object)
    tempZ_phT=np.array(tempPhTfile[4],dtype=object)
    tempZ=np.concatenate((tempZ_x,tempZ_z,tempZ_phT))
    ##### phT ################
    tempphT_x=np.array(tempXfile[5],dtype=object)
    tempphT_z=np.array(tempZfile[5],dtype=object)
    tempphT_phT=np.array(tempPhTfile[5],dtype=object)
    tempphT=np.concatenate((tempphT_x,tempphT_z,tempphT_phT))    
    ##### Asy ################
    tempAsy_x=np.array(tempXfile[6],dtype=object)
    tempAsy_z=np.array(tempZfile[6],dtype=object)
    tempAsy_phT=np.array(tempPhTfile[6],dtype=object)
    tempAsy=np.concatenate((tempAsy_x,tempAsy_z,tempAsy_phT))
    tempdf_dat['Hadron']=tempHad
    tempdf_dat['Dependence']=tempDep
    tempdf_dat['QQ']=tempQ2
    tempdf_dat['x']=tempX
    tempdf_dat['z']=tempZ
    tempdf_dat['phT']=tempphT
    tempdf_dat['A_SIDIS']=tempAsy    
    return tempdf_dat

    #    return tempAsy,tempAsyErr


def ASiv_Val_Comp(datfile):
    had_len=len(hadarray(datfile))
    temHads=hadarray(datfile)
    temp_SivData=[]
    for i in range(0,had_len):
        temp_SivData.append(ASiv_data_Comp(datfile,temHads[i]))        
    return pd.concat((temp_SivData))




def Asymmetry_Kinematics_for_Hadron(SIDISdatafilename,hadron,dep):
    #kperp2Avg=Kp2A
    #pperpAvg=Pp2A
    if(SIDISdatafilename==SIDIS_DataFilesArray[0]):
        PDFfile=SIDIS_PDFs_Array[0]
        if(hadron=='pi+'):
            FFfile=SIDIS_FFs_HERMES_p_2020[0]
        elif(hadron=='pi-'):
            FFfile=SIDIS_FFs_HERMES_p_2020[1]
        elif(hadron=='pi0'):
            FFfile=SIDIS_FFs_HERMES_p_2020[2]                      
        elif(hadron=='k+'):
            FFfile=SIDIS_FFs_HERMES_p_2020[3]                      
        elif(hadron=='k-'):
            FFfile=SIDIS_FFs_HERMES_p_2020[4]                      
    elif(SIDISdatafilename==SIDIS_DataFilesArray[1]):
        PDFfile=SIDIS_PDFs_Array[1]
        if(hadron=='pi+'):
            FFfile=SIDIS_FFs_COMPASS_d_2009[0]
        elif(hadron=='pi-'):
            FFfile=SIDIS_FFs_COMPASS_d_2009[1]
        elif(hadron=='pi0'):
            FFfile=SIDIS_FFs_COMPASS_d_2009[2]                      
        elif(hadron=='k+'):
            FFfile=SIDIS_FFs_COMPASS_d_2009[3]                      
        elif(hadron=='k-'):
            FFfile=SIDIS_FFs_COMPASS_d_2009[4]                      
    elif(SIDISdatafilename==SIDIS_DataFilesArray[2]):
        PDFfile=SIDIS_PDFs_Array[2]
        if(hadron=='pi+'):
            FFfile=SIDIS_FFs_COMPASS_p_2015[0]
        elif(hadron=='pi-'):
            FFfile=SIDIS_FFs_COMPASS_p_2015[1]
        elif(hadron=='pi0'):
            FFfile=SIDIS_FFs_COMPASS_p_2015[2]                      
        elif(hadron=='k+'):
            FFfile=SIDIS_FFs_COMPASS_p_2015[3]                      
        elif(hadron=='k-'):
            FFfile=SIDIS_FFs_COMPASS_p_2015[4]                             
    tempvals_all=pd.read_csv(PDFfile)
    #tempvals=tempvals_all[(tempvals_all["hadron"]==hadron)]
    tempvals=tempvals_all[(tempvals_all["hadron"]==hadron)&(tempvals_all["1D_dependence"]==dep)]
    tempdf_had=pd.DataFrame({'Hadron':[],'Dependence':[],'QQ':[],'x':[],'z':[],'phT':[]})
    had=tempvals['hadron']
    dependence=tempvals['1D_dependence']
    QQ=tempvals['QQ']
    x=tempvals['x']
    z=tempvals['z']
    phT=tempvals['phT']
    tempdf_had['Hadron']=had
    tempdf_had['Dependence']=dependence
    tempdf_had['QQ']=QQ
    tempdf_had['x']=x
    tempdf_had['z']=z
    tempdf_had['phT']=phT
    return tempdf_had




def Theory_Kinematics(datfile):
    had_len=len(hadarray(datfile))
    temHads=hadarray(datfile)
    fittot=[]
    tempdf=pd.DataFrame({'Hadron':[],'Dependence':[],'QQ':[],'x':[],'z':[],'phT':[]})
    for i in range(0,had_len):
        if temHads[i]=="pi+":
            tempfitx=Asymmetry_Kinematics_for_Hadron(datfile,"pi+","x")
            tempfitz=Asymmetry_Kinematics_for_Hadron(datfile,"pi+","z")
            tempfitphT=Asymmetry_Kinematics_for_Hadron(datfile,"pi+","phT")
            tempfit=pd.concat((tempfitx,tempfitz,tempfitphT))
            fittot.append(tempfit)
            tempdf.append(tempfit)
        elif temHads[i]=="pi-":
            tempfitx=Asymmetry_Kinematics_for_Hadron(datfile,"pi-","x")
            tempfitz=Asymmetry_Kinematics_for_Hadron(datfile,"pi-","z")
            tempfitphT=Asymmetry_Kinematics_for_Hadron(datfile,"pi-","phT")
            tempfit=pd.concat((tempfitx,tempfitz,tempfitphT))
            fittot.append(tempfit)
            tempdf.append(tempfit)
        elif temHads[i]=="pi0":
            tempfitx=Asymmetry_Kinematics_for_Hadron(datfile,"pi0","x")
            tempfitz=Asymmetry_Kinematics_for_Hadron(datfile,"pi0","z")
            tempfitphT=Asymmetry_Kinematics_for_Hadron(datfile,"pi0","phT")
            tempfit=pd.concat((tempfitx,tempfitz,tempfitphT))
            fittot.append(tempfit)
            tempdf.append(tempfit)
        elif temHads[i]=="k+":
            tempfitx=Asymmetry_Kinematics_for_Hadron(datfile,"k+","x")
            tempfitz=Asymmetry_Kinematics_for_Hadron(datfile,"k+","z")
            tempfitphT=Asymmetry_Kinematics_for_Hadron(datfile,"k+","phT")
            tempfit=pd.concat((tempfitx,tempfitz,tempfitphT))
            fittot.append(tempfit)
            tempdf.append(tempfit)
        elif temHads[i]=="k-":
            tempfitx=Asymmetry_Kinematics_for_Hadron(datfile,"k-","x")
            tempfitz=Asymmetry_Kinematics_for_Hadron(datfile,"k-","z")
            tempfitphT=Asymmetry_Kinematics_for_Hadron(datfile,"k-","phT")
            tempfit=pd.concat((tempfitx,tempfitz,tempfitphT))
            fittot.append(tempfit)
            tempdf.append(tempfit)
    return pd.concat((fittot))



Th_Kins=Theory_Kinematics(Dat2)
Th_Kins.to_csv('HERMES2020_Theory_Kins.csv')
Data_Kins=ASiv_Val_Comp(Dat2)
Data_Kins.to_csv('HERMES2020_Data_Kins.csv')


Th_Kins=Theory_Kinematics(Dat3)
Th_Kins.to_csv('COMPASS2009_Theory_Kins.csv')
Data_Kins=ASiv_Val_Comp(Dat3)
Data_Kins.to_csv('COMPASS2009_Data_Kins.csv')


Th_Kins=Theory_Kinematics(Dat4)
Th_Kins.to_csv('COMPASS2015_Theory_Kins.csv')
Data_Kins=ASiv_Val_Comp(Dat4)
Data_Kins.to_csv('COMPASS2015_Data_Kins.csv')