import lhapdf
import pandas as pd
import numpy as np
PDFdataset = lhapdf.mkPDF("cteq61")

FF_PiP_dataset=["NNFF10_PIp_nlo"]
FF_PiM_dataset=["NNFF10_PIm_nlo"]
FF_Pi0_dataset=["NNFF10_PIsum_nlo"]
FF_KP_dataset=["NNFF10_KAp_nlo"]
FF_KM_dataset=["NNFF10_KAm_nlo"]

SIDIS_Dat1='../Data/HERMES_p_2009.csv'
SIDIS_Dat2='../Data/HERMES_p_2020.csv'
SIDIS_Dat3='../Data/COMPASS_d_2009.csv'
SIDIS_Dat4='../Data/COMPASS_p_2015.csv'
SIDIS_DataFilesArray=[SIDIS_Dat1,SIDIS_Dat2,SIDIS_Dat3,SIDIS_Dat4]


DY_Dat1='../Data/COMPASS_p_DY_2017.csv'


#### SIDIS Process ###############

def SIDIS_xFxQ2(dataset,flavor,x,QQ):
    temp_parton_dist_x=np.array(dataset.xfxQ2(flavor, x, QQ),dtype=object)
    #temp_parton_dist_x=np.array(dataset.xfxQ(flavor, x, QQ),dtype=object)
    return temp_parton_dist_x

def generate_SIDIS_PDFs(datafile,PDFset,OutFileName):
    tempvals=pd.read_csv(datafile)
    dataframe=pd.DataFrame({'hadron':[],'1D_dependence':[],'x':[],'z':[],'phT':[],'QQ':[],'sbar':[],'ubar':[],'dbar':[],'d':[],'u':[],'s':[]})
    temphad=tempvals['hadron']
    tempdep=tempvals['1D_dependence']
    tempx=tempvals['x']
    tempz=tempvals['z']
    tempphT=tempvals['phT']
    tempQQ=tempvals['Q2']
    dataframe['hadron']=temphad 
    dataframe['1D_dependence']=tempdep
    dataframe['x']=tempx
    dataframe['z']=tempz
    dataframe['phT']=tempphT
    dataframe['QQ']=tempQQ
    dataframe['sbar']=SIDIS_xFxQ2(PDFset,-3,tempx,tempQQ)
    dataframe['ubar']=SIDIS_xFxQ2(PDFset,-2,tempx,tempQQ)
    dataframe['dbar']=SIDIS_xFxQ2(PDFset,-1,tempx,tempQQ)
    dataframe['d']=SIDIS_xFxQ2(PDFset,1,tempx,tempQQ)
    dataframe['u']=SIDIS_xFxQ2(PDFset,2,tempx,tempQQ)
    dataframe['s']=SIDIS_xFxQ2(PDFset,3,tempx,tempQQ)
    print(dataframe)
    return dataframe.to_csv(OutFileName)


def SIDIS_FF_zFzQ(dataset,flavor,zz,QQ):
    # Here "0" represents the central values from the girds
    temp_zD1=lhapdf.mkPDF(dataset[0], 0)
    zD1_vec=np.array(temp_zD1.xfxQ2(flavor,zz,QQ),dtype=object)
    return zD1_vec


def generate_SIDIS_FFs(datafile,FFset,hadron,OutFileName):
    tempvals_all=pd.read_csv(datafile)
    tempvals=tempvals_all[(tempvals_all["hadron"]==hadron)]
    dataframe=pd.DataFrame({'hadron':[],'1D_dependence':[],'z':[],'QQ':[],'sbar':[],'ubar':[],'dbar':[],'d':[],'u':[],'s':[]})
    temphad=tempvals['hadron']
    tempdep=tempvals['1D_dependence']
    tempz=tempvals['z']
    tempQQ=tempvals['Q2']
    dataframe['hadron']=temphad
    dataframe['1D_dependence']=tempdep
    dataframe['z']=tempz
    dataframe['QQ']=tempQQ
    dataframe['sbar']=SIDIS_FF_zFzQ(FFset,-3,tempz,tempQQ)
    dataframe['ubar']=SIDIS_FF_zFzQ(FFset,-2,tempz,tempQQ)
    dataframe['dbar']=SIDIS_FF_zFzQ(FFset,-1,tempz,tempQQ)
    dataframe['d']=SIDIS_FF_zFzQ(FFset,1,tempz,tempQQ)
    dataframe['u']=SIDIS_FF_zFzQ(FFset,2,tempz,tempQQ)
    dataframe['s']=SIDIS_FF_zFzQ(FFset,3,tempz,tempQQ)
    print(dataframe)
    return dataframe.to_csv(OutFileName)


#### DY Process ######################

def DY_xFxQ2(dataset,flavor,x,QQ):
    temp_parton_dist_x=np.array(dataset.xfxQ(flavor, x, QQ),dtype=object)
    #temp_parton_dist_x=np.array(dataset.xfxQ(flavor, x, QQ),dtype=object)
    return temp_parton_dist_x

def generate_DY_PDFs(datafile,PDFset,OutFileName_x1,OutFileName_x2):
    tempvals=pd.read_csv(datafile)
    tempQM=tempvals['QM']
    tempQT=tempvals['QT']
    ####### Generating grids for x1 ###################
    dataframe_x1=pd.DataFrame({'x':[],'QM':[],'QT':[],'sbar':[],'ubar':[],'dbar':[],'d':[],'u':[],'s':[]})
    tempx1=tempvals['x1']
    dataframe_x1['x']=tempx1
    dataframe_x1['QM']=tempQM
    dataframe_x1['QT']=tempQT    
    dataframe_x1['sbar']=SIDIS_xFxQ2(PDFset,-3,tempx1,tempQM)
    dataframe_x1['ubar']=SIDIS_xFxQ2(PDFset,-2,tempx1,tempQM)
    dataframe_x1['dbar']=SIDIS_xFxQ2(PDFset,-1,tempx1,tempQM)
    dataframe_x1['d']=SIDIS_xFxQ2(PDFset,1,tempx1,tempQM)
    dataframe_x1['u']=SIDIS_xFxQ2(PDFset,2,tempx1,tempQM)
    dataframe_x1['s']=SIDIS_xFxQ2(PDFset,3,tempx1,tempQM)
    dataframe_x1.to_csv(OutFileName_x1)
    ####### Generating grids for x2 ###################
    dataframe_x2=pd.DataFrame({'x':[],'QM':[],'QT':[],'sbar':[],'ubar':[],'dbar':[],'d':[],'u':[],'s':[]})
    tempx2=tempvals['x2']
    dataframe_x2['x']=tempx2
    dataframe_x2['QM']=tempQM
    dataframe_x2['QT']=tempQT    
    dataframe_x2['sbar']=SIDIS_xFxQ2(PDFset,-3,tempx2,tempQM)
    dataframe_x2['ubar']=SIDIS_xFxQ2(PDFset,-2,tempx2,tempQM)
    dataframe_x2['dbar']=SIDIS_xFxQ2(PDFset,-1,tempx2,tempQM)
    dataframe_x2['d']=SIDIS_xFxQ2(PDFset,1,tempx2,tempQM)
    dataframe_x2['u']=SIDIS_xFxQ2(PDFset,2,tempx2,tempQM)
    dataframe_x2['s']=SIDIS_xFxQ2(PDFset,3,tempx2,tempQM)
    dataframe_x2.to_csv(OutFileName_x2)
    return dataframe_x1,dataframe_x2


##### Lets generate SIDIS PDF grids #####

generate_SIDIS_PDFs(SIDIS_Dat1,PDFdataset,'SIDIS_PDFs/PDFs_HERMES_p_2009.csv')
generate_SIDIS_PDFs(SIDIS_Dat2,PDFdataset,'SIDIS_PDFs/PDFs_HERMES_p_2020.csv')
generate_SIDIS_PDFs(SIDIS_Dat3,PDFdataset,'SIDIS_PDFs/PDFs_COMPASS_d_2009.csv')
generate_SIDIS_PDFs(SIDIS_Dat4,PDFdataset,'SIDIS_PDFs/PDFs_COMPASS_p_2015.csv')

##### Lets generate SIDIS FF grids #####

generate_SIDIS_FFs(SIDIS_Dat1,FF_PiP_dataset,'pi+','SIDIS_FFs/FF_PiP_HERMES_p_2009.csv')
generate_SIDIS_FFs(SIDIS_Dat2,FF_PiP_dataset,'pi+','SIDIS_FFs/FF_PiP_HERMES_p_2020.csv')
generate_SIDIS_FFs(SIDIS_Dat3,FF_PiP_dataset,'pi+','SIDIS_FFs/FF_PiP_COMPASS_d_2009.csv')
generate_SIDIS_FFs(SIDIS_Dat4,FF_PiP_dataset,'pi+','SIDIS_FFs/FF_PiP_COMPASS_p_2015.csv')

generate_SIDIS_FFs(SIDIS_Dat1,FF_PiM_dataset,'pi-','SIDIS_FFs/FF_PiM_HERMES_p_2009.csv')
generate_SIDIS_FFs(SIDIS_Dat2,FF_PiM_dataset,'pi-','SIDIS_FFs/FF_PiM_HERMES_p_2020.csv')
generate_SIDIS_FFs(SIDIS_Dat3,FF_PiM_dataset,'pi-','SIDIS_FFs/FF_PiM_COMPASS_d_2009.csv')
generate_SIDIS_FFs(SIDIS_Dat4,FF_PiM_dataset,'pi-','SIDIS_FFs/FF_PiM_COMPASS_p_2015.csv')

generate_SIDIS_FFs(SIDIS_Dat1,FF_Pi0_dataset,'pi0','SIDIS_FFs/FF_Pi0_HERMES_p_2009.csv')
generate_SIDIS_FFs(SIDIS_Dat2,FF_Pi0_dataset,'pi0','SIDIS_FFs/FF_Pi0_HERMES_p_2020.csv')
generate_SIDIS_FFs(SIDIS_Dat3,FF_Pi0_dataset,'pi0','SIDIS_FFs/FF_Pi0_COMPASS_d_2009.csv')
generate_SIDIS_FFs(SIDIS_Dat4,FF_Pi0_dataset,'pi0','SIDIS_FFs/FF_Pi0_COMPASS_p_2015.csv')

generate_SIDIS_FFs(SIDIS_Dat1,FF_KP_dataset,'k+','SIDIS_FFs/FF_KP_HERMES_p_2009.csv')
generate_SIDIS_FFs(SIDIS_Dat2,FF_KP_dataset,'k+','SIDIS_FFs/FF_KP_HERMES_p_2020.csv')
generate_SIDIS_FFs(SIDIS_Dat3,FF_KP_dataset,'k+','SIDIS_FFs/FF_KP_COMPASS_d_2009.csv')
generate_SIDIS_FFs(SIDIS_Dat4,FF_KP_dataset,'k+','SIDIS_FFs/FF_KP_COMPASS_p_2015.csv')

generate_SIDIS_FFs(SIDIS_Dat1,FF_KM_dataset,'k-','SIDIS_FFs/FF_KM_HERMES_p_2009.csv')
generate_SIDIS_FFs(SIDIS_Dat2,FF_KM_dataset,'k-','SIDIS_FFs/FF_KM_HERMES_p_2020.csv')
generate_SIDIS_FFs(SIDIS_Dat3,FF_KM_dataset,'k-','SIDIS_FFs/FF_KM_COMPASS_d_2009.csv')
generate_SIDIS_FFs(SIDIS_Dat4,FF_KM_dataset,'k-','SIDIS_FFs/FF_KM_COMPASS_p_2015.csv')



##### Lets generate DY PDF grids #####

generate_DY_PDFs(DY_Dat1,PDFdataset,'DY_PDFs/PDFs_x1_COMPASS_p_DY_2017.csv','DY_PDFs/PDFs_x2_COMPASS_p_DY_2017.csv')


