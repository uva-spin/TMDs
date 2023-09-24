import numpy as np

## DY ###
#DY_DataFilesArray=np.array(['../Data/COMPASS_p_DY_2017.csv'])

## SIDIS ###
Dat1='../../Data/HERMES13p_cosphi.csv'
Dat2='../../Data/HERMES13p_cos2phi.csv'
Dat3='../../Data/HERMES13d_cosphi.csv'
Dat4='../../Data/HERMES13d_cos2phi.csv'
SIDIS_DataFilesArrayR=[Dat1,Dat2]


###########################################################################
#####################  DY PDFs #########################################
###########################################################################

#DY_PDFs_COMPASS_p_2017_x1='../Calc_Grids_DSS/DY_PDFs/PDFs_x1_COMPASS_p_DY_2017.csv'
#DY_PDFs_COMPASS_p_2017_x2='../Calc_Grids_DSS/DY_PDFs/PDFs_x2_COMPASS_p_DY_2017.csv'

###########################################################################
#####################  SIDIS PDFs #########################################
###########################################################################


SIDIS_PDFs_HERMES13p_cosphi='../Calc_Grids_DSS/SIDIS_PDFs/PDFs_HERMES13p_cosphi.csv'
SIDIS_PDFs_HERMES13p_cos2phi='../Calc_Grids_DSS/SIDIS_PDFs/PDFs_HERMES13p_cos2phi.csv'
SIDIS_PDFs_HERMES13d_cosphi='../Calc_Grids_DSS/SIDIS_PDFs/PDFs_HERMES13d_cosphi.csv'
SIDIS_PDFs_HERMES13d_cos2phi='../Calc_Grids_DSS/SIDIS_PDFs/PDFs_HERMES13d_cos2phi.csv'

SIDIS_PDFs_Array=(SIDIS_PDFs_HERMES13p_cosphi,SIDIS_PDFs_HERMES13p_cos2phi,SIDIS_PDFs_HERMES13d_cosphi,SIDIS_PDFs_HERMES13d_cos2phi)


###########################################################################
#####################  SIDIS FFs #########################################
###########################################################################


SIDIS_FFs_PiP_HERMES13p_cosphi='../Calc_Grids_DSS/SIDIS_FFs/FF_PiP_HERMES13p_cosphi.csv'
SIDIS_FFs_PiP_HERMES13p_cos2phi='../Calc_Grids_DSS/SIDIS_FFs/FF_PiP_HERMES13p_cos2phi.csv'
SIDIS_FFs_PiP_HERMES13d_cosphi='../Calc_Grids_DSS/SIDIS_FFs/FF_PiP_HERMES13d_cosphi.csv'
SIDIS_FFs_PiP_HERMES13d_cos2phi='../Calc_Grids_DSS/SIDIS_FFs/FF_PiP_HERMES13d_cos2phi.csv'


SIDIS_FFs_PiM_HERMES13p_cosphi='../Calc_Grids_DSS/SIDIS_FFs/FF_PiM_HERMES13p_cosphi.csv'
SIDIS_FFs_PiM_HERMES13p_cos2phi='../Calc_Grids_DSS/SIDIS_FFs/FF_PiM_HERMES13p_cos2phi.csv'
SIDIS_FFs_PiM_HERMES13d_cosphi='../Calc_Grids_DSS/SIDIS_FFs/FF_PiM_HERMES13d_cosphi.csv'
SIDIS_FFs_PiM_HERMES13d_cos2phi='../Calc_Grids_DSS/SIDIS_FFs/FF_PiM_HERMES13d_cos2phi.csv'


SIDIS_FFs_Pi0_HERMES13p_cosphi='../Calc_Grids_DSS/SIDIS_FFs/FF_Pi0_HERMES13p_cosphi.csv'
SIDIS_FFs_Pi0_HERMES13p_cos2phi='../Calc_Grids_DSS/SIDIS_FFs/FF_Pi0_HERMES13p_cos2phi.csv'
SIDIS_FFs_Pi0_HERMES13d_cosphi='../Calc_Grids_DSS/SIDIS_FFs/FF_Pi0_HERMES13d_cosphi.csv'
SIDIS_FFs_Pi0_HERMES13d_cos2phi='../Calc_Grids_DSS/SIDIS_FFs/FF_Pi0_HERMES13d_cos2phi.csv'


SIDIS_FFs_KP_HERMES13p_cosphi='../Calc_Grids_DSS/SIDIS_FFs/FF_KP_HERMES13p_cosphi.csv'
SIDIS_FFs_KP_HERMES13p_cos2phi='../Calc_Grids_DSS/SIDIS_FFs/FF_KP_HERMES13p_cos2phi.csv'
SIDIS_FFs_KP_HERMES13d_cosphi='../Calc_Grids_DSS/SIDIS_FFs/FF_KP_HERMES13d_cosphi.csv'
SIDIS_FFs_KP_HERMES13d_cos2phi='../Calc_Grids_DSS/SIDIS_FFs/FF_KP_HERMES13d_cos2phi.csv'


SIDIS_FFs_KM_HERMES13p_cosphi='../Calc_Grids_DSS/SIDIS_FFs/FF_KM_HERMES13p_cosphi.csv'
SIDIS_FFs_KM_HERMES13p_cos2phi='../Calc_Grids_DSS/SIDIS_FFs/FF_KM_HERMES13p_cos2phi.csv'
SIDIS_FFs_KM_HERMES13d_cosphi='../Calc_Grids_DSS/SIDIS_FFs/FF_KM_HERMES13d_cosphi.csv'
SIDIS_FFs_KM_HERMES13d_cos2phi='../Calc_Grids_DSS/SIDIS_FFs/FF_KM_HERMES13d_cos2phi.csv'


SIDIS_FFs_HERMES13p_cosphi=(SIDIS_FFs_PiP_HERMES13p_cosphi,SIDIS_FFs_PiM_HERMES13p_cosphi,SIDIS_FFs_Pi0_HERMES13p_cosphi,SIDIS_FFs_KP_HERMES13p_cosphi,SIDIS_FFs_KM_HERMES13p_cosphi)
SIDIS_FFs_HERMES13p_cos2phi=(SIDIS_FFs_PiP_HERMES13p_cos2phi,SIDIS_FFs_PiM_HERMES13p_cos2phi,SIDIS_FFs_Pi0_HERMES13p_cos2phi,SIDIS_FFs_KP_HERMES13p_cos2phi,SIDIS_FFs_KM_HERMES13p_cos2phi)
SIDIS_FFs_HERMES13d_cosphi=(SIDIS_FFs_PiP_HERMES13d_cosphi,SIDIS_FFs_PiM_HERMES13d_cosphi,SIDIS_FFs_Pi0_HERMES13d_cosphi,SIDIS_FFs_KP_HERMES13d_cosphi,SIDIS_FFs_KM_HERMES13d_cosphi)
SIDIS_FFs_HERMES13d_cos2phi=(SIDIS_FFs_PiP_HERMES13d_cos2phi,SIDIS_FFs_PiM_HERMES13d_cos2phi,SIDIS_FFs_Pi0_HERMES13d_cos2phi,SIDIS_FFs_KP_HERMES13d_cos2phi,SIDIS_FFs_KM_HERMES13d_cos2phi)