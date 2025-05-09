import numpy as np

## DY ###
#DY_DataFilesArray=np.array(['../Data/COMPASS_p_DY_2017.csv'])

## SIDIS ###
Dat1='../NN_SIDIS_Fit_Results/Result_SIDIS_HERMES2009.csv'
Dat2='../NN_SIDIS_Fit_Results/Result_SIDIS_HERMES2020.csv'
Dat3='../NN_SIDIS_Fit_Results/Result_SIDIS_COMPASS2009.csv'
Dat4='../NN_SIDIS_Fit_Results/Result_SIDIS_COMPASS2015.csv'
SIDIS_DataFilesArrayR=[Dat2,Dat4]


###########################################################################
#####################  DY PDFs #########################################
###########################################################################

#DY_PDFs_COMPASS_p_2017_x1='../Calc_Grids_DSS/DY_PDFs/PDFs_x1_COMPASS_p_DY_2017.csv'
#DY_PDFs_COMPASS_p_2017_x2='../Calc_Grids_DSS/DY_PDFs/PDFs_x2_COMPASS_p_DY_2017.csv'

###########################################################################
#####################  SIDIS PDFs #########################################
###########################################################################


SIDIS_PDFs_HERMES_p_2009='../Calc_Grids_DSS/SIDIS_PDFs/PDFs_HERMES_p_2009.csv'
SIDIS_PDFs_HERMES_p_2020='../Calc_Grids_DSS/SIDIS_PDFs/PDFs_HERMES_p_2020.csv'
SIDIS_PDFs_COMPASS_d_2009='../Calc_Grids_DSS/SIDIS_PDFs/PDFs_COMPASS_d_2009.csv'
SIDIS_PDFs_COMPASS_p_2015='../Calc_Grids_DSS/SIDIS_PDFs/PDFs_COMPASS_p_2015.csv'

SIDIS_PDFs_Array=(SIDIS_PDFs_HERMES_p_2009,SIDIS_PDFs_HERMES_p_2020,SIDIS_PDFs_COMPASS_d_2009,SIDIS_PDFs_COMPASS_p_2015)


###########################################################################
#####################  SIDIS FFs #########################################
###########################################################################


SIDIS_FFs_PiP_HERMES_p_2009='../Calc_Grids_DSS/SIDIS_FFs/FF_PiP_HERMES_p_2009.csv'
SIDIS_FFs_PiP_HERMES_p_2020='../Calc_Grids_DSS/SIDIS_FFs/FF_PiP_HERMES_p_2020.csv'
SIDIS_FFs_PiP_COMPASS_d_2009='../Calc_Grids_DSS/SIDIS_FFs/FF_PiP_COMPASS_d_2009.csv'
SIDIS_FFs_PiP_COMPASS_p_2015='../Calc_Grids_DSS/SIDIS_FFs/FF_PiP_COMPASS_p_2015.csv'


SIDIS_FFs_PiM_HERMES_p_2009='../Calc_Grids_DSS/SIDIS_FFs/FF_PiM_HERMES_p_2009.csv'
SIDIS_FFs_PiM_HERMES_p_2020='../Calc_Grids_DSS/SIDIS_FFs/FF_PiM_HERMES_p_2020.csv'
SIDIS_FFs_PiM_COMPASS_d_2009='../Calc_Grids_DSS/SIDIS_FFs/FF_PiM_COMPASS_d_2009.csv'
SIDIS_FFs_PiM_COMPASS_p_2015='../Calc_Grids_DSS/SIDIS_FFs/FF_PiM_COMPASS_p_2015.csv'


SIDIS_FFs_Pi0_HERMES_p_2009='../Calc_Grids_DSS/SIDIS_FFs/FF_Pi0_HERMES_p_2009.csv'
SIDIS_FFs_Pi0_HERMES_p_2020='../Calc_Grids_DSS/SIDIS_FFs/FF_Pi0_HERMES_p_2020.csv'
SIDIS_FFs_Pi0_COMPASS_d_2009='../Calc_Grids_DSS/SIDIS_FFs/FF_Pi0_COMPASS_d_2009.csv'
SIDIS_FFs_Pi0_COMPASS_p_2015='../Calc_Grids_DSS/SIDIS_FFs/FF_Pi0_COMPASS_p_2015.csv'


SIDIS_FFs_KP_HERMES_p_2009='../Calc_Grids_DSS/SIDIS_FFs/FF_KP_HERMES_p_2009.csv'
SIDIS_FFs_KP_HERMES_p_2020='../Calc_Grids_DSS/SIDIS_FFs/FF_KP_HERMES_p_2020.csv'
SIDIS_FFs_KP_COMPASS_d_2009='../Calc_Grids_DSS/SIDIS_FFs/FF_KP_COMPASS_d_2009.csv'
SIDIS_FFs_KP_COMPASS_p_2015='../Calc_Grids_DSS/SIDIS_FFs/FF_KP_COMPASS_p_2015.csv'


SIDIS_FFs_KM_HERMES_p_2009='../Calc_Grids_DSS/SIDIS_FFs/FF_KM_HERMES_p_2009.csv'
SIDIS_FFs_KM_HERMES_p_2020='../Calc_Grids_DSS/SIDIS_FFs/FF_KM_HERMES_p_2020.csv'
SIDIS_FFs_KM_COMPASS_d_2009='../Calc_Grids_DSS/SIDIS_FFs/FF_KM_COMPASS_d_2009.csv'
SIDIS_FFs_KM_COMPASS_p_2015='../Calc_Grids_DSS/SIDIS_FFs/FF_KM_COMPASS_p_2015.csv'


SIDIS_FFs_HERMES_p_2009=(SIDIS_FFs_PiP_HERMES_p_2009,SIDIS_FFs_PiM_HERMES_p_2009,SIDIS_FFs_Pi0_HERMES_p_2009,SIDIS_FFs_KP_HERMES_p_2009,SIDIS_FFs_KM_HERMES_p_2009)
SIDIS_FFs_HERMES_p_2020=(SIDIS_FFs_PiP_HERMES_p_2020,SIDIS_FFs_PiM_HERMES_p_2020,SIDIS_FFs_Pi0_HERMES_p_2020,SIDIS_FFs_KP_HERMES_p_2020,SIDIS_FFs_KM_HERMES_p_2020)
SIDIS_FFs_COMPASS_d_2009=(SIDIS_FFs_PiP_COMPASS_d_2009,SIDIS_FFs_PiM_COMPASS_d_2009,SIDIS_FFs_Pi0_COMPASS_d_2009,SIDIS_FFs_KP_COMPASS_d_2009,SIDIS_FFs_KM_COMPASS_d_2009)
SIDIS_FFs_COMPASS_p_2015=(SIDIS_FFs_PiP_COMPASS_p_2015,SIDIS_FFs_PiM_COMPASS_p_2015,SIDIS_FFs_Pi0_COMPASS_p_2015,SIDIS_FFs_KP_COMPASS_p_2015,SIDIS_FFs_KM_COMPASS_p_2015)

