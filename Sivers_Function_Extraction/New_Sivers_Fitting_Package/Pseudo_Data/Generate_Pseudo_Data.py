import os
from DY_Sivers_Data_Generation import *
from SIDIS_Sivers_Data_Generation import *


OutputFolder='Set_1'
os.mkdir(OutputFolder)

Pseudo_SIDIS_HERMES2009.to_csv(str(OutputFolder)+'/Pseudo_SIDIS_HERMES2009.csv')
Pseudo_SIDIS_HERMES2020.to_csv(str(OutputFolder)+'/Pseudo_SIDIS_HERMES2020.csv')
Pseudo_SIDIS_COMPASS2009.to_csv(str(OutputFolder)+'/Pseudo_SIDIS_COMPASS2009.csv')
Pseudo_SIDIS_COMPASS2015.to_csv(str(OutputFolder)+'/Pseudo_SIDIS_COMPASS2015.csv')

Pseudo_DY_COMPASS2017.to_csv(str(OutputFolder)+'/Pseudo_DY_COMPASS2017.csv')