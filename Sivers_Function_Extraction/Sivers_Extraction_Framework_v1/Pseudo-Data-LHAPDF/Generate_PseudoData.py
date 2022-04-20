import os
from SIDIS_Generator import *
from DY_Generator import *


OutputFolder='Set_3'
os.mkdir(OutputFolder)

Pseudo_SIDIS_HERMES2009.to_csv(str(OutputFolder)+'/Pseudo_SIDIS_HERMES2009.csv')
Pseudo_SIDIS_HERMES2020.to_csv(str(OutputFolder)+'/Pseudo_SIDIS_HERMES2020.csv')
Pseudo_SIDIS_COMPASS2009.to_csv(str(OutputFolder)+'/Pseudo_SIDIS_COMPASS2009.csv')
Pseudo_SIDIS_COMPASS2015.to_csv(str(OutputFolder)+'/Pseudo_SIDIS_COMPASS2015.csv')

Pseudo_DY_COMPASS2017.to_csv(str(OutputFolder)+'/Pseudo_DY_COMPASS2017.csv')