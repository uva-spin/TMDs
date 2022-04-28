import os
import pandas as pd
import numpy as np
import shutil

########## Here we need to define the paths ##########
Org_Data_path = 'Data/'
Generic_path = 'Data/'
HERMES2009 = 'HERMES_p_2009.csv'
HERMES2020 = 'HERMES_p_2020.csv'
COMPASS2009 = 'COMPASS_d_2009.csv'
COMPASS2015 = 'COMPASS_p_2015.csv'

Number_of_Replicas = 100

Replica_Output_Folder='Replicas'
#os.mkdir(Generic_path + Replica_Output_Folder)
if os.path.exists(Replica_Output_Folder):
    shutil.rmtree(Replica_Output_Folder)
os.makedirs(Replica_Output_Folder)

herm09 = pd.read_csv(Org_Data_path + HERMES2009).dropna(axis=0, how='all').dropna(axis=1, how='all')
herm20 = pd.read_csv(Org_Data_path + HERMES2020).dropna(axis=0, how='all').dropna(axis=1, how='all')
comp09 = pd.read_csv(Org_Data_path + COMPASS2009).dropna(axis=0, how='all').dropna(axis=1, how='all')
comp15 = pd.read_csv(Org_Data_path + COMPASS2015).dropna(axis=0, how='all').dropna(axis=1, how='all')

#### Combining all SIDIS data sets #####
df = pd.concat([herm09, herm20, comp09, comp15])
#df = pd.concat([herm20, comp09, comp15])
#df = pd.concat([herm20])
#df = pd.concat([herm20,comp15])

#### Creating a new column with 10% error ####
df["Siv_Rep_err"] = df["Siv"]*0.05


def GenerateReplicas(numReplicas):
    tempdf = df
    ycentral=np.array(df["Siv"])
    yerror=np.array(np.abs(df["Siv_Rep_err"]))
    for i in range(numReplicas):
        yrep = np.random.normal(ycentral,yerror)
        tempdf["Siv_Rep"] = yrep
        tempdf.to_csv(Replica_Output_Folder + '/Rep' + '_' + str(i) + '.csv')

GenerateReplicas(Number_of_Replicas)




