import tensorflow as tf
import pandas as pd
import numpy as np
#import lhapdf
import matplotlib.pyplot as plt
import os

def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")

folder_name ='Evaluations_TMDs_dfs'
create_folders(str(folder_name))


models_path = '/scratch/cee9hc/Unpolarized_TMD/E288/flavor_1/Arc_06_lrm6'
Models_folder = str(models_path)+'/'+'DNNmodels'
folders_array=os.listdir(Models_folder)
numreplicas=len(folders_array)
# numreplicas=3

def mse_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    return mse_loss


modelsArray = []
for i in range(numreplicas):
    testmodel = tf.keras.models.load_model(str(Models_folder)+'/' + str(folders_array[i]),custom_objects={'mse_loss': mse_loss})
    modelsArray.append(testmodel)

modelsArray = np.array(modelsArray)

# model = tf.keras.models.load_model('model.h5',custom_objects={'mse_loss': mse_loss})
# modnnu = model.get_layer('nnu')
# modnnubar = model.get_layer('nnubar')


########### Import pseudodata file 
df = pd.read_csv('E288_pseudo_data.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')


#### S(k) used for pseudo-data ####

def Skq(k):
    return np.exp(-4*k**2/(4*k**2 + 4))

def Skqbar(k):
    return np.exp(-4*k**2/(4*k**2 + 1))





def Generate_Comparison_Data(df, num_replicas, kv):
    x1vals = df['x1']
    x2vals = df['x2']
    pTvals = df['pT']
    QMvals = df['QM']

    Avals = df['A']

    fu_xA = df['fu_xA']
    fubar_xB = df['fubar_xB']

    fu_xB = df['fu_xB']
    fubar_xA = df['fubar_xA']

    Kvals = np.linspace(kv,kv,len(x1vals))
    #Kvals = np.linspace(0,0,len(x1vals))
    # pT_k_vals = pTvals - Kvals
    pT_k_vals = Kvals


    true_fxAk = fu_xA * Skq(Kvals)
    true_fbxBk = fubar_xB * Skqbar(pT_k_vals) 

    true_fxBk = fu_xB * Skq(Kvals)
    true_fbxAk = fubar_xA * Skqbar(pT_k_vals) 

    concat_inputs_xA = np.column_stack((x1vals,Kvals,QMvals))
    concat_inputs_xB = np.column_stack((x2vals,pT_k_vals,QMvals))


    tempfu = []
    tempfubar = []

    tempfu_rev = []
    tempfubar_rev = []

    sigma = []

    for i in range(num_replicas):
        t = modelsArray[i]
        modnnu = t.get_layer('nnu')
        modnnubar = t.get_layer('nnubar')
        temp_pred_fu = modnnu.predict(concat_inputs_xA)
        temp_pred_fubar = modnnubar.predict(concat_inputs_xB)
        tempfu.append(list(temp_pred_fu))
        tempfubar.append(list(temp_pred_fubar))

        temp_pred_fu_rev = modnnu.predict(concat_inputs_xA)
        temp_pred_fubar_rev = modnnubar.predict(concat_inputs_xB)
        tempfu_rev.append(list(temp_pred_fu_rev))
        tempfubar_rev.append(list(temp_pred_fubar_rev))

        # temp_sigma = t.predict([x1vals, x2vals,pTvals,QMvals])
        # sigma.append(temp_sigma[0]) # Index 0 because we only need the cross-section
     
    pred_fxAk = np.array(tempfu)
    pred_fxAk_mean = np.array(pred_fxAk.mean(axis=0))
    pred_fxAk_err = np.array(pred_fxAk.std(axis=0))

    pred_fbxBk = np.array(tempfubar)
    pred_fbxBk_mean = np.array(pred_fbxBk.mean(axis=0))
    pred_fbxBk_err = np.array(pred_fbxBk.std(axis=0))

    pred_fxBk = np.array(tempfu_rev)
    pred_fxBk_mean = np.array(pred_fxBk.mean(axis=0))
    pred_fxBk_err = np.array(pred_fxBk.std(axis=0))

    pred_fbxAk = np.array(tempfubar_rev)
    pred_fbxAk_mean = np.array(pred_fbxAk.mean(axis=0))
    pred_fbxAk_err = np.array(pred_fbxAk.std(axis=0))

    return Kvals, true_fxAk, pred_fxAk_mean, true_fbxBk, pred_fbxBk_mean, true_fxBk, pred_fxBk_mean, true_fbxAk, pred_fbxAk_mean



#import plotly.graph_objects as go

# def create_3D_Comparison_Plots(df,numreplicas,kv):
#     Kvals, true_fxAk, pred_fxAk, true_fbxBk, pred_fbxBk, true_fxBk, pred_fxBk_, true_fbxAk, pred_fbxAk = Generate_Comparison_Data(df,numreplicas,kv)
#     df1 = pd.DataFrame({'xA':df['x1'], 'k':Kvals, 'fxk':true_fxAk})
#     df2 = pd.DataFrame({'xA':df['x1'], 'k':Kvals, 'fxk':pred_fxAk.flatten()})
#     df3 = pd.DataFrame({'xB':df['x2'], 'k':Kvals, 'fxk':true_fbxBk})
#     df4 = pd.DataFrame({'xB':df['x2'], 'k':Kvals, 'fxk':pred_fbxBk.flatten()})
 
#     scatter_1 = go.Scatter3d(x=df1['xA'], y=df1['k'], z=df1['fxk'], mode='markers', name = 'fxA_true', marker=dict(size=2, color='blue'))
#     scatter_2 = go.Scatter3d(x=df2['xA'], y=df2['k'], z=df2['fxk'], mode='markers', name = 'fxA_pred', marker=dict(size=2, color='red'))

#     scatter_3 = go.Scatter3d(x=df3['xB'], y=df3['k'], z=df3['fxk'], mode='markers', name = 'fxA_true', marker=dict(size=2, color='blue'))
#     scatter_4 = go.Scatter3d(x=df4['xB'], y=df4['k'], z=df4['fxk'], mode='markers', name = 'fxA_pred', marker=dict(size=2, color='red'))   

#     fig1 = go.Figure(data=[scatter_1, scatter_2])
#     fig1.write_html('test1.html')

#     fig2 = go.Figure(data=[scatter_3, scatter_4])
#     fig2.write_html('test2.html')

# create_3D_Comparison_Plots(df,numreplicas,1)

# tempdf = create_3D_Comparison_Plots(df,numreplicas,1)
# print(tempdf)


def create_3D_Comparison_dfs(df,numreplicas,kv):
    Kvals, true_fxAk, pred_fxAk, true_fbxBk, pred_fbxBk, true_fxBk, pred_fxBk, true_fbxAk, pred_fbxAk = Generate_Comparison_Data(df,numreplicas,kv)
    df_fxAk_true = pd.DataFrame({'xA':df['x1'], 'k':Kvals, 'fxk':true_fxAk})
    df_fxAk_pred = pd.DataFrame({'xA':df['x1'], 'k':Kvals, 'fxk':pred_fxAk.flatten()})
    df_fbxBk_true = pd.DataFrame({'xB':df['x2'], 'k':Kvals, 'fxk':true_fbxBk})
    df_fbxBk_pred = pd.DataFrame({'xB':df['x2'], 'k':Kvals, 'fxk':pred_fbxBk.flatten()})
    df_fxBk_true = pd.DataFrame({'xB':df['x2'], 'k':Kvals, 'fxk':true_fxBk})
    df_fxBk_pred = pd.DataFrame({'xB':df['x2'], 'k':Kvals, 'fxk':pred_fxBk.flatten()})
    df_fbxAk_true = pd.DataFrame({'xA':df['x1'], 'k':Kvals, 'fxk':true_fbxAk})
    df_fbxAk_pred = pd.DataFrame({'xA':df['x1'], 'k':Kvals, 'fxk':pred_fbxAk.flatten()})
    return df_fxAk_true, df_fxAk_pred, df_fbxBk_true, df_fbxBk_pred, df_fxBk_true, df_fxBk_pred, df_fbxAk_true, df_fbxAk_pred
 

dfk00 = create_3D_Comparison_dfs(df,numreplicas,0.0)
dfk05 = create_3D_Comparison_dfs(df,numreplicas,0.5)
dfk10 = create_3D_Comparison_dfs(df,numreplicas,1.0)
dfk15 = create_3D_Comparison_dfs(df,numreplicas,1.5)
dfk20 = create_3D_Comparison_dfs(df,numreplicas,2.0)


#### .csv files for k=0 #####
dfk00[0].to_csv(str(folder_name)+'/k_00_fxAk_true.csv')
dfk00[1].to_csv(str(folder_name)+'/k_00_fxAk_pred.csv')
dfk00[2].to_csv(str(folder_name)+'/k_00_fbxBk_true.csv')
dfk00[3].to_csv(str(folder_name)+'/k_00_fbxBk_pred.csv')
dfk00[4].to_csv(str(folder_name)+'/k_00_fxBk_true.csv')
dfk00[5].to_csv(str(folder_name)+'/k_00_fxBk_pred.csv')
dfk00[6].to_csv(str(folder_name)+'/k_00_fbxAk_true.csv')
dfk00[7].to_csv(str(folder_name)+'/k_00_fbxAk_pred.csv')


#### .csv files for k=0.5 #####
dfk05[0].to_csv(str(folder_name)+'/k_05_fxAk_true.csv')
dfk05[1].to_csv(str(folder_name)+'/k_05_fxAk_pred.csv')
dfk05[2].to_csv(str(folder_name)+'/k_05_fbxBk_true.csv')
dfk05[3].to_csv(str(folder_name)+'/k_05_fbxBk_pred.csv')
dfk05[4].to_csv(str(folder_name)+'/k_05_fxBk_true.csv')
dfk05[5].to_csv(str(folder_name)+'/k_05_fxBk_pred.csv')
dfk05[6].to_csv(str(folder_name)+'/k_05_fbxAk_true.csv')
dfk05[7].to_csv(str(folder_name)+'/k_05_fbxAk_pred.csv')


#### .csv files for k=1.0 #####
dfk10[0].to_csv(str(folder_name)+'/k_10_fxAk_true.csv')
dfk10[1].to_csv(str(folder_name)+'/k_10_fxAk_pred.csv')
dfk10[2].to_csv(str(folder_name)+'/k_10_fbxBk_true.csv')
dfk10[3].to_csv(str(folder_name)+'/k_10_fbxBk_pred.csv')
dfk10[4].to_csv(str(folder_name)+'/k_10_fxBk_true.csv')
dfk10[5].to_csv(str(folder_name)+'/k_10_fxBk_pred.csv')
dfk10[6].to_csv(str(folder_name)+'/k_10_fbxAk_true.csv')
dfk10[7].to_csv(str(folder_name)+'/k_10_fbxAk_pred.csv')


#### .csv files for k=1.5 #####
dfk15[0].to_csv(str(folder_name)+'/k_15_fxAk_true.csv')
dfk15[1].to_csv(str(folder_name)+'/k_15_fxAk_pred.csv')
dfk15[2].to_csv(str(folder_name)+'/k_15_fbxBk_true.csv')
dfk15[3].to_csv(str(folder_name)+'/k_15_fbxBk_pred.csv')
dfk15[4].to_csv(str(folder_name)+'/k_15_fxBk_true.csv')
dfk15[5].to_csv(str(folder_name)+'/k_15_fxBk_pred.csv')
dfk15[6].to_csv(str(folder_name)+'/k_15_fbxAk_true.csv')
dfk15[7].to_csv(str(folder_name)+'/k_15_fbxAk_pred.csv')


#### .csv files for k=2.0 #####
dfk20[0].to_csv(str(folder_name)+'/k_20_fxAk_true.csv')
dfk20[1].to_csv(str(folder_name)+'/k_20_fxAk_pred.csv')
dfk20[2].to_csv(str(folder_name)+'/k_20_fbxBk_true.csv')
dfk20[3].to_csv(str(folder_name)+'/k_20_fbxBk_pred.csv')
dfk20[4].to_csv(str(folder_name)+'/k_20_fxBk_true.csv')
dfk20[5].to_csv(str(folder_name)+'/k_20_fxBk_pred.csv')
dfk20[6].to_csv(str(folder_name)+'/k_20_fbxAk_true.csv')
dfk20[7].to_csv(str(folder_name)+'/k_20_fbxAk_pred.csv')