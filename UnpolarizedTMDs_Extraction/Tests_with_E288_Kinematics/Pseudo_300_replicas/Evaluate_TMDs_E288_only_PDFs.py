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


create_folders('Evaluations_TMDs_with_PDFs')


Models_folder = 'Previous_Models/DNNmodels'
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


## Load the pdf grids
pdfs0 = pd.read_csv('Eval_PDFs/Eval_pdfs_0.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
pdfs1 = pd.read_csv('Eval_PDFs/Eval_pdfs_1.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
pdfs2 = pd.read_csv('Eval_PDFs/Eval_pdfs_2.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
pdfs3 = pd.read_csv('Eval_PDFs/Eval_pdfs_3.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
pdfs4 = pd.read_csv('Eval_PDFs/Eval_pdfs_4.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
pdfs5 = pd.read_csv('Eval_PDFs/Eval_pdfs_5.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
pdfs6 = pd.read_csv('Eval_PDFs/Eval_pdfs_6.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
pdfs7 = pd.read_csv('Eval_PDFs/Eval_pdfs_7.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
pdfs8 = pd.read_csv('Eval_PDFs/Eval_pdfs_8.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
pdfs9 = pd.read_csv('Eval_PDFs/Eval_pdfs_9.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')

#### S(k) used for pseudo-data ####

def Skq(k):
    return np.exp(-4*k**2/(4*k**2 + 4))

def Skqbar(k):
    return np.exp(-4*k**2/(4*k**2 + 1))




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


def Generate_Comparison_Plots(df, num_replicas, output_name):
    x1vals = df['xA']
    x2vals = df['xB']
    pTvals = df['PT']
    QMvals = df['QM']

    fu_xA = df['fu_xA']
    fubar_xB = df['fubar_xB']
    #Kvals = np.linspace(0.1,2,len(x1vals))
    Kvals = np.linspace(0,0,len(x1vals))
    pT_k_vals = pTvals - Kvals

    # true_values_1 = fu_xA * Skq(Kvals)  
    # true_values_2 = fubar_xB * Skqbar(pT_k_vals) 

    true_values_1 = fu_xA 
    true_values_2 = fubar_xB

    concatenated_inputs_1 = np.column_stack((x1vals,Kvals,QMvals))
    concatenated_inputs_2 = np.column_stack((x2vals,pT_k_vals,QMvals))

    # predicted_values_1 = modnnu.predict(concatenated_inputs_1)
    # predicted_values_2 = modnnubar.predict(concatenated_inputs_2)

    tempfu = []
    tempfubar = []

    for i in range(num_replicas):
        t = modelsArray[i]
        modnnu = t.get_layer('nnu')
        modnnubar = t.get_layer('nnubar')
        temp_pred_fu = modnnu.predict(concatenated_inputs_1)
        temp_pred_fubar = modnnubar.predict(concatenated_inputs_2)
        tempfu.append(list(temp_pred_fu))
        tempfubar.append(list(temp_pred_fubar))
    
    # tempfu = np.array(tempfu)
    # tempfu = np.array(tempfu.mean(axis=0))
    # tempfu_err = np.array(tempfu.std(axis=0))    

    tempfu = np.array(tempfu)
    tempfu_mean = np.array(tempfu.mean(axis=0))
    tempfu_err = np.array(tempfu.std(axis=0))

    tempfubar = np.array(tempfubar)
    tempfubar_mean = np.array(tempfubar.mean(axis=0))
    tempfubar_err = np.array(tempfubar.std(axis=0))

    # return (tempfu_mean-tempfu_err).flatten()


    plt.figure(1, figsize=(10, 6))
    plt.plot(x1vals, true_values_1, label='True nnu', linestyle='--')
    plt.plot(x1vals, tempfu_mean, 'r', label='Predicted nnu')
    plt.fill_between(x1vals, (tempfu_mean-tempfu_err).flatten(), (tempfu_mean+tempfu_err).flatten(), facecolor='r', alpha=0.3)
    plt.title('Comparison of True and Predicted nnu PDF Values')
    plt.xlabel('x1')
    plt.ylabel('nnu')
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig('Evaluations_TMDs_with_PDFs/True_Pred_fx_q_'+str(output_name)+'.pdf')
    plt.close()

    plt.figure(2, figsize=(10, 6))
    plt.plot(x2vals, true_values_2, label='True nnubar', linestyle='--')
    plt.plot(x2vals, tempfubar_mean, 'r', label='Predicted nnubar')
    plt.fill_between(x2vals, (tempfubar_mean-tempfubar_err).flatten(), (tempfubar_mean+tempfubar_err).flatten(), facecolor='r', alpha=0.3)
    plt.title('Comparison of True and Predicted nnubar PDF Values')
    plt.xlabel('x2')
    plt.ylabel('nnubar')
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig('Evaluations_TMDs_with_PDFs/True_Pred_fx_qbar_'+str(output_name)+'.pdf')
    plt.close()



# def Generate_Comparison_Plots(df,output_name):
#     x1vals = df['xA']
#     x2vals = df['xB']
#     pTvals = df['PT']
#     QMvals = df['QM']

#     fu_xA = df['fu_xA']
#     fubar_xB = df['fubar_xB']
#     Kvals = np.linspace(0.1,2,len(x1vals))
#     pT_k_vals = pTvals - Kvals

#     true_values_1 = fu_xA * Skq(Kvals)  
#     true_values_2 = fubar_xB * Skqbar(pT_k_vals) 

#     concatenated_inputs_1 = np.column_stack((x1vals,Kvals,QMvals))
#     concatenated_inputs_2 = np.column_stack((x2vals,pT_k_vals,QMvals))

#     predicted_values_1 = modnnu.predict(concatenated_inputs_1)
#     predicted_values_2 = modnnubar.predict(concatenated_inputs_2)

#     plt.figure(1, figsize=(10, 6))
#     plt.plot(x1vals, true_values_1, label='True nnu', linestyle='--')
#     plt.plot(x1vals, predicted_values_1, label='Predicted nnu')
#     plt.title('Comparison of True and Predicted nnu Values')
#     plt.xlabel('x1')
#     plt.ylabel('nnu')
#     plt.legend()
#     plt.grid(True)
#     #plt.show()
#     plt.savefig('Evaluations_TMDs/True_Pred_fxk_q_'+str(output_name)+'.pdf')
#     plt.close()

#     plt.figure(2, figsize=(10, 6))
#     plt.plot(x2vals, true_values_2, label='True nnubar', linestyle='--')
#     plt.plot(x2vals, predicted_values_2, label='Predicted nnubar')
#     plt.title('Comparison of True and Predicted nnubar Values')
#     plt.xlabel('x2')
#     plt.ylabel('nnubar')
#     plt.legend()
#     plt.grid(True)
#     #plt.show()
#     plt.savefig('Evaluations_TMDs/True_Pred_fxk_qbar_'+str(output_name)+'.pdf')
#     plt.close()


Generate_Comparison_Plots(pdfs0,numreplicas,'set_0')
Generate_Comparison_Plots(pdfs1,numreplicas,'set_1')
Generate_Comparison_Plots(pdfs2,numreplicas,'set_2')
Generate_Comparison_Plots(pdfs3,numreplicas,'set_3')
Generate_Comparison_Plots(pdfs4,numreplicas,'set_4')
Generate_Comparison_Plots(pdfs5,numreplicas,'set_5')
Generate_Comparison_Plots(pdfs6,numreplicas,'set_6')
Generate_Comparison_Plots(pdfs7,numreplicas,'set_7')
Generate_Comparison_Plots(pdfs8,numreplicas,'set_8')
Generate_Comparison_Plots(pdfs9,numreplicas,'set_9')

# x1vals = pdfs1['xA']
# x2vals = pdfs1['xB']
# pTvals = pdfs1['PT']
# QMvals = pdfs1['QM']
# kk_values_loss = np.array(np.linspace(0,0,len(x1vals)))

# fu_xA = pdfs1['fu_xA']
# fubar_xB = pdfs1['fubar_xB']
# Kvals = np.linspace(0.1,2,len(x1vals))
# pT_k_vals = pTvals - Kvals

# true_values_1 = fu_xA * Skq(Kvals)  
# true_values_2 = fubar_xB * Skqbar(pT_k_vals) 

# concatenated_inputs_1 = np.column_stack((x1vals,Kvals,QMvals))
# concatenated_inputs_2 = np.column_stack((x2vals,pT_k_vals,QMvals))

# predicted_values_1 = modnnu.predict(concatenated_inputs_1)
# predicted_values_2 = modnnubar.predict(concatenated_inputs_2)


# plt.figure(3, figsize=(10, 6))
# plt.plot(x1vals, true_values_1, label='True nnu', linestyle='--')
# plt.plot(x1vals, predicted_values_1, label='Predicted nnu')
# plt.title('Comparison of True and Predicted nnu Values')
# plt.xlabel('x1')
# plt.ylabel('nnu')
# plt.legend()
# plt.grid(True)
# #plt.show()
# plt.savefig('True_Pred_fxk_q.pdf')

# plt.figure(4, figsize=(10, 6))
# plt.plot(x2vals, true_values_2, label='True nnubar', linestyle='--')
# plt.plot(x2vals, predicted_values_2, label='Predicted nnubar')
# plt.title('Comparison of True and Predicted nnubar Values')
# plt.xlabel('x2')
# plt.ylabel('nnubar')
# plt.legend()
# plt.grid(True)
# #plt.show()
# plt.savefig('True_Pred_fxk_qbar.pdf')
