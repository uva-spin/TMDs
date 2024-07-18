import tensorflow as tf
import pandas as pd
import numpy as np
#import lhapdf
import matplotlib.pyplot as plt
import os
from tensorflow_addons.activations import tanhshrink
tf.keras.utils.get_custom_objects().update({'tanhshrink': tanhshrink})

def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")


create_folders('Evaluations_TMDs')


Models_folder = 'DNNmodels'
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



#### S(k) for pseudo-data ####

# def Skq(k,Q):
#     return np.exp(-4*k**2/(4*k**2 + 4))*(1/10)*np.log(100/Q)

# def Skqbar(k,Q):
#     return np.exp(-4*k**2/(4*k**2 + 1))*(1/10)*np.log(100/Q)

#### S(k) for pseudo-data ####

def Sku(k,Q):
    return np.exp(-4*k**2/(4*k**2 + 4))*(1/10)*np.log(100/Q)

def Skubar(k,Q):
    return np.exp(-4*k**2/(4*k**2 + 1))*(1/10)*np.log(100/Q)

def Skd(k,Q):
    return np.exp(-4*k**2/(4*k**2 + 6))*(1/10)*np.log(100/Q)

def Skdbar(k,Q):
    return np.exp(-4*k**2/(4*k**2 + 2))*(1/10)*np.log(100/Q)

def Sks(k,Q):
    return np.exp(-4*k**2/(4*k**2 + 3))*(1/10)*np.log(100/Q)

def Sksbar(k,Q):
    return np.exp(-4*k**2/(4*k**2 + 0.5))*(1/10)*np.log(100/Q)



def Generate_Comparison_Plots(df, num_replicas, output_name):
    x1vals = df['x1']
    x2vals = df['x2']
    pTvals = df['pT']
    QMvals = df['QM']

    fu_xA = df['fu_xA']
    fubar_xB = df['fubar_xB']
    fd_xA = df['fd_xA']
    fdbar_xB = df['fdbar_xB']
    fs_xA = df['fs_xA']
    fsbar_xB = df['fsbar_xB']

    Kvals = np.linspace(0.1,2,len(x1vals))
    pT_k_vals = pTvals - Kvals

    true_values_u = fu_xA * Sku(Kvals,QMvals)  
    true_values_ubar = fubar_xB * Skubar(pT_k_vals,QMvals) 
    true_values_d = fd_xA * Sku(Kvals,QMvals)  
    true_values_dbar = fdbar_xB * Skubar(pT_k_vals,QMvals) 
    true_values_s = fs_xA * Sku(Kvals,QMvals)  
    true_values_sbar = fsbar_xB * Skubar(pT_k_vals,QMvals) 

    concatenated_inputs_1 = np.column_stack((x1vals,Kvals,QMvals))
    concatenated_inputs_2 = np.column_stack((x2vals,pT_k_vals,QMvals))

    # predicted_values_1 = modnnu.predict(concatenated_inputs_1)
    # predicted_values_2 = modnnubar.predict(concatenated_inputs_2)

    tempfu = []
    tempfubar = []
    tempfd = []
    tempfdbar = []
    tempfs = []
    tempfsbar = []

    create_folders('Evaluations_TMDs/'+str(output_name))

    for i in range(num_replicas):
        t = modelsArray[i]
        modnnu = t.get_layer('nnu')
        modnnubar = t.get_layer('nnubar')
        # modnnd = t.get_layer('nnd')
        # modnndbar = t.get_layer('nndbar')
        # modnns = t.get_layer('nns')
        # modnnsbar = t.get_layer('nnsbar')
        #################
        temp_pred_fu = modnnu.predict(concatenated_inputs_1)
        temp_pred_fubar = modnnubar.predict(concatenated_inputs_2)
        # temp_pred_fd = modnnd.predict(concatenated_inputs_1)
        # temp_pred_fdbar = modnndbar.predict(concatenated_inputs_2)
        # temp_pred_fs = modnns.predict(concatenated_inputs_1)
        # temp_pred_fsbar = modnnsbar.predict(concatenated_inputs_2)
        tempfu.append(list(temp_pred_fu))
        tempfubar.append(list(temp_pred_fubar))
        # tempfd.append(list(temp_pred_fd))
        # tempfdbar.append(list(temp_pred_fdbar))
        # tempfs.append(list(temp_pred_fs))
        # tempfsbar.append(list(temp_pred_fsbar))
    
    # tempfu = np.array(tempfu)
    # tempfu = np.array(tempfu.mean(axis=0))
    # tempfu_err = np.array(tempfu.std(axis=0))    

    tempfu = np.array(tempfu)
    tempfu_mean = np.array(tempfu.mean(axis=0))
    tempfu_err = np.array(tempfu.std(axis=0))

    tempfubar = np.array(tempfubar)
    tempfubar_mean = np.array(tempfubar.mean(axis=0))
    tempfubar_err = np.array(tempfubar.std(axis=0))

    # tempfd = np.array(tempfd)
    # tempfd_mean = np.array(tempfd.mean(axis=0))
    # tempfd_err = np.array(tempfd.std(axis=0))

    # tempfdbar = np.array(tempfdbar)
    # tempfdbar_mean = np.array(tempfdbar.mean(axis=0))
    # tempfdbar_err = np.array(tempfdbar.std(axis=0))

    # tempfs = np.array(tempfs)
    # tempfs_mean = np.array(tempfs.mean(axis=0))
    # tempfs_err = np.array(tempfs.std(axis=0))

    # tempfsbar = np.array(tempfsbar)
    # tempfsbar_mean = np.array(tempfsbar.mean(axis=0))
    # tempfsbar_err = np.array(tempfsbar.std(axis=0))

    # return (tempfu_mean-tempfu_err).flatten()



    plt.figure(1, figsize=(10, 6))
    plt.plot(x1vals, true_values_u, 'b.', label='True nnu')
    plt.plot(x1vals, tempfu_mean, 'r.', label='Predicted nnu')
    #plt.fill_between(x1vals, (tempfu_mean-tempfu_err).flatten(), (tempfu_mean+tempfu_err).flatten(), facecolor='r', alpha=0.3)
    plt.title('Comparison of True and Predicted nnu Values')
    plt.xlabel('x1')
    plt.ylabel('nnu')
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig('Evaluations_TMDs/'+str(output_name)+'/True_Pred_fxk_u.pdf')
    plt.close()

    plt.figure(2, figsize=(10, 6))
    plt.plot(x2vals, true_values_ubar, 'b.', label='True nnubar')
    plt.plot(x2vals, tempfubar_mean, 'r.', label='Predicted nnubar')
    #plt.fill_between(x2vals, (tempfubar_mean-tempfubar_err).flatten(), (tempfubar_mean+tempfubar_err).flatten(), facecolor='r', alpha=0.3)
    plt.title('Comparison of True and Predicted nnubar Values')
    plt.xlabel('x2')
    plt.ylabel('nnubar')
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig('Evaluations_TMDs/'+str(output_name)+'/True_Pred_fxk_ubar.pdf')
    plt.close()


    # plt.figure(3, figsize=(10, 6))
    # plt.plot(x1vals, true_values_d, 'b.', label='True nnd')
    # plt.plot(x1vals, tempfd_mean, 'r.', label='Predicted nnd')
    # #plt.fill_between(x1vals, (tempfd_mean-tempfd_err).flatten(), (tempfd_mean+tempfd_err).flatten(), facecolor='r', alpha=0.3)
    # plt.title('Comparison of True and Predicted nnd Values')
    # plt.xlabel('x1')
    # plt.ylabel('nnd')
    # plt.legend()
    # plt.grid(True)
    # #plt.show()
    # plt.savefig('Evaluations_TMDs/'+str(output_name)+'/True_Pred_fxk_d.pdf')
    # plt.close()

    # plt.figure(4, figsize=(10, 6))
    # plt.plot(x2vals, true_values_dbar, 'b.', label='True nndbar')
    # plt.plot(x2vals, tempfdbar_mean, 'r.', label='Predicted nndbar')
    # #plt.fill_between(x2vals, (tempfdbar_mean-tempfdbar_err).flatten(), (tempfdbar_mean+tempfdbar_err).flatten(), facecolor='r', alpha=0.3)
    # plt.title('Comparison of True and Predicted nndbar Values')
    # plt.xlabel('x2')
    # plt.ylabel('nndbar')
    # plt.legend()
    # plt.grid(True)
    # #plt.show()
    # plt.savefig('Evaluations_TMDs/'+str(output_name)+'/True_Pred_fxk_dbar.pdf')
    # plt.close()


    # plt.figure(5, figsize=(10, 6))
    # plt.plot(x1vals, true_values_s, 'b.', label='True nns')
    # plt.plot(x1vals, tempfs_mean, 'r.', label='Predicted nns')
    # #plt.fill_between(x1vals, (tempfs_mean-tempfs_err).flatten(), (tempfs_mean+tempfs_err).flatten(), facecolor='r', alpha=0.3)
    # plt.title('Comparison of True and Predicted nns Values')
    # plt.xlabel('x1')
    # plt.ylabel('nns')
    # plt.legend()
    # plt.grid(True)
    # #plt.show()
    # plt.savefig('Evaluations_TMDs/'+str(output_name)+'/True_Pred_fxk_s.pdf')
    # plt.close()

    # plt.figure(6, figsize=(10, 6))
    # plt.plot(x2vals, true_values_sbar, 'b.', label='True nnsbar')
    # plt.plot(x2vals, tempfsbar_mean, 'r.', label='Predicted nnsbar')
    # #plt.fill_between(x2vals, (tempfsbar_mean-tempfsbar_err).flatten(), (tempfsbar_mean+tempfsbar_err).flatten(), facecolor='r', alpha=0.3)
    # plt.title('Comparison of True and Predicted nnsbar Values')
    # plt.xlabel('x2')
    # plt.ylabel('nnsbar')
    # plt.legend()
    # plt.grid(True)
    # #plt.show()
    # plt.savefig('Evaluations_TMDs/'+str(output_name)+'/True_Pred_fxk_sbar.pdf')
    # plt.close()


# Generate_Comparison_Plots(pdfs0,numreplicas,'set_0')
# Generate_Comparison_Plots(pdfs1,numreplicas,'set_1')
# Generate_Comparison_Plots(pdfs2,numreplicas,'set_2')
# Generate_Comparison_Plots(pdfs3,numreplicas,'set_3')
# Generate_Comparison_Plots(pdfs4,numreplicas,'set_4')
# Generate_Comparison_Plots(pdfs5,numreplicas,'set_5')
# Generate_Comparison_Plots(pdfs6,numreplicas,'set_6')
# Generate_Comparison_Plots(pdfs7,numreplicas,'set_7')
# Generate_Comparison_Plots(pdfs8,numreplicas,'set_8')
# Generate_Comparison_Plots(pdfs9,numreplicas,'set_9')

Generate_Comparison_Plots(df,numreplicas,'overall')