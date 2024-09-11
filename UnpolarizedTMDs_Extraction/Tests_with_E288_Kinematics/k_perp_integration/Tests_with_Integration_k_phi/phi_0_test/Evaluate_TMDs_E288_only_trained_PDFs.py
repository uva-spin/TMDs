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


foldername = 'Evaluations_PDFs_DataPoints'
create_folders(foldername)


models_path = '/scratch/cee9hc/Unpolarized_TMD/E288/flavor_1/Phase_2/Test_05'
Models_folder = str(models_path)+'/'+'DNNmodels'
folders_array=os.listdir(Models_folder)
numreplicas=len(folders_array)
#numreplicas=3

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


k_upper = 6.0

def Generate_PDFs(model,x,qM):
    k_values = tf.linspace(0.0, k_upper, 100)
    dk = k_values[1] - k_values[0]
    product_sum = []  
    for k_val in k_values:
        concatenated_inputs = np.column_stack((x, k_val, qM))      
        #model_input = tf.keras.layers.Concatenate()(concatenated_inputs)
        #model_input = tf.keras.layers.Concatenate()([x, k_val, qM])
        model_output = model(concatenated_inputs)
        model_output = tf.multiply(k_val,model_output)
        product_sum.append(model_output)
    # Summing over all k values using tf.reduce_sum
    tmd_product_sum = tf.reduce_sum(product_sum, axis=0) * dk
    tmd_product_sum = np.array(tmd_product_sum) 
    return tmd_product_sum.flatten()[0]



def Generate_Comparison_Plots(df, num_replicas, output_name):
    x1vals = df['x1']
    x2vals = df['x2']
    pTvals = df['pT']
    QMvals = df['QM']

    Avals = df['A']

    fu_xA = df['fu_xA']
    fubar_xB = df['fubar_xB']

    fu_xB = df['fu_xB']
    fubar_xA = df['fubar_xA']

    Kvals = np.linspace(0,k_upper,len(x1vals))
    # Kvals = np.linspace(1,1,len(x1vals))
    #Kvals = np.linspace(0,0,len(x1vals))
    # pT_k_vals = pTvals - Kvals
    pT_k_vals = Kvals

    # true_values_1 = fu_xA * Skq(Kvals)  
    # true_values_2 = fubar_xB * Skqbar(pT_k_vals) 

    true_values_1 = fu_xA 
    true_values_2 = fubar_xB

    true_values_1_rev = fu_xB
    true_values_2_rev = fubar_xA


    predfuxA = []
    predfubarxB = []
    predfuxB = []
    predfubarxA = []    


    for i in range(num_replicas):
        t = modelsArray[i]
        print('**** Scanning model: '+str(i)+' ****')
        modnnu = t.get_layer('nnu')
        modnnubar = t.get_layer('nnubar')
        tempfuxA = []
        tempfubarxB = []
        tempfuxB = []
        tempfubarxA = []
        for j in range(0,len(x1vals)):
            tempxA = x1vals[j]
            tempxB = x2vals[j]
            tempQM = QMvals[j]
            temp_fu_xA = Generate_PDFs(modnnu,tempxA,tempQM)
            #print(temp_fu_xA)
            tempfuxA.append(temp_fu_xA)
            temp_fubar_xB = Generate_PDFs(modnnubar,tempxB,tempQM)
            #print(temp_fubar_xB)
            tempfubarxB.append(temp_fubar_xB)
            temp_fu_xB = Generate_PDFs(modnnu,tempxB,tempQM)
            tempfuxB.append(temp_fu_xB)
            temp_fubar_xA = Generate_PDFs(modnnubar,tempxA,tempQM)
            tempfubarxA.append(temp_fubar_xA)
        #######################################
        predfuxA.append(tempfuxA)
        predfubarxB.append(tempfubarxB)
        predfuxB.append(tempfuxB)
        predfubarxA.append(tempfubarxA)
    ###########################################
    tempfu = np.array(predfuxA)
    tempfu_mean = np.array(tempfu.mean(axis=0))
    #tempfu_mean=tempfu_mean[0]
    tempfu_err = np.array(tempfu.std(axis=0))

    tempfubar = np.array(predfubarxB)
    tempfubar_mean = np.array(tempfubar.mean(axis=0))
    #tempfubar_mean=tempfubar_mean[0]
    tempfubar_err = np.array(tempfubar.std(axis=0))


    tempfu_rev = np.array(predfuxB)
    tempfu_mean_rev = np.array(tempfu_rev.mean(axis=0))
    #tempfu_mean_rev=tempfu_mean_rev[0]
    tempfu_err_rev = np.array(tempfu_rev.std(axis=0))

    tempfubar_rev = np.array(predfubarxA)
    tempfubar_mean_rev = np.array(tempfubar_rev.mean(axis=0))
    #tempfubar_mean_rev=tempfubar_mean_rev[0]
    tempfubar_err_rev = np.array(tempfubar_rev.std(axis=0))

    #print(tempfu_mean)
    #print(tempfubar_mean)
    #print(len(true_values_2))
    #print(len(tempfubar_mean))

    plt.figure(1, figsize=(10, 6))
    plt.plot(x1vals, true_values_1, 'b.', label='True nnu')
    plt.plot(x1vals, tempfu_mean, 'r.', label='Predicted nnu')
    #plt.fill_between(x1vals, (tempfu_mean-tempfu_err).flatten(), (tempfu_mean+tempfu_err).flatten(), facecolor='r', alpha=0.3)
    plt.title('Comparison of True and Predicted nnu PDF Values')
    plt.xlabel('x1')
    plt.ylabel('nnu')
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig(str(foldername)+'/True_Pred_fx_q_'+str(output_name)+'.pdf')
    plt.close()

    plt.figure(2, figsize=(10, 6))
    plt.plot(x2vals, true_values_2, 'b.', label='True nnubar')
    plt.plot(x2vals, tempfubar_mean, 'r.', label='Predicted nnubar')
    #plt.fill_between(x2vals, (tempfubar_mean-tempfubar_err).flatten(), (tempfubar_mean+tempfubar_err).flatten(), facecolor='r', alpha=0.3)
    plt.title('Comparison of True and Predicted nnubar PDF Values')
    plt.xlabel('x2')
    plt.ylabel('nnubar')
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig(str(foldername)+'/True_Pred_fx_qbar_'+str(output_name)+'.pdf')
    plt.close()


    plt.figure(3, figsize=(10, 6))
    plt.plot(x2vals, true_values_1_rev, 'b.', label='True nnu')
    plt.plot(x2vals, tempfu_mean_rev, 'r.', label='Predicted nnu')
    #plt.fill_between(x2vals, (tempfu_mean_rev-tempfu_err_rev).flatten(), (tempfu_mean_rev+tempfu_err_rev).flatten(), facecolor='r', alpha=0.3)
    plt.title('Comparison of True and Predicted nnu PDF Values')
    plt.xlabel('x2')
    plt.ylabel('nnu')
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig(str(foldername)+'/True_Pred_fx_q_rev_'+str(output_name)+'.pdf')
    plt.close()

    plt.figure(4, figsize=(10, 6))
    plt.plot(x1vals, true_values_2_rev, 'b.', label='True nnubar')
    plt.plot(x1vals, tempfubar_mean_rev, 'r.', label='Predicted nnubar')
    #plt.fill_between(x1vals, (tempfubar_mean_rev-tempfubar_err_rev).flatten(), (tempfubar_mean_rev+tempfubar_err_rev).flatten(), facecolor='r', alpha=0.3)
    plt.title('Comparison of True and Predicted nnubar PDF Values')
    plt.xlabel('x1')
    plt.ylabel('nnubar')
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig(str(foldername)+'/True_Pred_fx_qbar_rev_'+str(output_name)+'.pdf')
    plt.close()


    # # 3D scatter plot
    # fig = plt.figure(5)
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x1vals, x2vals, Avals, c='r', marker='o', label='Actual')
    # # Plot the model predictions
    # ax.scatter(x1vals, x2vals, tempA_mean, c='b', marker='^', label='Predicted')
    # ax.set_xlabel('x1')
    # ax.set_ylabel('x2')
    # ax.set_zlabel('A')
    # ax.set_title('Actual vs Predicted')
    # ax.legend()
    # #plt.show()
    # plt.savefig('Evaluations_TMDs_DataPoints/Actual_vs_Predicted_CS_'+str(output_name)+'.pdf')
    # plt.close()

Generate_Comparison_Plots(df,numreplicas,'points')
