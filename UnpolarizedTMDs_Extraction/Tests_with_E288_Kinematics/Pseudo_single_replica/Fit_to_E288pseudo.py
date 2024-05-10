import tensorflow as tf
import pandas as pd
import numpy as np
#import lhapdf
import matplotlib.pyplot as plt



########### Import pseudodata file 
df = pd.read_csv('E288_pseudo_data.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
x1Vals = np.array(df['x1'])
x2Vals = np.array(df['x2'])
pTVals = np.array(df['pT'])
QMvals = np.array(df['QM'])
Kvals = np.linspace(0.1,2,len(x1Vals))
pT_k_vals = pTVals - Kvals
fu = df['fu_xA']
fubar = df['fubar_xB']


################ Defining the DNN model ####################
Hidden_Layers=7
Nodes_per_HL=500
Learning_Rate = 0.0001
L1_reg = 10**(-12)
EPOCHS = 300




def create_nn_model(name):
    inp = tf.keras.Input(shape=(3))
    initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=42)
    x = tf.keras.layers.Dense(240, activation='tanh', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(inp)
    x1 = tf.keras.layers.Dense(32, activation='relu', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x)
    x2 = tf.keras.layers.Dense(32, activation='relu', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x1)
    x3 = tf.keras.layers.Dense(32, activation='relu', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x2)
    x4 = tf.keras.layers.Dense(32, activation='relu', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x3)
    x5 = tf.keras.layers.Dense(32, activation='relu', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x4)
    x6 = tf.keras.layers.Dense(32, activation='relu', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x5)
    nnout = tf.keras.layers.Dense(1, activation='relu', kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x6)
    mod = tf.keras.Model(inp, nnout, name=name)
    return mod



def createModel_DY():
    x1 = tf.keras.Input(shape=(1), name='x1')
    x2 = tf.keras.Input(shape=(1), name='x2')
    qT = tf.keras.Input(shape=(1), name='qT')
    qM = tf.keras.Input(shape=(1), name='QM')


    modnnu = create_nn_model('nnu')
    modnnubar = create_nn_model('nnubar')

    pdf_k_val = 0    
    nnu_pdf_input = tf.keras.layers.Concatenate()([x1, qT*0 + pdf_k_val, qM])
    nnubar_pdf_input = tf.keras.layers.Concatenate()([x2, qT*0 - pdf_k_val, qM])

    modnnu_pdf_eval = modnnu(nnu_pdf_input)
    modnnubar_pdf_eval = modnnubar(nnubar_pdf_input)

    k_values = tf.linspace(0.1, 2.0, 100)
    dk = k_values[1] - k_values[0]

    tmd1_list, tmd2_list = [], []
    product_list = []  # List to store TMD values for each k value
    for k_val in k_values:      
        nnu_input = tf.keras.layers.Concatenate()([x1, qT*0 + k_val, qM])
        nnubar_input = tf.keras.layers.Concatenate()([x2, qT - k_val, qM])

        nnu_x1 = modnnu(nnu_input)
        nnubar_x2 = modnnubar(nnubar_input)

        nnu_x1_rev = modnnu(nnubar_input)
        nnubar_x2_rev = modnnubar(nnu_input)

        product_1 = tf.multiply(nnu_x1, nnubar_x2)
        product_2 = tf.multiply(nnu_x1_rev, nnubar_x2_rev)

        result = tf.add(product_1,product_2)
        product_list.append(result)

    # Summing over all k values using tf.reduce_sum
    tmd_product_sum = tf.reduce_sum(product_list, axis=0) * dk 

    return tf.keras.Model([x1, x2, qT, qM], [tmd_product_sum, modnnu_pdf_eval, modnnubar_pdf_eval])

model = createModel_DY()

def mse_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    return mse_loss


# def custom_loss(y_true, y_pred):
#     return mse_loss(y_true, y_pred) 


model.compile(optimizer='adam', loss=mse_loss)
# history = model.fit([df['x1'],df['x2'],df['pT']], df['A'], epochs=EPOCHS, batch_size=32, verbose=2)
history = model.fit([df['x1'],df['x2'],df['pT'],df['QM']], [df['A'], fu, fubar], epochs=EPOCHS, batch_size=32, verbose=2)
model.save('model.h5', save_format='h5')


predictions = model.predict([df['x1'],df['x2'],df['pT'],df['QM']])[0]

# 3D scatter plot
fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1Vals, pTVals, df['A'], c='r', marker='o', label='Actual')
ax.scatter(x1Vals, pTVals, predictions, c='b', marker='^', label='Predicted')
ax.set_xlabel('x1')
ax.set_ylabel('pT')
ax.set_zlabel('A')
ax.set_title('Actual vs Predicted')
ax.legend()
#plt.show()
plt.savefig('Actual_vs_Predicted_Integral.pdf')


# modnnu = model.get_layer('nnu')
# modnnubar = model.get_layer('nnubar')


# #### S(k) used for pseudo-data ####

# def Skq(k):
#     return np.exp(-4*k**2/(4*k**2 + 4))

# def Skqbar(k):
#     return np.exp(-4*k**2/(4*k**2 + 1))


# ### Here for the evaluation use different arrays for x and k ###


# x1vals = np.linspace(0.014,0.035,100)
# x2vals = np.linspace(0.014,0.035,100)
# pTvals = np.linspace(0.1,3,100)
# QMvals = np.linspace(5.5,5.5,100) ## Here we fixed the QM dependence as Q2
# kk_values_loss = np.array(np.linspace(0,0,len(x1vals)))

# true_values_1 = fu * Skq(Kvals)  
# true_values_2 = fubar * Skqbar(pT_k_vals) 

# concatenated_inputs_1 = np.column_stack((x1Vals,Kvals,QMvals))
# concatenated_inputs_2 = np.column_stack((x2Vals,pT_k_vals,QMvals))

# predicted_values_1 = modnnu.predict(concatenated_inputs_1)
# predicted_values_2 = modnnubar.predict(concatenated_inputs_2)

# # predicted_values_1 = model.predict([df['x1'],df['x2'],df['pT']])[1]
# # predicted_values_2 = model.predict([df['x1'],df['x2'],df['pT']])[2]


# plt.figure(3, figsize=(10, 6))
# plt.plot(x1Vals, true_values_1, label='True nnu', linestyle='--')
# plt.plot(x1Vals, predicted_values_1, label='Predicted nnu')
# plt.title('Comparison of True and Predicted nnu Values')
# plt.xlabel('x1')
# plt.ylabel('nnu')
# plt.legend()
# plt.grid(True)
# #plt.show()
# plt.savefig('True_Pred_fxk_q.pdf')

# plt.figure(4, figsize=(10, 6))
# plt.plot(x2Vals, true_values_2, label='True nnubar', linestyle='--')
# plt.plot(x2Vals, predicted_values_2, label='Predicted nnubar')
# plt.title('Comparison of True and Predicted nnubar Values')
# plt.xlabel('x2')
# plt.ylabel('nnubar')
# plt.legend()
# plt.grid(True)
# #plt.show()
# plt.savefig('True_Pred_fxk_qbar.pdf')