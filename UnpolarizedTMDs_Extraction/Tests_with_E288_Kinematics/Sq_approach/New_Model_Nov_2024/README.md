# This is the log of the evolution of this folder

#Test0
This folder contains some initial files.

#Test1
Pseudodata was generated for the function A(pT) = integration_0_2 dk [S1(k)S2(pT-k) + S1(pT-k)S2(k)]
The two DNNs represents the two flavors S1 and S2 k dependence

#Test2
Testing with different options in the architecture

#Test3 
Considering the phi integration
Updated the pseudo-data generation script as well as fitting script

#Test4
Considering two inputs k and QM for S1 and S2
k integration limits 0 to 2

#Test5
Same as Test4 but k integration from 0.0001 to 10

#Test6
Same as Test5 but splitting the training part which saves two separate models and an additional script to make plots based on the saved models. Train only for 100 epochs.

#Test7
Same as Test6 but using 1000 epochs
architecture: 2,64,64,128,64,1
Result: Doesn't seem to be optimized yet

#Test8
Same as Test6 (added the loss plot)
500 epochs
trial1: Epochs=500,  Arc: 2, 64, 32, 16, 8, 1: not the best
trial2: Epochs=500, Arc:32x11,1 : not the best

#Test9
Same as Test8 but with only one input for the DNNs

#Test10
Same as Test8 but generating pseudodata with only a few values of QM (1,2,3) and use one of that in the evaluation.

#Test11
Same as Test8 but connecting QM part of the DNN differently (separate sub-structure in the architecture).


#Obseverations at this stage
Both Test10 and Test11 seems going in the right direction but need to optimize

#Test12
Same as Test11 but changed the architecture to have layers 64,32,16 for each k and QM paths and 1000 epochs

#Test13
Same as Test10: 1000 epochs and LR = 0.0001: the loss got down to ~400

#Test14
Same as Tes10/Test13 but with Epochs = 1000
So far, this is the best one

#Test15
Same as Test10/Test13 let's call it Arc#1 with
LR = 0.001, Epochs = 1000, Arc: 256,128,64,32,16,8,1
kins = phibins = 25

#Test16
Same as Test15 but with LR = 0.0001
Not optimized yet.

#Test17
Took Test14 model and added another layer of 128 nodes after the input layer.

#Test18
Took Test14 model and used sigmoid activation function in the output layer
Loss function is large ~62000
So looks like "sigmoid" in the final layer doesn't work

#Test19
Took Test12 model and used sigmoid activation function in the output layer
Loss function is large ~62000
So looks like "sigmoid" in the final layer doesn't work

At this stage, let's call the two models
Mod_A: k,Q2 inputs processed together in the DNN model: Test14
Mod_B: k,Q2 inputs processed separately inside the DNN model: Test11,12,19

#Test20
Used Mod_A with Keras Tuner
This contains trials with different approeaches even including the tensor board visualization. Note: delete the folders (logs, my_tuner_dir) before running the Fit.py code. 
Arc | Layer1 | activ1 |Layer2 | activ2 | Layer3 | activ3  | Optimizer | Learning Rate
23  | 128    | relu6  | 16    |  relu6 |  8     | relu6   | adam      | 0.00302184616
24  | 128    | relu6  | 32    |  relu  |  8     | relu6   | rmsprop   | 0.00614510259
25  | 48     | relu6  | 64    |  relu6 |  8     | relu6   | adam      | 0.00697462396
26  | 112    | relu   | 32    |  relu  | 24     | relu    | adam      | 0.00771223638



#Test21
Took Test14 model and added another layer of 128 nodes after the input layer + changed the learning rate to 0.00005

#Test22
Same as 21 but with 25 phi bins

#Test23
Arc | Layer1 | activ1 |Layer2 | activ2 | Layer3 | activ3  | Optimizer | Learning Rate
23  | 128    | relu6  | 16    |  relu6 |  8     | relu6   | adam      | 0.00302184616


#Test24
Arc | Layer1 | activ1 |Layer2 | activ2 | Layer3 | activ3  | Optimizer | Learning Rate
24  | 128    | relu6  | 32    |  relu  |  8     | relu6   | rmsprop   | 0.00614510259

#Test25
Arc | Layer1 | activ1 |Layer2 | activ2 | Layer3 | activ3  | Optimizer | Learning Rate
25  | 48     | relu6  | 64    |  relu6 |  8     | relu6   | adam      | 0.00697462396

#Test26
Arc | Layer1 | activ1 |Layer2 | activ2 | Layer3 | activ3  | Optimizer | Learning Rate
26  | 112    | relu   | 32    |  relu  | 24     | relu    | adam      | 0.00771223638

#Test27
Implementing the constraints for S1 and S2 in which those integrals of k separately equal to 1. Here, I used 25 bins for k and phi
QM for S1 and S2 was used as zeros.

#Test28
Implementing f(x) values using lhapdf. Modified the Test27 code accordingly to test this.


#Test29
Same as Test27 but with time stamps and epochs 2000
QM for S1 and S2 was used as 2.0.

#Test30
NO PERMUTATION
Wanted to try without permutation (from Test29 code)
Saved the results for epochs = {2000, 3000} separate folders

#Test31
NO PERMUTATION
Same code as Test30 but with 2000 epochs with 100 kbins and 100 phi bins


#Test32
Same as Test14 but with 3000 epochs

#Test33
Modified the Test32 to use the constraints implemented in Test27. QM was used as 2 GeV.







