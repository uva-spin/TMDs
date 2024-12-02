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

#Test19
Took Test12 model and used sigmoid activation function in the output layer




