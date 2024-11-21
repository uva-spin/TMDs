# This is the log of the evolution of this folder

## k_0_2_v2_From_Working_Code

This folder was taken from the previous test with the following conditions
1. k limit from 0 to 2 GeV
2. qT - k was considered
3. Permutation was considered
4. DNN was for f(x,k,QM



## k_0_2_v2_with_Sk_Factorized

This folder contains the modifications to the scripts in k_0_2_v2_From_Working_Code in order to have the DNN as Sq(k,QM) with two inputs and factorized the Collinear PDFs out. Used the same pseudo-data as in the previous folder.
1. k limit from 0 to 2 GeV
2. qT - k was considered
3. Permutation was considered
4. DNN was for S(k,QM)


## k_0_2_with_vec_k_phi_0
This is with the Sk factorized form 
1. k limit from 0 to 2 GeV
2. sqrt(qT^2+k^2-2qTkcos(phi)) was considered with phi=0, Generated pseudodata with phi=0
3. Permutation was considered
4. DNN was for S(k,QM)


## k_0_2_with_vec_k_phi_90
This is with the Sk factorized form 
1. k limit from 0 to 2 GeV
2. sqrt(qT^2+k^2-2qTkcos(phi)) was considered with phi=90, Generated pseudodata with phi=90
3. Permutation was considered
4. DNN was for S(k,QM)
