# TMDs

## LHAPDF Instructions

Please refer instructions on this wiki page: https://confluence.its.virginia.edu/pages/viewpage.action?pageId=46553452 


## Instructions for running on Rivanna

Implement the following commands before submitting the job using 'sbatch'

* module load anaconda/2020.11-py3.8

* pip install --user iminuit

* pip install --user tabulate

* Installing Tensorflow manually to one's Rivanna directory:

module load anaconda
conda create -n tf-2.7 python=3.8 ipykernel cudatoolkit=11 cudnn
source activate tf-2.7
pip install tensorflow-gpu==2.7rc1
python -m ipykernel install --user --name tf-2.7 --display-name tf-2.7
You can add additional package with conda install or pip install (after source activate tf-2.7). The last command installs a Jupyter kernel for the tf-2.7 conda environment. In order to use the required cuda, cudnn, etc. libraries, you need to edit the generated ~/.local/share/jupyter/kernels/tf-2.7/kernel.json file so it looks like this (e.g. addition of lines 14-16:

{
 "argv": [
  "/home/das5pzq/.conda/envs/tf-2.7/bin/python",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "tf-2.7",
 "language": "python",
 "metadata": {
  "debugger": true
 },
 "env": {
  "LD_LIBRARY_PATH": "/home/das5pzq/.conda/envs/tf-2.7/lib:$LD_LIBRARY_PATH"
 }
}
The new kernel will be available as “tf-2.7” in Open OnDemand JupyterLab sessions. Alternatively, you use these lines in SLURM scripts:

module load anaconda
source activate tf-2.7
python your_script.py
