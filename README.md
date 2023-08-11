# STRAP: Spatio-Temporal Framework for Real Estate Apprisal

This repository contains a PyTorch implementation of the paper

STRAP: A Spatio-Temporal Framework for Real Estate Apprisal., Lee et al., CIKM 2023 (Short).

## Requirements

### Install Anaconda environment

We assume you have Linux device to run these scripts.
First, install anaconda environment to train deep learning model.

```
wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
bash Anaconda3-2020.07-Linux-x86_64.sh
source ~/.bashrc
```


### Install dependencies
We assume you have access to a GPU that can run CUDA 11.1 and CUDNN 8. 
Then, the simplest way to install all required dependencies is to create an anaconda environment by running

```
conda env create -f requirements.yaml
```

After the instalation ends you can activate your environment with
```
conda activate strap
```

Then furhter install few more dependencies
```
pip install hydra-core --upgrade
pip install opencv-python
```

### Signup Wandb

https://wandb.ai/home

## Instructions

### Check data preprocessing

If you want to see data pre-processing procedure, check the below jupyter notebook scripts.

```
./data/dataset_summary.ipynb
```

### Train model

If you want to train the model from your own, use the `main.py` script

```
python main.py
```
