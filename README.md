# Run experiment on Kubeflow

## Create virtual env

Open a terminal in Kubeflow. Make sure you are in the right location and run:

```console
git clone git@github.com:mickaelassaraf/medical_code_inference.git
```

Make sure `pyenv` is installed
(see Engineering wiki [here](https://www.notion.so/qantev/Manage-Python-versions-c0083f5a47e54c2788734adef5c2f296) or [official documentations](https://github.com/pyenv/pyenv#getting-pyenv) to do so) and run:

```console
pyenv install 3.10.11
pyenv virtualenv 3.10.11  py310
```

Make sure you can access the virtual env in Jupyter notebooks with following lines. Please note you don't execute these lines in the virtual env.

```console
pip install —user ipykernel
python -m ipykernel install --user --name=py310
```

## Config virtual env

Once the virtual env is created, activate it using:

```console
pyenv activate py310
```

In the virtual env, run the following lines to install packages needed for the files to run smoothly:

```console
pip install hydra
pip install hydra-core
pip install wandb
pip install wget
pip install omegaconf
pip install vaex
pip install -r requirements.txt
```

Now your virtual env is set up and ready !

## RoBERTa-PM weights

RoBERTa-PM is needed for training and for inference.
The weights can be find in the [Qantev Drive](https://drive.google.com/drive/folders/1thf5Ckn3sZkrUVQ-N0km2nJrlzcQYWNk) in the following folders `Shared/Tech/ML models/PLM_ICD`.  
You need to extract the file `RoBERTa-base-PM-M3-Voc-hf.tar` and put it in `medical_code_inference`.

## Logging configuration

We use [Weights&Biases](https://wandb.ai/site) for logging experiments. If you don't have an account you need to create one. Once this is done, ask one of the team members to add you to the Qantev space on W&B.  
Open the notebook `preliminary_process.ipynb` and run all cells in it. It will dezip the RoBERTa-PM weights + connect to W&B for logging. For W&B, you will be prompted for your API key that you can find in your user settings.  
You only need to do that operation once.

## Get the data

You have basically 2 options to get the MIMIC data the model will use:

**1. Use directly the dataset already preprocessed:**  
You can copy the already processed dataset from the [Qantev Drive](https://drive.google.com/drive/folders/1kK1FJ-rcnvPdJwGJuVx4ubZ7c7hiYd4v) following this path `Qantev Shared/Tech/MIMIC`.  
You then select `mimiciii_clean_icd.feather` or `mimiciii_clean_cpt.feather` and you put the file in the respective place `medical_code_inference/files/data/mimiciii_clean` (for ICD) or `medical_code_inference/files/data/mimiciii_clean_cpt` (for CPT).  
For MIMIC-IV, you perform the same opeartions with according names. Note that there are no CPT on MIMIC-IV, but two ICD classifications.

**2. Preprocessing MIMIC (NOT PREFERRED):**  
You can copy the MIMIC-III dataset from the [Qantev Drive](https://drive.google.com/drive/folders/1kK1FJ-rcnvPdJwGJuVx4ubZ7c7hiYd4v) following this path `Qantev Shared/Tech/MIMIC` and select `mimic-iii-clinical-database-1.4.zip`  
Put this file in `medical_inference/MIMIC` dezip it and then start the script `prepare_mimiciii_cpt.py` or `prepare_mimiciii_clean.py` if you want to process it for CPT or ICD. \
If you want to preprocess MIMIC-IV, you put both folders `mimic-iv` and `mimic-iv-note` under `MIMIC` and run `prepare_mimiciv.py`.  
Note theses processes might be very long.

For the insurer's data, it can be found on the datalake at this path `datalake/structured_data/axa_icd10.feather` and should be put it in `files/data/axa_icd10`. This data has been obtained by running claims on OCR + translating them in english + reformat them in the same schema as MIMIC-IV after preprocessing.
Also, a merge between MIMIC-IV data and the insurer's data has been performed and can be found in the datalake at this path `datalake/structured_data/mimic_axa_icd10.feather`.
It should be put here `files/data/mimic_axa`
**THIS DATA SHOULD NEVER LEAVE THE DATALAKE**
**IN PARTICULAR MAKE SURE TO NEVER COMMIT THIS DATA ON GITHUB.**

## Run Training

To train the model for the different use case you should run:

- PLM-ICD on MIMIC-III: `python main.py experiment=mimiciii_clean/plm_icd gpu=0`
- PLM-CPT on MIMIC-III : `python main.py experiment=mimiciii_clean/plm_cpt gpu=0`
- PLM-ICD hierarchical on MIMIC-III: `python main.py experiment=mimiciii_clean/plm_icd_hierarchical_embedding gpu=0`
- PLM-CPT hierarchical on MIMIC-III: `python main.py experiment=mimiciii_clean/plm_cpt_hierarchical_embedding gpu=0`
- PLM-ICD on MIMIC-IV: `python main.py experiment=mimiciv_icd10/plm_icd gpu=0`
- PLM-ICD on MIMIC-IV + AXA: `python main.py experiment=axa/plm_icd gpu=0`
- PLM-ICD hierarchical on MIMIC-IV + AXA: `python main.py experiment=axa/plm_icd_hierarchical gpu=0`

`gpu=0` ensures that we use the GPU labeled 0.

## Evaluation

Some checkpoints for the models are available in the [Qantev Drive](https://drive.google.com/drive/folders/1xpqzrYDCT6HBsuOQVUo9NTxr-r7hXpOi) following this path `Qantev Shared/Tech/ML models/PLM_ICD/model_checkpoints`.

If you just want to evaluate the models using the provided model_checkpoints you need to set `trainer.epochs=0` and provide the path to the models checkpoint `load_model=path/to/model_checkpoint`. Make sure you use the correct model-checkpoint with the correct configs.

Examples:

- Evaluate PLM-ICD on MIMIC-IV ICD-10 on GPU 0: `python main.py experiment=mimiciv_icd10/plm_icd gpu=0 load_model=path/to/model_checkpoints/mimiciv_icd10/plm_icd epochs=0`
- Evaluate PLM-CPT hierarchical on MIMIC-III clean on GPU 0: `python main.py experiment=mimiciii_clean/plm_cpt_hierarchical_embedding gpu=0 load_model=path/to/model_checkpoints/mimiciii_clean/plm_cpt_hierarchical epochs=0`

\
\
\
\

# README from the original github repo for reference

# ⚕️Automated Medical Coding on MIMIC-III and MIMIC-IV: A Critical Review and Replicability Study

Official source code repository for the SIGIR 2023 paper [Automated Medical Coding on MIMIC-III and MIMIC-IV: A Critical Review and Replicability Study](https://arxiv.org/abs/2304.10909)

```bibtex
@inproceedings{edinAutomatedMedicalCoding2023,
  address = {Taipei, Taiwan},
  title = {Automated {Medical} {Coding} on {MIMIC}-{III} and {MIMIC}-{IV}: {A} {Critical} {Review} and {Replicability} {Study}},
  isbn = {978-1-4503-9408-6},
  shorttitle = {Automated {Medical} {Coding} on {MIMIC}-{III} and {MIMIC}-{IV}},
  doi = {10.1145/3539618.3591918},
  booktitle = {Proceedings of the 46th {International} {ACM} {SIGIR} {Conference} on {Research} and {Development} in {Information} {Retrieval}},
  publisher = {ACM Press},
  author = {Edin, Joakim and Junge, Alexander and Havtorn, Jakob D. and Borgholt, Lasse and Maistro, Maria and Ruotsalo, Tuukka and Maaløe, Lars},
  year = {2023}
}
```

## Introduction

Automatic medical coding is the task of automatically assigning diagnosis and procedure codes based on discharge summaries from electronic health records. This repository contains the code used in the paper Automated medical coding on MIMIC-III and MIMIC-IV: A Critical Review and Replicability Study. The repository contains code for training and evaluating medical coding models and new splits for MIMIC-III and the newly released MIMIC-IV. The following models have been implemented:

| Model       | Paper                                                                                                                          | Original Code                                                                          |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- |
| CNN         | [Explainable Prediction of Medical Codes from Clinical Text](https://aclanthology.org/N18-1100/)                               | [link](https://github.com/jamesmullenbach/caml-mimic)                                  |
| Bi-GRU      | [Explainable Prediction of Medical Codes from Clinical Text](https://aclanthology.org/N18-1100/)                               | [link](https://github.com/jamesmullenbach/caml-mimic)                                  |
| CAML        | [Explainable Prediction of Medical Codes from Clinical Text](https://aclanthology.org/N18-1100/)                               | [link](https://github.com/jamesmullenbach/caml-mimic)                                  |
| MultiResCNN | [ICD Coding from Clinical Text Using Multi-Filter Residual Convolutional Neural Network](https://arxiv.org/pdf/1912.00862.pdf) | [link](https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network) |
| LAAT        | [A Label Attention Model for ICD Coding from Clinical Text](https://arxiv.org/abs/2007.06351)                                  | [link](https://github.com/aehrc/LAAT)                                                  |
| PLM-ICD     | [PLM-ICD: Automatic ICD Coding with Pretrained Language Models](https://aclanthology.org/2022.clinicalnlp-1.2/)                | [link](https://github.com/MiuLab/PLM-ICD)                                              |

The splits are found in `files/data`. The splits are described in the paper.

## How to reproduce results

### Setup Conda environment

1. Create a conda environement `conda create -n coding python=3.10`
2. Install the packages `pip install . -e`

### Prepare MIMIC-III

This code has been developed on MIMIC-III v1.4.

1. Download the MIMIC-III data into your preferred location `path/to/mimiciii`. Please note that you need to complete training to acces the data. The training is free, but takes a couple of hours. - [link to data access](https://physionet.org/content/mimiciii/1.4/)
2. Open the file `src/settings.py`
3. Change the variable `DOWNLOAD_DIRECTORY_MIMICIII` to the path of your downloaded data `path/to/mimiciii`
4. If you want to use the MIMIC-III full and MIMIC-III 50 from the [Explainable Prediction of Medical Codes from Clinical Text](https://aclanthology.org/N18-1100/) you need to run `python prepare_data/prepare_mimiciii_mullenbach.py`
5. If you want to use MIMIC-III clean from our paper you need to run `python prepare_data/prepare_mimiciii.py`

### Prepare MIMIC-IV

This code has been developed on MIMIC-IV and MIMIC-IV v2.2.

1. Download MIMIC-IV and MIMIC-IV-NOTE into your preferred location `path/to/mimiciv` and `path/to/mimiciv-note`. Please note that you need to complete training to acces the data. The training is free, but takes a couple of hours. - [link to data access](https://physionet.org/content/mimiciii/1.4/)
2. Open the file `src/settings.py`
3. Change the variable `DOWNLOAD_DIRECTORY_MIMICIV` to the path of your downloaded data `path/to/mimiciv`
4. Change the variable `DOWNLOAD_DIRECTORY_MIMICIV_NOTE` to the path of your downloaded data `path/to/mimiciv-note`
5. Run `python prepare_data/prepare_mimiciv.py`

### Before running experiments

1. Create a weights and biases account. It is possible to run the experiments without wandb.
2. Download the [model checkpoints](https://drive.google.com/file/d/1hYeJhztAd-JbhqHojY7ZpLtkBcthD8AK/view?usp=share_link) and unzip it. Please note that these model weights can't be used commercially due to the MIMIC License.
3. If you want to train PLM-ICD, you need to download [RoBERTa-base-PM-M3-Voc](https://dl.fbaipublicfiles.com/biolm/RoBERTa-base-PM-M3-Voc-hf.tar.gz), unzip it and change the `model_path` parameter in `configs/model/plm_icd.yaml` to the path of the download.

### Running experiments

#### Training

You can run any experiment found in `configs/experiment`. Here are some examples:

- Train PLM-ICD on MIMIC-III clean on GPU 0: `python main.py experiment=mimiciii_clean/plm_icd gpu=0`
- Train CAML on MIMIC-III full on GPU 6: `python main.py experiment=mimiciii_full/caml gpu=6`
- Train LAAT on MIMIC-IV ICD-9 full on GPU 6: `python main.py experiment=mimiciv_icd9/laat gpu=6`
- Train LAAT on MIMIC-IV ICD-9 full on GPU 6 without weights and biases: `python main.py experiment=mimiciv_icd9/laat gpu=6 callbacks=no_wandb trainer.print_metrics=true`

#### Evaluation

If you just want to evaluate the models using the provided model_checkpoints you need to do set `trainer.epochs=0` and provide the path to the models checkpoint `load_model=path/to/model_checkpoint`. Make sure you the correct model-checkpoint with the correct configs.

Example:
Evaluate PLM-ICD on MIMIC-IV ICD-10 on GPU 1: `python main.py experiment=mimiciv_icd10/plm_icd gpu=1 load_model=path/to/model_checkpoints/mimiciv_icd10/plm_icd epochs=0`

## Overview of the repository

#### configs

We use [Hydra](https://hydra.cc/docs/intro/) for configurations. The condigs for every experiment is found in `configs/experiments`. Furthermore, the configuration for the sweeps are found in `configs/sweeps`. We used [Weights and Biases Sweeps](https://docs.wandb.ai/guides/sweeps) for most of our experiments.

#### files

This is where the images and data is stored.

#### notebooks

The directory only contains one notebook used for the code analysis. The notebook is not aimed to be used by others, but is included for others to validate our data analysis.

#### prepare_data

The directory contains all the code for preparing the datasets and generating splits.

#### reports

This is the code used to generate the plots and tables used in the paper. The code uses the Weights and Biases API to fetch the experiment results. The code is not usable by others, but was included for the possibility to validate our figures and tables.

#### src

This is were the code for running the experiments is found.

#### tests

The directory contains the unit tests

## Acknowledgement

Thank you Sotiris Lamprinidis for providing an efficient implementation of our multi-label stratification algorithm and some data preprocessing helper functions.
