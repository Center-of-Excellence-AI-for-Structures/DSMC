# Deep Soft Monotonic Clustering (DSMC) model

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Code for paper "A robust generalized deep monotonic feature extraction model for label-free prediction of degenerative phenomena".

In this paper, we propose a deep monotonic unsupervised clustering model for feature extraction and clustering analysis in deteriorating systems. The model innovatively extracts prognostic-related features from raw, multi-modal data, capturing increasing monotonic features representing system deterioration. Then, it performs monotonic clustering, revealing information about deterioration and possible recoveries.

![alt text](https://github.com/Center-of-Excellence-AI-for-Structures/DSMC/blob/master/Figs/DC_model.jpg)


## Table of Contents

- [Environment and Requirements](#environment-and-requirements)
- [Configuration and Installation](#configuration-and-installation)
- [Data Structure](#data-structure)
- [Example](#example)
- [Contributors](#contributors)

## Requirements

- Tested on Windows 10
- Either GPU or CPU (Tested on Nvidia GeForce RTX 2080 GPU)
- The developed version of the code mainly depends on the following `Python 3.9.12` packages.

  ```
  torch==1.11.0
  torchvision==0.12.0
  torchaudio==0.11.0
  numpy==1.23.4
  pandas==1.5.3
  matplotlib==3.5.2
  seaborn==0.12.1
  scikit-learn==1.2.2
  scikit-survival==0.21.0
  joblib==1.2.0
  tslearn==0.5.2
  vallenae==0.7.0
  tqdm==4.64.0
  ```

## Installation
The steps to configure and install the packages are the following:

1. Create an Anaconda environment and install PyTorch. In step 1c, please select the correct Pytorch version that matches your CUDA version from https://pytorch.org/get-started/previous-versions/. Open an Anaconda terminal and run the following:

 Step 1a

```
conda create -n dsmc_env python=3.9.12
```

 Step 1b

```
conda activate dsmc_env
```

 Step 1c

```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```



2. This repository can be directly installed through GitHub by the following commands:



```
conda install git
```

```
git clone https://github.com/Center-of-Excellence-AI-for-Structures/DSMC.git
```

```
cd DSMC
```

```
python setup.py install
```

```
conda install numpy-base==1.23.4
```

## Structure

In this project, three datasets are considered, namely the MIMIC-III (https://mimic.mit.edu/), the C-MAPSS dataset (https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6), and the F-MOC (Fatigue Monitoring of Composites) dataset, i.e. an experimental campaign (https://data.mendeley.com/drafts/4zm6jh8jkd). 
The C-MAPSS dataset is publicly available and free. The only required file to be saved in the working directory (`dsmc` folder) is the `train_FD001.txt` downloaded from the CMAPSS dataset. Then, this file will be automatically moved inside the `CMAPS/` folder. 

The MIMIC-III dataset is publicly available and free, but it requires signing a data use agreement and passing a recognized course in protecting human research participants that includes Health Insurance Portability and Accountability Act (HIPAA) requirements. Approval requires at least a week.

After approval, the following CSV files should be saved in the working directory (`dsmc` folder):

- ADMISSIONS.csv
- CHARTEVENTS.csv
- D_ITEMS.csv
- D_LABITEMS.csv
- LABEVENTS.csv
- PATIENTS.csv

Then,  these files will be automatically moved inside the `MIMIC/data/` folder.

The F-MOC dataset is publicly available and free. It is stored and licensed under the Mendeley umbrella. After downloading the .zip file, simply extract the two folders named "ACOUSTIC" and "DIC" into the working directory (`dsmc` folder). Details about this dataset and the data acquisition process can be found at https://data.mendeley.com/drafts/4zm6jh8jkd. Unlike the other datasets where the pretrained models are automatically installed to the local system, this dataset's trained models exceed the maximum size of files that can be stored in GitHub. Consequently, we stored these models to the Mendeley repository as well with the name "models.zip". If these pretrained models are needed, simply extract the .zip file inside the `dsmc/models/` directory.
 
### Data Files Distribution
The files and folders of the project are distributed in the following manner ('--Required' means that these files and folders are necessary to be created before running the `main.py`, the rest are automatically created)

```

../DSMC/
      └── setup.py
      └── Readme.md
      └── requirements.txt
      └── LICENSE
    
      ├── dsmc/

            ├── bayesian_opt/                                -- Required      
            │ │   └── __init__.py                            -- Required 
            │ │   └── bayesian_optimization.py               -- Required 
            │ │   └── event.py                               -- Required 
            │ │   └── logger.py                              -- Required 
            │ │   └── observer.py                            -- Required 
            │ │   └── target_space.py                        -- Required 
            │ │   └── util.py                                -- Required 
            
            ├── CMAPS/                                       
            │ │   └── original/
            │ │ │    └── sp_0.csv
            │ │ │    └── sp_1.csv
            │ │ │    └── ...
            │ │   └── sorted/
            │ │ │    └── sp_0.csv
            │ │ │    └── sp_1.csv
            │ │ │    └── ...
            │ │   └── train_FD001.txt                        -- Required (should be put in the `dsmc' folder directory)
            
            ├── events/
            │ │   └── test_cmaps_events.csv
            │ │   └── test_mimic_events.csv
            │ │   └── train_cmaps_events.csv
            │ │   └── train_mimic_events.csv
            
            ├── hyperparameters/
            │ │   └── hyper_cmaps.json
            │ │   └── hyper_mimic.json
            │ │   └── hyper_both.json
            
            ├── MIMIC/                                        
            │ │   └── data/                                   
            │ │ │      └── ADMISSIONS.csv                     -- Required (should be put in the `dsmc' folder directory)
            │ │ │      └── CHARTEVENTS.csv                    -- Required (should be put in the `dsmc' folder directory)
            │ │ │      └── D_ITEMS.csv                        -- Required (should be put in the `dsmc' folder directory)
            │ │ │      └── D_LABITEMS.csv                     -- Required (should be put in the `dsmc' folder directory)
            │ │ │      └── LABEVENTS.csv                      -- Required (should be put in the `dsmc' folder directory)
            │ │ │      └── PATIENTS.csv                       -- Required (should be put in the `dsmc' folder directory)
            │ │   └── vital_signs/
            │ │ │      └── time_series_0.csv
            │ │ │      └── time_series_1.csv
            │ │ │      └── ...
            │ │   └── demographic.csv
            │ │   └── lab.csv
        
            
            ├── models/                                       
            │ │   └── cmaps/
            │ │ │      └── ae_model_cmaps.pt                 -- Required (if the user chooses pretrained=True)
            │ │ │      └── dc_model_cmaps.pt                 -- Required (if the user chooses pretrained=True)
            │ │   └── mimic/
            │ │ │      └── ae_model_mimic.pt                 -- Required (if the user chooses pretrained=True)
            │ │ │      └── dc_model_mimic.pt                 -- Required (if the user chooses pretrained=True)
            │ │   └── both/
            │ │ │      └── ae_model_both.pt                  -- Required (if the user chooses pretrained=True)
            │ │ │      └── dc_model_both.pt                  -- Required (if the user chooses pretrained=True)
          
            
            ├── results/
            │ │   └── cmaps/
            │ │ │      └── cluster_embds/
            │ │ │      └── clusters/
            │ │ │      └── compare/
            │ │ │ |        └── clusters/
            │ │ │ |        └── prognostics/
            │ │ │      └── figs/
            │ │ │      └── loss/
            │ │ │      └── prognostics/
            │ │ │      └── time_grads/
            │ │ │      └── z_space
            │ │   └── mimic/
            │ │ │      └── cluster_embds/
            │ │ │      └── clusters/
            │ │ │      └── figs/
            │ │ │      └── loss/
            │ │ │      └── prognostics/
            │ │ │      └── time_grads/
            │ │ │      └── z_space
            │ │   └── both/
            │ │ │      └── cluster_embds/
            │ │ │      └── clusters/
            │ │ │      └── figs/
            │ │ │      └── loss/
            │ │ │      └── prognostics/
            │ │ │      └── time_grads/
            │ │ │      └── z_space

            ├── run_prognostics/                             -- Required      
            │ │   └── __init__.py                            -- Required 
            │ │   └── prognostic_models.py                   -- Required
            │ │   └── hsmm/                                  -- Required
            │ │ |      └── hsmm_base.py                      -- Required
            │ │ |      └── hsmm_utils.py                     -- Required
            │ │ |      └── hsmm_base.py                      -- Required
            │ │ |      └── mle.py                            -- Required
            │ │ |      └── smoothed.cp39-win_amd64.pyd       -- Required

            ├── scalers/
            │ │   └── cmaps/
            │ │ │      └── scaler_t_f.save
            │ │ │      └── scaler_x.save
            │ │ │      └── scaler_y.save
            │ │   └── mimic/
            │ │ │      └── scaler_demo.save
            │ │ │      └── scaler_t_f.save
            │ │ │      └── scaler_x.save
            │ │ │      └── scaler_y.save
            │ │   └── both/
            │ │ │      └── scaler_t_f.save
            │ │ │      └── scaler_x.save
            │ │ │      └── scaler_y.save

            ├── sensors/                                       
            │ │   └── ACOUSTIC/                              -- Required (should be put in the `dsmc' folder directory)
            │ │ │    └── spec1.pridb                         -- Required (should be put in the `dsmc' folder directory)
            │ │ │    └── spec2.pridb                         -- Required (should be put in the `dsmc' folder directory)
            │ │ │    └── ...
            │ │   └── DIC/                                   -- Required (should be put in the `dsmc' folder directory)
            │ │ │    └── spec1_6020.tif                      -- Required (should be put in the `dsmc' folder directory)
            │ │ │    └── spec1_6070.tif                      -- Required (should be put in the `dsmc' folder directory)
            │ │ │    └── ...
            
            ├── conv_lstm.py                                 -- Required
            ├── hyperparameters.py                           -- Required 
            ├── main.py                                      -- Required
            ├── mimic_data.py                                -- Required   
            ├── models.py                                    -- Required 
            ├── read_files.py                                -- Required 
            ├── run_models.py                                -- Required
            ├── sepsis_score_systems.py                      -- Required
            ├── settings.py                                  -- Required 
            ├── utils.py                                     -- Required 
            ├── visualize.py                                 -- Required 

```

## Example

To specifically describe how to train and use the DSMC model, we show an example below. To run the code from the Anaconda terminal with default values, go to the `dsmc` folder inside the `DSMC` directory and run the `main.py` file via the commands:

```
cd dsmc
```

```
python main.py
```

This runs the DSMC model for the C-MAPSS dataset by default. If you want to run the trained models for the MIMIC-III dataset (ensure the required files are saved to the working directory) without retraining from scratch, run the command:

```
python main.py --mimic True --pretrained True
```

If you want to enable the Bayesian Optimization algorithm and not rely on the existing hyperparameters, and to train the DSMC model on the C-MAPSS dataset, run the command:

`python main.py --bayesian_opt True`

If you want to run the trained models for the F-MOC dataset (ensure the required files are saved to the working directory) without retraining from scratch, run the command:

```
python main.py --both True --pretrained True
```

See the `main.py` file for different existing variables and options.

### Results

The results are saved inside the directory `../DSMC/dsmc/results/`The clustering results for 10 trajectories of the C-MAPSS and MIMIC-III datasets, respectively are shown below:

![alt text](https://github.com/Center-of-Excellence-AI-for-Structures/DSMC/blob/master/Figs/Clustering_results.jpg)


>**Note**
>The results may be slightly different for different hardware setups. Additionally, varying tuned hyperparameters may be used after running the Bayesian Optimization algorithm on different hardware. This explains why we presented in the paper the mean and variance of the losses over 10 independent runs of the code, for the 2 first datasets that require a limited amount of memory.

>**Warning**
>The corresponding figures come after setting the seeding of the algorithm, which is different depending on the computer system, thus the tuned hyperparameters correspond to our specific hardware (Nvidia GeForce RTX 2080 GPU). Therefore, for reproducibility, it is highly recommended to run the `main.py` with its default arguments for the C-MAPSS dataset, whilst using the pre-trained models for the MIMIC-III and F-MOC dataset or running the Bayesian Optimization algorithm from scratch (however, this may take a long time depending on the hardware system). See this thread for running the same models on different hardware https://discuss.pytorch.org/t/large-difference-in-results-between-cpu-and-cuda/184858/5.


## Contributors

- [Panagiotis Komninos](https://github.com/panoskom)
- [Athanasios Kontogiannis](https://github.com/thanoskont)


