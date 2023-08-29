# Elixir 

This repository contains the implementation of *Elixir*, a machine learning framework for control traffic prediction in SDN systems

## Overview

We provide all source codes, including *Elixir* implementation and experiment scripts. The provided codes are organized as follows.

* [Part 1](#Repository-organization) describes the source-code organization of this repository.
* [Part 2](#Prerquisites) contains the steps to configure dependencies and compilation to run *Elixir*.
* [Part 3](#Execution-guides) gives a general guide on executing the codes. 

## Repository organization 

The repository is organized as follows:

* `Elixir/`: contains the core implementation of *Elixir*. 
* `Elixir/dataset_new_avg/`: contains the dataset of *Elixir*. 
   * `Elixir/dataset_new_avg/`: Training dataset
       * 'fl_default_of10.csv', 'fl_default_of13.csv', 'odl_default_of10.csv', 'odl_default_of13.csv', 'onos_default_of10.csv', 'onos_default_of13.csv', 'onos_proxyarp_of13.csv', 'onos_stats1_of13.csv', 'p4_default.csv'
   * `dataset_new_avg/inf_realvalue/scoring/`: Scroing datset
       * 'fl_default_of10.csv', 'fl_default_of13.csv', 'odl_default_of10.csv', 'odl_default_of13.csv', 'onos_default_of10.csv', 'onos_default_of13.csv', 'onos_proxyarp_of13.csv', 'onos_stats1_of13.csv', 'p4_default.csv'
       * 'fl_default_of10.xlsx', 'fl_default_of13.xlsx', 'odl_default_of10.xlsx', 'odl_default_of13.xlsx', 'onos_default_of10.xlsx', 'onos_default_of13.xlsx', 'onos_proxyarp_of13.xlsx', 'onos_stats1_of13.xlsx', 'p4_default.xlsx'
   * `dataset_new_avg/inf_realvalue/evaluation/`: Evaluation datset
       * 'fl_default_of10.csv', 'fl_default_of13.csv', 'odl_default_of10.csv', 'odl_default_of13.csv', 'onos_default_of10.csv', 'onos_default_of13.csv', 'onos_proxyarp_of13.csv', 'onos_stats1_of13.csv', 'p4_default.csv'
       * 'fl_default_of10.xlsx', 'fl_default_of13.xlsx', 'odl_default_of10.xlsx', 'odl_default_of13.xlsx', 'onos_default_of10.xlsx', 'onos_default_of13.xlsx', 'onos_proxyarp_of13.xlsx', 'onos_stats1_of13.xlsx', 'p4_default.xlsx'
 * `Elixir/dataset_new_avg/exlixir_model_save/`: pre-trained model (example of trained models of model space)

     
## Prerquisites 
-  Install required dependencies and libraries with conda. Please install [conda](https://www.anaconda.com/download)
-  Execute conda
    `$conda activate`
-  Import conda env
    `$conda env create -f Elixir_env.yaml`
-  Activate env
    `$conda activate Elixir_env`

## Execution guides 

### 1. Training (Model Space) 
  `$ python trainmodel.py [training_dataset csv name]`
  - e.g., `$ python trainmodel.py onos_default_of10.csv onos_default_of13.csv`
  - Please check location of training dataset path: `dataset_new_avg/`
  - Training result(trained models) path: `dataset_new_avg/elixir-modelsave/`
### 2. Inference code 
  `$ python inference.py [test_type] [training_dataset csv name]`
  - inference dataset path: `dataset_new_avg/inf_realvalue/[test_type]/`
    - require both `csv` file and `xlsx` file
    - test_type: scoring or evaluation
  - scoring: `$ python inference.py scoring onos_default_of10.csv onos_default_of13.csv`
  - evaluation: `$ python inference.py evaluation onos_default_of10.csv onos_default_of13.csv`

  - result files(xlsx):  `dataset_new_avg/result/`
### 3. Model selection
1) Naive model: `$ python naiiveselction.py`
   - result path (`txt`): `dataset_new_avg/result/modelselect/naiive.txt`
   - result summary path (`xlsx`): `dataset_new_avg/result/modelselect/naiiveselectModel.xlsx`
3) Freq model: `$ python freq_modelselection.py`
   - result path: `dataset_new_avg/result/modelselect/freq.txt`
   - result summary path: `dataset_new_avg/result/modelselect/freqselectModel.xlsx`
5) Supreme model: `$ python modelselection.py`
   - result path: `dataset_new_avg/result/modelselect/supreme.txt`
   - result summary path: `dataset_new_avg/result/modelselect/naiiveselectModel.xlsx`


### 4. Prediction Accuracy Summary
1) RMSE: `$ python selectedmodelonlysummary_RMSE.py`
* result: `dataset_new_avg/result/modelselect`

2) Freq rate: `$ python freqscore_FREQ.py`
* result: `dataset_new_avg/result/modelselect`
