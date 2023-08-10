# Elixir 

This repository contains the implementation of *Elixir*, a machine learning framework for control traffic prediction in SDN systems

## Overview

We provide all source codes, including *Elixir* implementation and experiment scripts. The provided codes xx.

* [Part 1](#Repository-organization) describes the source-code organization of this repository.
* [Part 2](#Prerquisites) contains the steps to configure dependencies and compilation to run *Elixir*.
* [Part 3](#Execution-guides) gives a general guide on executing the codes. 

## Repository organization 

The repository is organized as follows:

* `Elixir/`: contains the core implementation of *Elixir*. 
* `Elixir/dataset_new_avg/`: contains the dataset of *Elixir*. 
   * `Elixir/dataset_new_avg/`: Training dataset
       * 'fl_default_of10.csv', 'fl_default_of13.csv', 'odl_default_of10.csv', 'odl_default_of13.csv', 'onos_default_of10.csv', 'onos_default_of13.csv', 'onos_proxyarp_of13.csv', 'onos_stats1_of13.csv', 'p4_default.csv'
   * `dataset_new_avg/test_datacenter/`: Inference test dataset
       * 'scor_test.csv'
   * `dataset_new_avg/scoring/: Scroing datset
       * 'fl_default_of10.csv', 'fl_default_of13.csv', 'odl_default_of10.csv', 'odl_default_of13.csv', 'onos_default_of10.csv', 'onos_default_of13.csv', 'onos_proxyarp_of13.csv', 'onos_stats1_of13.csv', 'p4_default.csv'
   * `dataset_new_avg/exlixir_model_save/`
   * `dataset_new_avg/inf_realvalue/new_score_csv/`
   * `dataset_new_avg/result/`
   * `dataset_new_avg/timestamp/`
* `Elixir/topology_experiment/`: contains scripts for virtual network topology using Mininet.
   *  `topology_experiment/vncreation/`: contains scripts for virtual network creations
     
## Prerquisites 
-  install required dependencies and libraries with conda (Please install [conda](https://www.anaconda.com/download))
-  execute conda
    `$conda activate`
-  import conda env
    `$conda env create -f Elixir_env.yaml`
-  activate env
    `$conda activate Elixir_env`

## Execution guides 

### 1. train code 
  `$ python trainmodel.py`
  ```
  dataset path: dataset_new_avg/xxx.csv
  result path: dataset_new_avg/result/modelselect/
  ```
### 2. inference code: 
  `$ python inference.py`
  ```
  - inference dataset path: dataset_new_avg/inf_realvalue/<directory>/xxx.csv (xlsx)
  - require both csv file and xlsx file
  - modify code (question sheet, answer sheet path) 
  - result:  dataset_new_avg/result/xxx.xlsx
  ```
### 3. modelselection code
```
1) naive: $ python naiiveselction.py
2) freq: $ python freq_modelselection.py
3) supreme: $ python modelselection.py

* result: dataset_new_avg/result/modelselect/naiive.txt,freq.txt,supreme.txt
````
### 4. summary
```
1) rmse: $ selectedmodelonlysummary_RMSE.py
* result: dataset_new_avg/result/modelselect

2) freq rate: $ freqscore_FREQ.py
* result: dataset_new_avg/result/modelselect
```
