# Elixir 

This repository contains the implementation of *Elixir*, a machine learning framework for control traffic prediction in SDN systems

## Overview

We provide all source codes, including *Elixir* implementation and experiment scripts. The provided codes xx.

The code also includes the training and inference process of *Meteor* predictor, a machine learning model for control traffic prediction. With the *Meteor* predictor, *Meteor* achieves control channel isolation. 

* [Part 1](#Repository-organization) describes the source-code organization of this repository.
* [Part 2](#Settings) contains the steps to configure dependencies and compilation to run *Meteor*. We provide setting steps for physical network emulation, SDN controller, *Meteor*, network configurations, and *Meteor* predictor.
* [Part 3](#Execution-guide) gives a general guide on executing the codes. 

## Repository organization 

The repository is organized as follows:

* `topology_experiment/`: contains scripts for virtual network topology using Mininet.
   *  `topology_experiment/vncreation/`: contains scripts for virtual network creations
* `dataset_new_avg/`: contains the core implementation of *Elixir*. 
   * `dataset_new_avg/exlixir_model_save/` 
   * `dataset_new_avg/inf_realvalue/new_score_csv/`
   * `dataset_new_avg/result/`
   * `dataset_new_avg/scoring/`
   * `dataset_new_avg/test_datacenter/`
   * `dataset_new_avg/timestamp/`

## Execution instructions 

### 1. train code 
  `$ python trainmodel.py`
  ```
  dataset path: dataset_new_avg/xxx.csv
  result paht: dataset_new_avg/result/modelselect/
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
