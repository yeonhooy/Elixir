# Elixir 

This repository contains the implementation of *Elixir*, a network hypervisor for control channel isolation with control traffic prediction.

## Overview

We provide all source codes, including *Elixir* implementation and experiment scripts. The provided codes run the entire virtualized SDN system consisting of emulated physical network, *Meteor* hypervisor, virtualized networks, and SDN controllers for the virtualized networks.

The code also includes the training and inference process of *Meteor* predictor, a machine learning model for control traffic prediction. With the *Meteor* predictor, *Meteor* achieves control channel isolation. 

* [Part 1](#Repository-organization) describes the source-code organization of this repository.
* [Part 2](#Settings) contains the steps to configure dependencies and compilation to run *Meteor*. We provide setting steps for physical network emulation, SDN controller, *Meteor*, network configurations, and *Meteor* predictor.
* [Part 3](#Execution-guide) gives a general guide on executing the codes. 

## Repository organization 

The repository is organized as follows:

* `PhysicalTopology/`: contains scripts for physical network topology using Mininet; `linear.py`, `fattree.py`.
* `SDNcontroller/`: contains scripts for executing SDN controllers that control virtualized networks; `onos.sh`.
* `Meteor/`: contains the core implementation of *Meteor* hypervisor. *Meteor* is built as Java Maven project.
   * `Meteor/run_meteor.sh` starts the *Meteor* hypervisor
   * `Meteor/vnetCreation/` contains scripts for virtual network creations
   * `Meteor/MeteorPredictor/` contains scripts for control traffic inference by *Meteor* predictor
   * `Meteor/MeteorPredictor/model/meterPredictor.pt` is the pre-trained *Meteor* predictor model used in our study
* `MeteorPredictor_training/`: contains training codes and training dataset for *Meteor* predictor.
  * We implemented *Meteor* predictor as an LSTM autoencoder based on this [implementation](https://github.com/lkulowski/LSTM_encoder_decoder).


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
