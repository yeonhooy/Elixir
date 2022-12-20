# Elixir 

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
