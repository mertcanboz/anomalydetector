# MS Anomaly Detector (Fork)
## Dev Environment 
Tested on Windows 11 64bit with conda 4.10.3
1. Install Anaconda
2. Microsoft Visual C++ 14.0 or greater is required. Install them from https://visualstudio.microsoft.com/visual-cpp-build-tools/
3. Create Conda Environment:
```
(base)> conda env create -f conda.yml
(base)> conda activate devenv-anomalydetector
(devenv-anomalydetector)> pip install -e .
```
4. Run tests:
```
(devenv-anomalydetector)> python -m unittest discover -s tests
```
5. Run Main:
```
(devenv-anomalydetector)> python .\main.py
```

## Reproduce Paper Results

### KPI dataset

#### Model SR
Extract `kpi\kpi.7z` in its folder. Activate the environment. 

```
(base)> conda activate devenv-anomalydetector
(devenv-anomalydetector)> python kpi\ingest_kpi_dataset.py --csv-input-file kpi\kpi_train.csv --generate-data-and-plots
(devenv-anomalydetector)> python kpi\ingest_kpi_dataset.py --csv-input-file kpi\kpi_test.csv --generate-data-and-plots

(devenv-anomalydetector)> python sr\sr_evalue.py --csv-input-dir kpi\kpi_test_ts_data --parallel
```

| Dataset | Model | `batch_size` | `mag_window` | `score_window` | `threshold` | F1 Score | Precision | Recall | Time |
| ------- |------|--------------| ------------ | -------------- | ----------- |----------| --------- | ------ | ---- |
| KPI (train) | SR   | -1           | 3                    | 10000 | 0.375 | 0.66361  | 0.81299 | 0.5906 | 10.12s |
| KPI (test) | SR   | -1           | 3            | 10000 | 0.375 | 0.67523  | 0.75665 | 0.60962 | 9.77s |


#### Model SR-CNN
Extract `kpi\kpi.7z` in its folder. Activate the environment. 

```
(base)> conda activate devenv-anomalydetector
(devenv-anomalydetector)> python kpi\ingest_kpi_dataset.py --csv-input-file kpi\kpi_train.csv --generate-data-and-plots
(devenv-anomalydetector)> python kpi\ingest_kpi_dataset.py --csv-input-file kpi\kpi_test.csv --generate-data-and-plots

(devenv-anomalydetector)> python srcnn\generate_data.py --data kpi\kpi_train_ts_data --window 1440
(devenv-anomalydetector)> python srcnn\train.py --data kpi\kpi_train_ts_data --window 1440 --epoch 300 --use-gpu
(devenv-anomalydetector)> python srcnn\evalue.py --data kpi\kpi_train_ts_data_subset  --window 1440 --delay 7
```

| Dataset | Model | `batch_size` | `mag_window` | `score_window` | `threshold` | F1 Score | Precision | Recall | Time |
| ------- |------|--------------| ------------ | -------------- | ----------- |----------| --------- | ------ | ---- |
| KPI (train) | SR   | -1           | 3                    | 10000 | 0.375 | 0.66361  | 0.81299 | 0.5906 | 10.12s |
| KPI (test) | SR   | -1           | 3            | 10000 | 0.375 | 0.67523  | 0.75665 | 0.60962 | 9.77s |

```
best overall threshold : 0.22 best score : 0.628716707937168
(devenv-anomalydetector) azureuser@gpubox:~/cloudfiles/code/Users/emer.rodriguez/anomalydetector$ python srcnn/evalue.py --data kpi/kpi_train_ts_data  --window 1440 --delay 7 --epoch 170

***********************************************
data source: kpi/kpi_test_ts_data      model: sr_cnn
-------------------------------
precision 0.800918949630691
recall 0.5942448680351906
f1 0.6822739659725803
-------------------------------
time used for making predictions: 21431.189744710922 seconds
delay : 7
tem best 0.22859537385832854 0.01
tem best 0.30784463475523066 0.02
tem best 0.3580755043457171 0.03
tem best 0.39677573705325925 0.04
tem best 0.43102654938417756 0.05
tem best 0.4605182001176206 0.060000000000000005
tem best 0.4835088180435014 0.06999999999999999
tem best 0.49350386573235927 0.08
tem best 0.511933345873397 0.09
tem best 0.5279456582024438 0.09999999999999999
tem best 0.5413659535891308 0.11
tem best 0.553536817317272 0.12
tem best 0.5580088218324845 0.13
tem best 0.5672583127006541 0.14
tem best 0.5763094278807414 0.15000000000000002
tem best 0.58031176652356 0.16
tem best 0.588718071903275 0.17
tem best 0.590497921929753 0.18000000000000002
tem best 0.6359541810861052 0.19
tem best 0.642444887394758 0.2
tem best 0.6443967796021328 0.21000000000000002
tem best 0.6508230876662774 0.22
tem best 0.6564421437553521 0.23
tem best 0.6621816284737633 0.24000000000000002
tem best 0.6668866160783019 0.25
tem best 0.6697135602634293 0.26
tem best 0.673446182918822 0.27
tem best 0.6779395748457102 0.28
tem best 0.6823921345575689 0.29000000000000004
tem best 0.6863453552627278 0.3
tem best 0.6893918196033249 0.31
tem best 0.6926806665673767 0.32
tem best 0.6960016642396505 0.33
tem best 0.6989176202713901 0.34
tem best 0.701816712407186 0.35000000000000003
tem best 0.7051637791012454 0.36000000000000004
tem best 0.708203354101677 0.37
tem best 0.7115479870872974 0.38
tem best 0.7142967376859434 0.39
tem best 0.7167299561709419 0.4
tem best 0.7170798848180814 0.5
tem best 0.7185975748217448 0.51
tem best 0.7191912562328511 0.52
tem best 0.7208315810645262 0.53
tem best 0.7225069448901196 0.54
tem best 0.7240336999163932 0.55
tem best 0.7254479494687218 0.56
tem best 0.7260479476582689 0.73
tem best 0.7268045244295877 0.75
tem best 0.7276709661718181 0.76
best overall threshold : 0.76 best score : 0.7276709661718181
(devenv-anomalydetector) azureuser@gpubox:~/cloudfiles/code/Users/emer.rodriguez/anomalydetector$

```
### Notes from paper

Parameters from paper, page 6, section 5.2 Metrics

| Description  | Parameter | SR/SR-CNN |
|--------------|----------|-----------|
| Shape h_q(f) | q | 3 |
| Number of local average of preceding points | z | 21 |
| Threshold | tau | 3 |
| Number of estimated points | k | 5 |
| Sliding window size | omega | 1440 (KPI), 64 (Yahoo) |

---
Upstream README:

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

Users can run SR by refering sample here

https://github.com/microsoft/anomalydetector/blob/master/main.py
This sample only RUN SR, for SR-CNN please refer the below section. Both SR and SR-CNN use the same evaluation in evaluate.py.

The SR-CNN project is consisted of three major parts.<br> 
1.generate_data.py is used for preprocess the data, where the original continuous time series are splited according to window size and  artificial outliers are injected in proportion. <br> 
`
python generate_data.py --data <dataset>
`<br> 
where dataset is the file name of data folder.If you want to change the default config, you can use the command line args:<br>
`
python generate_data.py -data <dataset> --window 256 --step 128
`<br> 
2.train.py is the network trianing module of SR-CNN. SR transformer is applied on each time-series before training.<br> 
`
python trian.py -data <dataset>
`<br> 
3.evalue.py is the evaluation module.As mentioned in our paper, <br>
`
We evaluate our model from three aspects,accuracy,efficiency and generality.We use precision,recall and F1-score to indicate the  accuracy of our model.In real applications,the human operators do not care about the point-wise metrics. It is acceptable for an algorithm to trigger an alert for any point in a contiguous anomaly segment if the delay is not too long.Thus,we adopt the evaluation  strategy following[23].We mark the whole segment of continuous anomalies as a positive sample which means no matter how many anomalies have been detected in this segment,only one effective detection will be counted.If any point in ananomaly segment can be detected by the algorithm,and the delay of this point is no more than k from the start point of the anomaly segment, we say this segment is detected correctly.Thus,all points in this segment are treated as correct,and the points outside the anomaly segments are treated as normal. 
`<br>
we set different delays to verify whether a whole section of anomalies can be detected in time. For example,  When delay = 7, for an entire segment of anomaly, if the anomaly detector can issue an alarm at its first 7 points, it is considered that the entire segment of anomaly has been successfully detected, otherwise it is considered to have not been detected.<br> 
Run the code:<br>
`
python evalue.py -data <dataset>
`<br> 
