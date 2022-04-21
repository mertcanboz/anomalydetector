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
(devenv-anomalydetector)> python srcnn/train.py --data kpi/kpi_train_ts_data --window 1440 --epoch 300 --use-gpu >  srcnn_train_all_gpu.log
# Train Epoch: 110 [0/46298 (0%)] Loss: 21.338150

(devenv-anomalydetector)> python srcnn/evalue.py --data kpi/kpi_test_ts_data  --window 1440 --delay 7 --epoch 110  --use-gpu > srcnn_evalue_all_gpu_110.log
(devenv-anomalydetector)> python srcnn/evalue.py --data kpi/kpi_test_ts_data  --window 1440 --delay 7 --epoch 300  --use-gpu > srcnn_evalue_all_gpu_300.log

```
Trained using Nvidia K80 on Azure ML. 

| Dataset | Model | epoch | `delay` | `window` | `threshold` | F1 Score | Precision | Recall  | Time                                             |
| ------- | ----- |------|---------| ----- |-----------|----------|-----------|---------|--------------------------------------------------|
| KPI (test) | SRCNN | 7 | 110 | 1440 | 0.65 | 0.74906  | 0.75574   | 0.74250 | 55094.58s (train 300 epochs), 21600.68s (evalue) |

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
