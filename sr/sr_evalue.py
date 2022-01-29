import argparse
import glob
import logging
import multiprocessing as mp
import os
import pickle
import time


from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
import numpy as np
import pandas as pd

from msanomalydetector import SpectralResidual
from msanomalydetector import THRESHOLD, MAG_WINDOW, SCORE_WINDOW, DetectMode
from srcnn.competition_metric import evaluate_for_all_series

# logging.basicConfig(level=logging.INFO)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def run_spectral_residual(data):
    ts_data, params = data
    detector = SpectralResidual(series=ts_data,
                                sensitivity=99,  # only applies to margin
                                detect_mode=DetectMode.anomaly_only,
                                # threshold=0.1,
                                # mag_window=MAG_WINDOW,
                                # score_window=SCORE_WINDOW,
                                # batch_size=-1
                                **params)
    ts_result = detector.detect()

    result = pd.concat([ts_data.loc[:, ["timestamp", "label"]],
                        ts_result.loc[:, ["isAnomaly", "score"]]], axis=1).reset_index(drop=True)
    result.timestamp = result.timestamp.astype("datetime64").view("uint64")
    result.isAnomaly = result.isAnomaly.astype("int64")
    result = [result[col].to_list() for col in result.columns]

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SR Evaluate')
    parser.add_argument('--csv-input-dir', type=str, required=True, help='Dataset CSV input dir')
    parser.add_argument('--parallel', action='store_true', help='Run model in parallel')
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.csv_input_dir)

    dataset = [pd.read_csv(ts_filepath) for ts_filepath in glob.glob(f"{dataset_dir}/*.csv")]
    trials = Trials()

    def objective(hyperopt_params):
        run_parallel = hyperopt_params.pop("run_parallel")
        t_0 = time.time()
        data_with_params = [(ts_data, hyperopt_params) for ts_data in dataset]

        if run_parallel:
            results = pool.map(run_spectral_residual, data_with_params)

        else:
            results = [run_spectral_residual(ts_data_params) for ts_data_params in data_with_params]

        total_fscore, pre, rec, TP, FP, TN, FN = evaluate_for_all_series(results, delay=7, prt=False)

        print(f"Params: {hyperopt_params}, "
              f"F1: {np.round(total_fscore, 5)}, "
              f"Precision: {np.round(pre, 5)}, "
              f"Recall: {np.round(rec, 5)}, "
              f"Time: {np.round(time.time() - t_0, 2)}s")

        return {'loss': -1.0 * total_fscore, 'status': STATUS_OK}

    space = hp.choice("sr", [
        {
            "threshold": hp.choice("threshold", [0.35, 0.36, 0.37, 0.375, 0.38, 0.40, 0.50]),
            "mag_window": hp.choice("mag_window", [MAG_WINDOW]),
            "score_window": hp.choice("score_window", [-1, SCORE_WINDOW, 9000, 10000, 11000, 11500, 12000, 12500, 15000,
                                                       20000, 50000, 100000]),
            "batch_size": hp.choice("batch_size", [-1]),
            "run_parallel": hp.choice("run_parallel", [args.parallel])
        }
    ])
    if args.parallel:
        pool = mp.Pool(processes=4)

    best = fmin(objective, space, algo=tpe.suggest, max_evals=10, trials=trials, rstate=np.random.default_rng(1))
    # objective({'batch_size': -1, 'mag_window': 3, 'score_window': 10000, 'threshold': 0.375, 'run_parallel': args.parallel}); exit(0)

    trials_fname = "trials.pkl"
    pickle.dump(trials, open(trials_fname, "wb"))
    # trials = pickle.load(open(trials_fname, "rb"))

    trial_rows = []
    for trial in trials.trials:
        param_idxs = {param: value[0] for param, value in trial["misc"]["vals"].items()}
        trial_row = space_eval(space, param_idxs)
        trial_row["loss"] = trial["result"]["loss"]
        trial_rows.append(trial_row)

    trials_frame = pd.DataFrame(trial_rows)
    print(trials_frame.sort_values("loss"))

    # KPI Test
    # Params: {'batch_size': -1, 'mag_window': 3, 'score_window': 10000, 'threshold': 0.375}, F1: 0.67523, Precision: 0.75665, Recall: 0.60962, Time: 9.77s

    # KPI Train
    # Params: {'batch_size': -1, 'mag_window': 3, 'score_window': 10000, 'threshold': 0.375}, F1: 0.66361, Precision: 0.81299, Recall: 0.5606, Time: 10.12s

    # import cProfile
    # from pstats import SortKey
    # cProfile.run("objective({'batch_size': -1, 'mag_window': 3, 'score_window': 40, 'threshold': 0.25})",
    #              sort=SortKey.TIME)
