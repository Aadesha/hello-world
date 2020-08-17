#!/usr/bin/env python3 -u

from sktime.classification.interval_based import TimeSeriesForest
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
# from sktime.utils.time_series import time_series_slope
import pandas as pd
import os
import numpy as np
import time

# Load data
data_path = '/Users/aa25desh/Univariate_ts'


def load_data(file_path):
    with open(file_path) as fl:
        for line in fl:
            if line.strip():
                if "@data" in line.lower():
                    break

        df = pd.read_csv(fl, delimiter=',', header=None)
        y = df.pop(df.shape[1] - 1)
        X = pd.DataFrame([[row] for _, row in df.iterrows()])  # transform into nested pandas dataframe
    return X, y

datasets = [
    #"ACSF1",
    "Adiac",
    # "AllGestureWiimoteX",
    # "AllGestureWiimoteY",
    # "AllGestureWiimoteZ",
    "ArrowHead",
    "Coffee",
    "Beef",
    "BeetleFly",
    "BirdChicken",
    "BME",
    "Car",
    "CBF",
    "Chinatown",
    "ChlorineConcentration",
    "CinCECGTorso",
    "Computers",
    "CricketX",
    "CricketY",
    "CricketZ",
    "Crop",
    "DiatomSizeReduction",
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxTW",
    # "DodgerLoopDay",
    # "DodgerLoopGame",
    # "DodgerLoopWeekend",
    "Earthquakes",
    "ECG200",
    "ECG5000",
    "ECGFiveDays",
    "ElectricDevices",
    "EOGHorizontalSignal",
    "EOGVerticalSignal",
    "EthanolLevel",
    "FaceAll",
    "FaceFour",
    "FacesUCR",
    "FiftyWords",
    "Fish",
    "FordA",
    "FordB",
    "FreezerRegularTrain",
    "FreezerSmallTrain",
    "Fungi",
    # "GestureMidAirD1",
    # "GestureMidAirD2",
    # "GestureMidAirD3",
    # "GesturePebbleZ1",
    # "GesturePebbleZ2",
    "Ham",
    #"HandOutlines",
    "Haptics",
    "Herring",
    "InlineSkate",
    "InsectEPGRegularTrain",
    "InsectEPGSmallTrain",
    "InsectWingbeatSound",
    "ItalyPowerDemand",
    "LargeKitchenAppliances",
    "Lightning2",
    "Lightning7",
    "Mallat",
    "Meat",
    "MedicalImages",
    # "MelbournePedestrian",
    "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxTW",
    "MixedShapesRegularTrain",
    "MixedShapesSmallTrain",
    "MoteStrain",
    "NonInvasiveFetalECGThorax1",
    "NonInvasiveFetalECGThorax2",
    "OliveOil",
    "OSULeaf",
    "PhalangesOutlinesCorrect",
    "Phoneme",
    # "PickupGestureWiimoteZ",
    "PigAirwayPressure",
    "PigArtPressure",
    "PigCVP",
    # "PLAID",
    "Plane",
    "PowerCons",
    "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxTW",
    "RefrigerationDevices",
    #"Rock",
    "ScreenType",
    "SemgHandGenderCh2",
    "SemgHandMovementCh2",
    "SemgHandSubjectCh2",
    # "ShakeGestureWiimoteZ",
    "ShapeletSim",
    "ShapesAll",
    "SmallKitchenAppliances",
    "SmoothSubspace",
    "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2",
    # "StarlightCurves",
    "Strawberry",
    "SwedishLeaf",
    "Symbols",
    "SyntheticControl",
    "ToeSegmentation1",
    "ToeSegmentation2",
    "Trace",
    "TwoLeadECG",
    "TwoPatterns",
    "UMD",
    "UWaveGestureLibraryAll",
    "UWaveGestureLibraryX",
    "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ",
    "Wafer",
    "Wine",
    "WordSynonyms",
    "Worms",
    "WormsTwoClass",
    "Yoga"
]


n_datasets = len(datasets)

"""
# Read in list of smaller time-series classification datasets
with open('pigs.txt', 'r') as f:
    datasets = [line.strip('\n') for line in f.readlines()]
n_datasets = len(datasets)

# Define param grid
n_estimators_list = [200]
features_list = [
    [np.mean, np.std, time_series_slope],
    [np.mean, np.std, time_series_slope, skew],
    [np.mean, np.std, time_series_slope, skew, kurtosis],
    [np.mean, np.std, time_series_slope, kurtosis]
]
n_intervals_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 'log', 'sqrt']
param_grid = {
    'n_estimators': n_estimators_list,
    'base_estimator__transform__n_intervals': n_intervals_list,
    'base_estimator__transform__features': features_list
}
cv = StratifiedKFold(n_splits=10)
"""

fname = open("results.txt", "w+")
fname.write('dataset,predict_bench,fit_bench,Accuracy\n')

# Run the fit and predict
for i, dataset in enumerate(datasets):

    print(f'Dataset: {i + 1}/{n_datasets} {dataset}')

    # pre-allocate results
    results = np.zeros(3)

    # load data
    train_file = os.path.join(data_path, f'{dataset}/{dataset}_TRAIN.ts')
    test_file = os.path.join(data_path, f'{dataset}/{dataset}_TEST.ts')

    x_train, y_train = load_data(train_file)
    x_test, y_test = load_data(test_file)

    tsf = TimeSeriesForest()

    #  clf = GridSearchCV(tsf, param_grid, scoring='neg_log_loss', cv=cv, refit=True, iid=False, error_score='raise',n_jobs=-1)

    # tune when enough samples for all classes are available
    try:
        s = time.time()
        tsf.fit(x_train, y_train)
        results[0] = time.time() - s

    # otherwise except errors in CV due to class imbalances and run un-tuned time series forest classifier
    except (ValueError, IndexError):
        clf = TimeSeriesForest(n_jobs=-1)
        s = time.time()
        tsf.fit(x_train, y_train)
        results[0] = time.time() - s

    # predict
    s = time.time()
    y_pred = tsf.predict(x_test)
    results[1] = time.time() - s

    # score
    results[2] = accuracy_score(y_test, y_pred)

    print('{},{},{},{}\n'.format(dataset, results[0], results[1], results[2]))
    # save results
    fname.write('{},{},{},{}\n'.format(dataset, results[0], results[1], results[2]))

fname.close()
