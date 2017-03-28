"""
Evaluation Process for BayesWeekdayEstimator with DBScan clustering on multiple levels.
"""
from masterthesis.models import TimeSensitiveMostFrequentRoute
from masterthesis.eval_1.util import make_df
from masterthesis.eval_1.util import cluster_into_spots

from tpm.data_model import *
from tpm.preprocessing import time_duplication_filter
from tpm.preprocessing import speed_filter_abs
from tpm.util.io import read_geolife

from datetime import timedelta

import pandas as pd
import numpy as np
import operator


def run(path, verbose=1):
    print(path)
    trajs = read_geolife(path)
    preprocessed = list()
    for traj in trajs:
        traj_new = time_duplication_filter(traj)
        traj_new = speed_filter_abs(traj_new, 300, in_kmh=True)
        preprocessed.append(traj_new)

    if verbose > 1:
        print('Trajectories preprocessed')

    df = make_df(preprocessed)

    if verbose > 1:
        print('Trajectories split at staypoints')

    length = len(df)
    df = cluster_into_spots(df, init_eps=300, levels=1, threshold=0.2)

    if verbose > 1:
        print('Trajectories clustered in sps')

    ts = df['end_date'].iloc[-1] - df['start_date'].iloc[0]
    train_weeks = int((ts.days / 7) * 4/5)
    duration = timedelta(days=7 * train_weeks)
    train_start = df['start_date'].iloc[0]
    train_end = train_start + duration
    train = df[train_start:train_end]
    test = df[train_end:]

    dates = list()
    for i in test.index:
        if len(dates) > 1:
            if dates[-1] == (i.month, i.day):
                continue
        dates.append((i.month, i.day))

    ts = TimeSensitiveMostFrequentRoute()
    ts.fit(train)
    preds = ts.predict_proba(test)
    ranks = list()
    for i in range(len(preds)):
        correct = False

        for j, pred in enumerate(preds[i].index):
            if pred == test['end_cluster'].iloc[i]:
                ranks.append((j + 1, len(preds[i])))
                correct = True
                break

        if not correct:
            ranks.append((None, len(preds[i])))


    if verbose > 0:
        print('Average rank/pred', np.mean(ranks, axis=0))
        print('Total predictions:', len(test))

    return ranks, len(test)
