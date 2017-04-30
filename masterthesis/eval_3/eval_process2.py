"""
Evaluation Process for FrequentistEstimator with K Means clustering
"""
from masterthesis.models import HeadingEstimator, HeadingMostFrequentTargetEstimator, KnownStartEstimator
from masterthesis.eval_1.util import make_df
from masterthesis.eval_1.util import cluster_into_spots
from masterthesis.eval_3.util import get_test_trajs, resolve_endcluster

from tpm.data_model import *
from tpm.preprocessing import time_duplication_filter
from tpm.preprocessing import speed_filter_abs
from tpm.util.io import read_geolife

from datetime import timedelta

import pandas as pd
import numpy as np
import operator


def run(path, test_traj_len, predict_after_frac=100, verbose=1):
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
    df = cluster_into_spots(df, init_eps=300, levels=0, threshold=0.2)

    if verbose > 1:
        print('Trajectories clustered in sps')

    ts = df['end_date'].iloc[-1] - df['start_date'].iloc[0]
    train_weeks = int((ts.days / 7) * 4/5)
    duration = timedelta(days=7 * train_weeks)
    train_start = df['start_date'].iloc[0]
    train_end = train_start + duration
    train = df[train_start:train_end]
    test = df[train_end:]

    test_trajs = get_test_trajs(test, preprocessed)

    kse = KnownStartEstimator()
    kse = kse.fit(train)

    filtered_test_trajs = list()
    for traj in test_trajs:
        if len(traj) >= test_traj_len:
            filtered_test_trajs.append(traj)

    if verbose > 1:
        print(len(filtered_test_trajs), "of", len(test_trajs), "have the minimal length of", test_traj_len)


    res = dict()
    for traj in filtered_test_trajs:

        res[traj[-1]] = kse.predict_proba(traj[:predict_after_frac])



    ranks = list()
    for p in res:
        predictions = res[p]
        truth = resolve_endcluster(train, p)
        if predictions is None or len(predictions) < 1:
            ranks.append((None, None))
        else:
            ranked = False
            for i in range(len(predictions)):
                if predictions.index[i] == truth:
                    ranks.append((i+1, len(predictions)))
                    ranked = True
                    break

            if not ranked:
                ranks.append((None, len(predictions)))


    return ranks



