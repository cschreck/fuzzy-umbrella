"""
Evaluation Process for BayesDepartureTimeEstimator with DBScan clustering on multiple levels.
"""
from masterthesis.models import BayesDepartureTimeEstimator
from masterthesis.preprocessing import DenseDepartureTimes
from masterthesis.eval_1.util import make_df
from masterthesis.eval_1.util import cluster_into_spots

from tpm.data_model import *
from tpm.preprocessing import time_duplication_filter
from tpm.preprocessing import speed_filter_abs
from tpm.util.io import read_geolife

from datetime import timedelta
from datetime import date

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



    ddf = DenseDepartureTimes(0.03)
    train_ddt = ddf.fit_transform(train.copy())
    bdte = BayesDepartureTimeEstimator()
    bdte = bdte.fit(train_ddt)

    counter_prob = 0

    ranks = list()

    for i, row in test.iterrows():
        x = pd.DataFrame(data=[[row['start_lat'], row['start_lon']]], index=[i], columns=['lat', 'lon'])
        pred = bdte.predict_proba(x)
        sorted_pred = sorted(pred.items(), key=operator.itemgetter(1), reverse=True)
        for key in pred.keys():
            time_wa, time_wb = bdte.resolve_start_time_cluster(key)
            dummydate = date(1970, 1, 1)
            # time_wa = (datetime.combine(dummydate,time_wa)-timedelta(hours=1)).time()
            # time_wb = (datetime.combine(dummydate,time_wb)+timedelta(hours=1)).time()
            if len(x.between_time(time_wa, time_wb)) == 1:
                    for j, s_pred in enumerate(sorted_pred):
                        if s_pred[0] == key:
                            if verbose > 1:
                                print('Ranked:', j + 1, 'of total', len(pred), 'predictions')
                            counter_prob += 1




    if verbose > 0:
        print('Average rank/pred', np.mean(ranks, axis=0))
        print('Number of no prob:', counter_prob)
        print('Total predictions:', len(test))

    return ranks, len(test), counter_prob