"""
Evaluation Process for FrequentistEstimator with DBScan clustering on multiple levels.
"""
from masterthesis.models import FrequentistEstimator
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

    print(dates)

    bwe = FrequentistEstimator()

    bwe = bwe.fit(train)

    print(train)


    print(test)

    counter_no_prob = 0

    ranks = list()

    for date in dates:
        month, day = date
        x = pd.DataFrame(data=[[49.475752, 8.482531]],
                         index=pd.DatetimeIndex([pd.Timestamp("2008-{}-{} 19:45:21".format(month, day))]),
                         columns=['lat', 'lon'])
        if verbose > 2:
            print(x)
        pred = bwe.predict_proba(x)

        sorted_pred = sorted(pred.items(), key=operator.itemgetter(1), reverse=True)



        for i, row in df.loc[(df.index.month == month) & (df.index.day == day)].iterrows():
            if verbose > 2:
                print((row['start_cluster'], row['end_cluster']))
            if (row['start_cluster'], row['end_cluster']) in pred.keys():
                if verbose > 2:
                    print(pred[(row['start_cluster'], row['end_cluster'])])
                for j, s_pred in enumerate(sorted_pred):
                    if s_pred[0] == (row['start_cluster'], row['end_cluster']):
                        if verbose > 2:
                            print('Ranked:', j + 1, 'of total', len(pred), 'predictions')
                        ranks.append((j+1, len(pred)))
            else:
                if verbose > 2:
                    print("no prob")
                counter_no_prob += 1

    if verbose > 0:
        print('Average rank/pred', np.mean(ranks, axis=0))
        print('Number of no prob:', counter_no_prob)
        print('Total predictions:', len(test))

    return ranks, len(test), counter_no_prob