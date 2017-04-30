import pickle
import pandas as pd

from masterthesis.eval_gl.util import cluster_into_spots
from masterthesis.eval_gl.util import get_test_trajs
from masterthesis.eval_gl.util import resolve_endcluster
from masterthesis.models import AverageEnsemble
from datetime import timedelta


def prepare1(paths_csv):
    dfs = list()
    for path in paths_csv:
        df = pd.read_csv(path, index_col=0)
        df = df.set_index(pd.DatetimeIndex(df.index))
        df['start_date'] = pd.to_datetime(df['start_date'])
        df['end_date'] = pd.to_datetime(df['end_date'])
        dfs.append(df)


    train_data = list()
    test_data = list()
    for df in dfs:
        ts = df['end_date'].iloc[-1] - df['start_date'].iloc[0]
        train_days = int(ts.days * 2/3)
        duration = timedelta(days=train_days)
        train_start = df['start_date'].iloc[0]
        train_end = train_start + duration
        train = df[train_start:train_end]
        test = df[train_end:]

        train_data.append(train)
        test_data.append(test)

    return train_data, test_data


def prepare2(train_dfs, clustering='dbscan', **kwargs):
    clustered_dfs = list()
    for i, df in enumerate(train_dfs):
        df = df.copy()
        if i % 10 == 0:
            print(i)
        if clustering == 'dbscan':
            df = cluster_into_spots(df, kwargs.get('eps'), kwargs.get('levels'), kwargs.get('threshold'))

        clustered_dfs.append(df)

    return clustered_dfs


def prepare3(paths_trajs, test_data):
    test_trajectories = list()
    for path, test in zip(paths_trajs, test_data):
        routes = pickle.load(open(path, 'rb'))
        test_trajs = get_test_trajs(test, routes)
        test_trajectories.append(test_trajs)

    return test_trajectories


def run(train, test_trajs, estimators, **kwargs):
    fitted_estimators = list()
    for estimator in estimators:
        fitted_estimators.append(estimator.fit(train))

    ranks = list()
    for route in test_trajs:
        preds = AverageEnsemble().average_proba([estimator.predict_proba(route[0]) for estimator in fitted_estimators])
        truth = resolve_endcluster(train, route[-1])
        if preds is not None:
            preds.index = preds.index.astype(object)
            prob = preds[preds.index == truth]
            rank = 1
            if len(prob) == 1:
                for pred in preds:
                    if pred > prob.iloc[0]:
                        rank += 1

                ranks.append((rank, len(preds)))
            else:
                ranks.append((None, len(preds)))
        else:
            ranks.append((None, None))



    return ranks
