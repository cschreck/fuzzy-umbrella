import pickle
import pandas as pd

from masterthesis.models import FrequentistEstimator
from masterthesis.eval_ad.util import cluster_into_spots
from masterthesis.eval_ad.util import make_df
from datetime import timedelta




def prepare(path, clustering='dbscan', **kwargs):
    routes = pickle.load(open(path, 'rb'))
    df = make_df(routes)
    if clustering == 'dbscan':
        df = cluster_into_spots(df, kwargs.get('eps'), 0, 0)

    return df


def run(df, estimator=FrequentistEstimator(), verbose=1):
    ts = df['end_date'].iloc[-1] - df['start_date'].iloc[0]
    train_days = int(ts.days * 2/3)
    duration = timedelta(days=train_days)
    train_start = df['start_date'].iloc[0]
    train_end = train_start + duration
    train = df[train_start:train_end]
    test = df[train_end:]
    if verbose > 1:
        print(train_start, train_end)

    estimator = estimator.fit(train)

    counter_no_prob = 0

    ranks = list()

    for i, row in test.iterrows():
        x = pd.DataFrame(data=[[row['start_lat'], row['start_lon']]], index=[i], columns=['lat', 'lon'])
        if verbose > 2:
            print(x)
        pred = estimator.predict_proba(x)

        cp = (row['start_cluster'], row['end_cluster'])

        if cp in pred.keys():
            prob = pred[cp]
            rank = 1

            for key in pred.keys():
                if key == cp:
                    continue
                if pred[key] > prob:
                    rank += 1

            ranks.append((rank, len(pred), prob))
        else:
            print(row['start_cluster'], row['end_cluster'])
            if verbose > 2:
                print("no prob")
            ranks.append((None, len(pred), None))
            counter_no_prob += 1

    if verbose > 0:
        #print('Average rank/pred', np.mean(ranks, axis=0))
        print('Number of no prob:', counter_no_prob)
        print('Total predictions:', len(test))

    return ranks, len(test), counter_no_prob