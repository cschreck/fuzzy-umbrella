import pickle
import pandas as pd

from masterthesis.eval_ad.util import cluster_into_spots
from masterthesis.eval_ad.util import make_df
from masterthesis.models import BayesDepartureTimeEstimator
from masterthesis.preprocessing import DenseDepartureTimes
from datetime import timedelta


def prepare(path, clustering='dbscan', **kwargs):
    routes = pickle.load(open(path, 'rb'))
    df = make_df(routes)
    if clustering == 'dbscan':
        df = cluster_into_spots(df, kwargs.get('eps'), 0, 0)

    return df


def run(df, verbose=1):
    ts = df['end_date'].iloc[-1] - df['start_date'].iloc[0]
    train_days = int(ts.days * 2/3)
    duration = timedelta(days=train_days)
    train_start = df['start_date'].iloc[0]
    train_end = train_start + duration
    train = df[train_start:train_end]
    test = df[train_end:]

    ddf = DenseDepartureTimes(15)
    train_ddt = ddf.fit_transform(train.copy())
    bdte = BayesDepartureTimeEstimator()
    bdte = bdte.fit(train_ddt)

    counter_prob = 0

    ranks = list()

    for i, row in test.iterrows():
        x = pd.DataFrame(data=[[row['start_lat'], row['start_lon']]], index=[i], columns=['lat', 'lon'])
        pred = bdte.predict_proba(x)
        cp = (row['start_cluster'], row['end_cluster'])
        found = False
        for key in pred.keys():
            time_wa, time_wb = bdte.resolve_start_time_cluster(key)

            if len(x.between_time(time_wa, time_wb)) == 1 and str(cp) == key.split('___')[0]:
                    prob = pred[key]
                    rank = 1
                    for key2 in pred.keys():
                        if key == key2:
                            continue
                        if pred[key2] > prob:
                            rank += 1

                    ranks.append((rank, len(pred), prob))
                    found = True

        if not found:
            counter_prob += 1
            ranks.append((None, len(pred)))

    if verbose > 0:
        print('Number of no prob:', counter_prob)
        print('Total predictions:', len(test))

    return ranks, len(test), counter_prob
