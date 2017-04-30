import pickle

from masterthesis.eval_ad.util import cluster_into_spots
from masterthesis.eval_ad.util import make_df
from masterthesis.eval_ad.util import get_test_trajs
from masterthesis.eval_ad.util import resolve_endcluster
from masterthesis.models import LocationSensitiveEstimator
from datetime import timedelta


def prepare(path, clustering='dbscan', **kwargs):
    routes = pickle.load(open(path, 'rb'))

    df = make_df(routes)

    if clustering == 'dbscan':
        df = cluster_into_spots(df, kwargs.get('eps'), 0, 0)

    ts = df['end_date'].iloc[-1] - df['start_date'].iloc[0]
    train_days = int(ts.days * 2/3)
    duration = timedelta(days=train_days)
    train_start = df['start_date'].iloc[0]
    train_end = train_start + duration
    train = df[train_start:train_end]
    test = df[train_end:]

    test_trajs = get_test_trajs(test, routes)

    return train, test_trajs


def run(train, test_trajs, estimator=LocationSensitiveEstimator(), **kwargs):
    estimator = estimator.fit(train)

    ranks = list()
    for route in test_trajs:
        preds = estimator.predict_proba(route[0])
        truth = resolve_endcluster(train, route[-1])
        if preds is not None:
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
