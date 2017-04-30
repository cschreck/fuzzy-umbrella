from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.cluster import DBSCAN
from datetime import timedelta
import pandas as pd

from gpxpy import geo

import numpy as np


class FuzzySpotTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, eps=30):
        self.eps = eps

    def fit(self, X, y=None, **fitparams):
        self.spot_ids_, self.dist_matrix_ = self.calc_pdist_matrix(X)
        dbscan = DBSCAN(eps=self.eps, min_samples=1, metric='precomputed')
        self.clusters_ = dbscan.fit_predict(self.dist_matrix_)
        return self

    def transform(self, X):
        start_clusters = list()
        end_clusters = list()

        for index, row in X.iterrows():
            start_clusters.append(self.clusters_[self.spot_ids_.index(row['start_spot'])])
            end_clusters.append(self.clusters_[self.spot_ids_.index(row['end_spot'])])

        X['start_cluster'] = start_clusters
        X['end_cluster'] = end_clusters
        return X

    def calc_pdist_matrix(self, X):
        spots = np.union1d(X['start_spot'].values, X['end_spot'].values)

        dist_matrix = list()
        for spot1 in spots:
            inner_matrix = list()

            if len(X[X['start_spot'] == spot1]) > 0:
                p1_ex = X[X['start_spot'] == spot1].iloc[0]
                p1_lat = float(p1_ex.start_lat)
                p1_lon = float(p1_ex.start_lon)
            else:
                p1_ex = X[X['end_spot'] == spot1].iloc[0]
                p1_lat = float(p1_ex.end_lat)
                p1_lon = float(p1_ex.end_lon)
            for spot2 in spots:
                if len(X[X['start_spot'] == spot2]) > 0:
                    p2_ex = X[X['start_spot'] == spot2].iloc[0]
                    p2_lat = float(p2_ex.start_lat)
                    p2_lon = float(p2_ex.start_lon)
                else:
                    p2_ex = X[X['end_spot'] == spot2].iloc[0]
                    p2_lat = float(p2_ex.end_lat)
                    p2_lon = float(p2_ex.end_lon)

                dist = geo.distance(p1_lat, p1_lon, None, p2_lat, p2_lon, None)
                inner_matrix.append(dist)

            dist_matrix.append(inner_matrix)

        return spots.tolist(), dist_matrix


class DenseDepartureTimes(BaseEstimator, TransformerMixin):
    def __init__(self, eps=15, look_ahead=timedelta(hours=1)):
        self.eps = eps
        self.look_ahead = look_ahead

    def fit(self, X, y=None, **fitparams):
        return self

    def transform(self, X):
        one_second = timedelta(seconds=1)
        look_ahead_data = list()
        idx_insert = list()
        X.reset_index(drop=True, inplace=True)

        if self.look_ahead:
            for i in range(len(X)):
                for j in range(i + 1, len(X)):
                    if X.iloc[j]['start_date'] - X.iloc[i]['end_date'] < self.look_ahead:
                        look_ahead_data.append([X.iloc[i]['start_lat'],
                                                X.iloc[i]['start_lon'],
                                                X.iloc[i]['start_date'] + one_second,
                                                X.iloc[j]['end_lat'],
                                                X.iloc[j]['end_lon'],
                                                X.iloc[j]['end_date'],
                                                X.iloc[i]['start_cluster'],
                                                X.iloc[j]['end_cluster'],
                                                ])

                    else:
                        break

        if len(look_ahead_data) > 0:
            X = pd.concat([X, pd.DataFrame(look_ahead_data, columns=X.columns.values)], ignore_index=True)

        X = pd.DataFrame(data=X, columns=X.columns.values)
        X = X.set_index(pd.DatetimeIndex(X['start_date'])).sort_index()

        dbscan = DBSCAN(eps=self.eps, min_samples=1, metric='precomputed')
        start_cluster_to_time = dict()
        for key, group in X.groupby(['start_cluster', 'end_cluster']):
            dist = self._calc_pdist_matrix(group)
            clusters = dbscan.fit_predict(dist)
            d = {timestamp: cluster for timestamp, cluster in zip(group.index, clusters)}
            start_cluster_to_time[key] = d

        start_time_cluster = list()
        for index, row in X.iterrows():
            cluster_pair = (row['start_cluster'], row['end_cluster'])
            start_time_cluster.append("{}___{}".format(cluster_pair, start_cluster_to_time[cluster_pair][index]))

        X['start_time_cluster'] = start_time_cluster
        return X

    def _time_to_minutes(self, time):
        return time.hour * 60 + time.minute + time.second / 60

    def _time_distance(self, t1, t2):
        diff = np.abs(self._time_to_minutes(t1) - self._time_to_minutes(t2))
        if diff > 720:
            diff = 1440 - diff
        return diff

    def _calc_pdist_matrix(self, group):
        matrix = list()

        for t1 in group.index.time:
            inner_dist = list()

            for t2 in group.index.time:
                inner_dist.append(self._time_distance(t1, t2))

            matrix.append(inner_dist)

        return matrix