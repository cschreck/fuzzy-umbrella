from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.cluster import DBSCAN

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
    def __init__(self, eps=0.5):
        self.eps = eps

    def fit(self, X, y=None, **fitparams):
        return self

    def transform(self, X):
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
            start_time_cluster.append("{}_{}".format(cluster_pair, start_cluster_to_time[cluster_pair][index]))

        X['start_time_cluster'] = start_time_cluster
        return X

    def _time_to_degree(self, time):
        return ((time.hour + (time.minute + (time.second / 60)) / 60) / 24) * 360

    def _time_distance(self, t1, t2):
        circumference = 2 * np.pi
        return (np.abs(self._time_to_degree(t1) - self._time_to_degree(t2))) * (circumference / 360)

    def _calc_pdist_matrix(self, group):
        matrix = list()

        for t1 in group.index.time:
            inner_dist = list()

            for t2 in group.index.time:
                inner_dist.append(self._time_distance(t1, t2))

            matrix.append(inner_dist)

        return matrix