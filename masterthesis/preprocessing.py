from masterthesis.database import get_spot_position
from masterthesis.database import get_start_waypoint_ids
from masterthesis.database import get_gps_ids
from masterthesis.database import get_end_waypoint
from masterthesis.database import get_spot_ids

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.cluster import DBSCAN

from gpxpy import geo

import pandas as pd
import numpy as np


def make_dataset(user_id):
    start_waypoints = get_start_waypoint_ids(user_id)
    start_gps_ids, dates_start = get_gps_ids(start_waypoints)

    start_spot_ids = []
    for res in get_spot_ids(start_gps_ids):
        start_spot_ids.append(res['m']['spotID'])

    start_spot_lat = list()
    start_spot_long = list()
    for start_spot_id in start_spot_ids:
        res = get_spot_position(start_spot_id)
        start_spot_lat.append(res['latitude'])
        start_spot_long.append(res['longitude'])

    end_waypoints = []
    for start_waypoint in start_waypoints:
        end_waypoints.append(get_end_waypoint(start_waypoint))

    end_gps_ids, dates_end = get_gps_ids(end_waypoints)
    end_spot_ids = []
    for res in get_spot_ids(end_gps_ids):
        if res is None:
            end_spot_ids.append(None)
            continue
        end_spot_ids.append(res['m']['spotID'])

    end_spot_lat = list()
    end_spot_long = list()
    for end_spot_id in end_spot_ids:
        res = get_spot_position(end_spot_id)
        end_spot_lat.append(res['latitude'])
        end_spot_long.append(res['longitude'])

    data = np.transpose([start_spot_ids, end_spot_ids, dates_start, dates_end,
                         start_spot_lat, start_spot_long, end_spot_lat, end_spot_long])

    columns = ['start_spot', 'end_spot', 'start_date', 'end_date',
               'start_lat', 'start_lon', 'end_lat', 'end_lon']

    df = pd.DataFrame(data=data, columns=columns)
    df = df.set_index(pd.DatetimeIndex(df['start_date'])).sort_index()

    return df


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

