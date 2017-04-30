from sklearn.cluster import dbscan
from sklearn.neighbors import DistanceMetric

from tpm.util.dist import haversine_distance
from tpm.data_model import R
from tpm.data_model import Point

from collections import Counter

import numpy as np
import pandas as pd


def staypoints_geolife(traj):
    time_thresh = 30 * 60
    dist_thresh = 250

    staypoints = list()
    i, i_max = 0, len(traj)
    while i < i_max:
        j = i + 1
        token = 0
        while j < i_max:
            dist = haversine_distance(traj[i], traj[j])
            if dist > dist_thresh:
                delta_time = traj[j].datetime - traj[i].datetime
                if delta_time.total_seconds() > time_thresh:
                    mean_point = np.mean([[p.lat, p.lon] for p in traj[i:j + 1]], axis=0)
                    arrival_time = traj[i].datetime
                    leave_time = traj[j].datetime
                    staypoints.append([mean_point, arrival_time, leave_time, i, j])
                    i = j
                    token = 1
                break
            j = j + 1
        if not token == 1:
            i = i + 1

    return staypoints


def make_df(trajs):
    data = list()
    for traj in trajs:
        if haversine_distance(traj[0], traj[-1]) < 100:
            continue

        fp = traj[0]
        sps = staypoints_geolife(traj)
        lp = traj[-1]

        if len(sps) > 1:
            data.append([fp.lat, fp.lon, fp.datetime, sps[0][0][0], sps[0][0][1], sps[0][1]])
            for i in range(1, len(sps) - 1):
                data.append([sps[i][0][0], sps[i][0][1], sps[i][1], sps[i + 1][0][0], sps[i + 1][0][1], sps[i + 1][2]])
            data.append([sps[-1][0][0], sps[-1][0][1], sps[-1][2], lp.lat, lp.lon, lp.datetime])
        else:
            data.append([fp.lat, fp.lon, fp.datetime, lp.lat, lp.lon, lp.datetime])

    df = pd.DataFrame(data, columns=['start_lat', 'start_lon', 'start_date', 'end_lat', 'end_lon', 'end_date'])
    df = df.set_index(pd.DatetimeIndex(df['start_date'])).sort_index()
    return df


def get_test_trajs(test, trajs):
    test_trajs = list()
    for i, row in test.iterrows():
        start = row['start_date']
        end = row['end_date']
        for traj in trajs:
            start_traj = None
            end_traj = None
            for i, p in enumerate(traj):
                if p.datetime == start:
                    start_traj = i

                if p.datetime == end:
                    end_traj = i + 1

            if start_traj is not None and end_traj is not None:
                test_traj = traj[start_traj:end_traj]

        test_trajs.append(test_traj)

    return test_trajs


def resolve_endcluster(train, p):
    end_clusters = {Point(lat, lon, None): end for lat, lon, end in
                    zip(train['end_lat'], train['end_lon'], train['end_cluster'])}
    end_cluster = None
    min_dist = 400
    for ec_point in end_clusters:
        dist = haversine_distance(p.lat, p.lon, ec_point.lat, ec_point.lon)
        if dist < min_dist:
            min_dist = dist
            end_cluster = end_clusters[ec_point]

    return end_cluster


def cluster_into_spots(df, init_eps=150, levels=2, threshold=0.1):
    start_points = list()
    end_points = list()
    length = len(df)

    for i in range(length):
        start_points.append([df['start_lat'].iloc[i], df['start_lon'].iloc[i]])
        end_points.append([df['end_lat'].iloc[i], df['end_lon'].iloc[i]])

    points = np.radians(np.vstack([start_points, end_points]))

    haversine = DistanceMetric.get_metric('haversine')
    dist = haversine.pairwise(points) * R

    clusters = dbscan(dist, metric='precomputed', min_samples=1, eps=init_eps)[1]
    clusters = np.array(clusters, dtype=np.object)

    for _ in range(levels):
        init_eps = init_eps * 0.5
        counts = dict(Counter(clusters))
        for key in counts:
            if counts[key] > threshold * length:
                idxs = np.where(clusters == key)[0]
                dist = haversine.pairwise(points[idxs]) * R
                inner_clusters = dbscan(dist, metric='precomputed', min_samples=1, eps=init_eps)[1]
                for i, idx in enumerate(idxs):
                    clusters[idx] = "{}_{}".format(clusters[idx], inner_clusters[i])

    start_clusters = list()
    end_clusters = list()
    for i, cluster in enumerate(clusters):
        if i < length:
            start_clusters.append(clusters[i])
        else:
            end_clusters.append(clusters[i % length + length])

    df['start_cluster'] = start_clusters
    df['end_cluster'] = end_clusters
    return df
