from sklearn.cluster import dbscan
from sklearn.neighbors import DistanceMetric

from tpm.data_model import R
from tpm.data_model import Point

from collections import Counter

import numpy as np
import pandas as pd

from masterthesis.util import haversine_distance



def make_df(trajs):
    data = list()
    for traj in trajs:
        fp = traj[0]
        lp = traj[-1]

        data.append([fp.lat, fp.lon, fp.datetime, lp.lat, lp.lon, lp.datetime])

    df = pd.DataFrame(data, columns=['start_lat', 'start_lon', 'start_date', 'end_lat', 'end_lon', 'end_date'])
    df = df.set_index(pd.DatetimeIndex(df['start_date'])).sort_index()
    return df


def accuracyN(results, N=1):
    total = len(results)
    correct = 0
    for res in results:
        if res[0] is None:
            continue
        if res[0] <= N:
            correct += 1

    return correct/total


def avg_percentile_rank(results, POIs):
    percentile_ranks = list()
    for res in results:
        if res[0] is not None:
            percentile_ranks.append((POIs - res[0] + 1)/(POIs))
        else:
            percentile_ranks.append(0)

    return sum(percentile_ranks)/len(percentile_ranks)


def get_test_start_points(test, trajs):
    test_start_points = list()
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

        test_start_points.append(test_traj[0])

    return test_start_points


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
