from math import *
from tpm.data_model import R
from tpm.data_model import Point



def haversine_distance(p1_lat, p1_lon, p2_lat, p2_lon):
    lat_rad1 = radians(p1_lat)
    lon_rad1 = radians(p1_lon)
    lat_rad2 = radians(p2_lat)
    lon_rad2 = radians(p2_lon)
    return 2 * R * asin(sqrt(sin((lat_rad2 - lat_rad1) / 2) ** 2 + cos(lat_rad1) * cos(lat_rad2) * (
        sin((lon_rad2 - lon_rad1) / 2) ** 2)))

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





def calc_heading(a, b):
    lat1 = radians(a.lat)
    lon1 = radians(a.lon)
    lat2 = radians(b.lat)
    lon2 = radians(b.lon)

    bearing = atan2(cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(lon2 - lon1), sin(lon2 - lon1) * cos(lat2))
    bearing = degrees(bearing)
    bearing = (bearing + 360) % 360

    return bearing




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