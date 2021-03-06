import datetime
import json
import time
import glob
import numpy as np

from .util import calc_heading

from tpm.data_model import Point, Trajectory
from os.path import splitext
from os.path import basename
from gpxdata import Document
from gpxpy import geo, parse

WALKING_SPEED = 1.4
BICYCLE_SPEED = 4.2
CAR_SPEED = 12.5
SAMPLING_RATE = 3


def create_car_json(file, user_id, start_time, route_id, wrong_order=True):
    speed = CAR_SPEED + np.random.normal(0, 2)
    sampling_rate = 1
    return create_route_json_from_file(
        file,
        speed,
        sampling_rate,
        wrong_order,
        start_time,
        user_id,
        route_id
    )


def create_walking_json(file, user_id, start_time, route_id, wrong_order=True):
    speed = WALKING_SPEED + np.random.normal(0, 0.3)
    return create_route_json_from_file(
        file,
        speed,
        SAMPLING_RATE,
        wrong_order,
        start_time,
        user_id,
        route_id
    )


def create_bicycle_json(file, user_id, start_time, route_id, wrong_order=True):
    speed = BICYCLE_SPEED + np.random.normal(0, 1)
    return create_route_json_from_file(
        file,
        speed,
        SAMPLING_RATE,
        wrong_order,
        start_time,
        user_id,
        route_id
    )


def create_route_json_from_file(
        file,
        speed,
        sampling_rate,
        wrong_order,
        start_time,
        user_id,
        route_id
):
    raw_points = file.get_points_data()
    gps_points = create_gps_points(
        raw_points,
        speed,
        sampling_rate,
        wrong_order
    )

    return create_route_json(
        gps_points,
        start_time,
        sampling_rate,
        user_id,
        route_id
    )


def create_gps_points(points, speed, sampling_rate, wrong_order, starttime):
    new_points = []
    tdsr = datetime.timedelta(seconds=sampling_rate)
    timecounter = starttime

    for p1, p2 in zip(points[:-1], points[1:]):
        if not wrong_order:
            p1_lat = p1.point.latitude
            p1_lon = p1.point.longitude

            p2_lat = p2.point.latitude
            p2_lon = p2.point.longitude
        else:
            p1_lat = p1.point.longitude
            p1_lon = p1.point.latitude

            p2_lat = p2.point.longitude
            p2_lon = p2.point.latitude

        distance = geo.distance(p1_lat, p1_lon, None, p2_lat, p2_lon, None)

        numb_of_points = int(distance / (speed * sampling_rate))
        new_points.append(Point(p1_lat, p1_lon, timecounter))

        if numb_of_points == 0:
            timecounter = timecounter + tdsr
            continue

        if p1_lat > p2_lat:
            if p1_lon > p2_lon:
                diff_lat = abs(p1_lat - p2_lat)
                diff_lon = abs(p1_lon - p2_lon)

                for i in range(numb_of_points + 1):
                    part_lat = (diff_lat / numb_of_points) * i
                    new_lat = p1_lat - part_lat

                    part_lon = (diff_lon / numb_of_points) * i
                    new_lon = p1_lon - part_lon

                    new_points.append(Point(new_lat, new_lon, timecounter))
                    timecounter = timecounter + tdsr
            else:
                diff_lat = abs(p1_lat - p2_lat)
                diff_lon = abs(p2_lon - p1_lon)

                for i in range(numb_of_points + 1):
                    part_lat = (diff_lat / numb_of_points) * i
                    new_lat = p1_lat - part_lat

                    part_lon = (diff_lon / numb_of_points) * i
                    new_lon = p1_lon + part_lon

                    new_points.append(Point(new_lat, new_lon, timecounter))
                    timecounter = timecounter + tdsr

        else:
            if p1_lon > p2_lon:
                diff_lat = abs(p2_lat - p1_lat)
                diff_lon = abs(p1_lon - p2_lon)

                for i in range(numb_of_points + 1):
                    part_lat = (diff_lat / numb_of_points) * i
                    new_lat = p1_lat + part_lat

                    part_lon = (diff_lon / numb_of_points) * i
                    new_lon = p1_lon - part_lon

                    new_points.append(Point(new_lat, new_lon, timecounter))
                    timecounter = timecounter + tdsr

            else:
                diff_lat = abs(p2_lat - p1_lat)
                diff_lon = abs(p2_lon - p1_lon)

                for i in range(numb_of_points + 1):
                    part_lat = (diff_lat / numb_of_points) * i
                    new_lat = p1_lat + part_lat

                    part_lon = (diff_lon / numb_of_points) * i
                    new_lon = p1_lon + part_lon

                    new_points.append(Point(new_lat, new_lon, timecounter))
                    timecounter = timecounter + tdsr

        timecounter = timecounter + tdsr

    return new_points


def create_route_json(points, start_time, sample_rate, user_id, route_id):
    sample_rate_delta = datetime.timedelta(seconds=sample_rate)
    route_points = []

    for i, j in zip(range(0, len(points) - 1), range(1, len(points))):
        data_id = i
        lat = points[i][0]
        lon = points[i][1]
        timey = start_time + (sample_rate_delta * i)
        timey = int(time.mktime(timey.timetuple())) * 1000
        head = calc_heading(points[i], points[j])
        dist_to_next = geo.distance(
            points[i][0], points[i][1], None,
            points[j][0], points[j][1], None
        )

        x = {
            "dataID": data_id,
            "latitude": lat,
            "longitude": lon,
            "time": timey,
            "userID": user_id,
            "routeID": route_id,
            "head": head,
            "speed": dist_to_next / sample_rate,
            "distanceToNextPoint": dist_to_next,
            "timediffToNextPoint": (sample_rate * 1000),
            "leadsTo": 0,
            "mappedToSpot": False,
            "spot": None,
            "closestSpotInfo": None,
            "pointInfoDBSAN": None,
            "processedDBSCAN": False,
            "clusterDBSCAN": None
        }

        route_points.append(x)

    trajectory = {"trajectory": [point for point in route_points]}

    return json.dumps(trajectory) \
        .replace("False", "false") \
        .replace("None", "null") \
        .replace("]}", '],"user":' + user_id + ',"routeID": ' +
                 str(route_id)+', "spotProcessed":false}')


def transform_kmls_to_gpxs(directory):
    kml_files = glob.glob("{}*.kml".format(directory))
    for kml_file in kml_files:
        file_name = splitext(basename(kml_file))
        doc = None
        with open(kml_file) as in_f:
            doc = Document.readKML(in_f)

        with open("{}{}.gpx".format(directory, file_name[0]), 'w') as out:
            Document.writeGPX(doc, out)


def read_gpxs(directory):
    gpx_filepaths = glob.glob("{}*.gpx".format(directory))
    gpx_files = {}
    for gpx_filepath in gpx_filepaths:
        with open(gpx_filepath) as f:
            gpx_file = parse(xml_or_file=f)
            gpx_files[splitext(basename(gpx_filepath))[0]] = gpx_file

    return gpx_files


def create_bicycle_traj(file, uiddummy, starttime, riddummy):
    raw_points = file.get_points_data()
    traj = Trajectory(create_gps_points(raw_points, BICYCLE_SPEED, SAMPLING_RATE, True, starttime))
    return traj


def create_car_traj(file, uiddummy, starttime, riddummy):
    raw_points = file.get_points_data()
    traj = Trajectory(create_gps_points(raw_points, CAR_SPEED, SAMPLING_RATE, True, starttime))
    return traj


def create_walking_traj(file, uiddummy, starttime, riddummy):
    raw_points = file.get_points_data()
    traj = Trajectory(create_gps_points(raw_points, WALKING_SPEED, SAMPLING_RATE, True, starttime))
    return traj

