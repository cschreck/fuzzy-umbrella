from math import atan2
from math import cos
from math import degrees
from math import radians
from math import sin
from math import tan
from math import pi
from math import asin
from math import sqrt

from tpm.data_model import R


from .database import query_neo4j
from gpxpy import geo

import numpy as np


def haversine_distance(p1_lat, p1_lon, p2_lat, p2_lon):
    lat_rad1 = radians(p1_lat)
    lon_rad1 = radians(p1_lon)
    lat_rad2 = radians(p2_lat)
    lon_rad2 = radians(p2_lon)
    return 2 * R * asin(sqrt(
        sin((lat_rad2 - lat_rad1) / 2) ** 2 + cos(lat_rad1) * cos(lat_rad2) * (sin((lon_rad2 - lon_rad1) / 2) ** 2)))


def calc_heading(a, b):
    lat1, lon1, lat2, lon2 = map(radians, [a[0], a[1], b[0], b[1]])

    bearing = atan2(cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(lon2 - lon1), sin(lon2 - lon1) * cos(lat2))
    bearing = degrees(bearing)
    bearing = (bearing + 360) % 360

    return bearing


def get_pdist_matrix(results=None):
    query = """
    MATCH (n:Spot)
    RETURN
        n.latitude AS latitude,
        n.longitude AS longitude,
        n.spotID as spotID
    """
    if not results:
        results = query_neo4j(query)

    distances = dict()
    for i in results:
        distance_to_i = dict()
        p1_lat = i['latitude']
        p1_lon = i['longitude']
        for j in results:
            p2_lat = j['latitude']
            p2_lon = j['longitude']
            distance = geo.distance(p1_lat, p1_lon, None, p2_lat, p2_lon, None)
            distance_to_i[j['spotID']] = distance

        distances[i['spotID']] = distance_to_i
    return distances


def time_to_degree(time):
    return ((time.hour + (time.minute + (time.second/60))/60)/24) * 360


def time_distance(t1, t2):
    circumference = 2 * np.pi
    return (np.abs(time_to_degree(t1) - time_to_degree(t2))) * (circumference/360)


def calculate_polygon(a, b, alpha):
    delta_one = calc_heading(a, b) - alpha
    delta_two = calc_heading(a, b) + alpha

    m_one = tan(delta_one + (pi / 2))
    m_two = tan(delta_two + (pi / 2))

    c_x = (m_one * a[0] - m_two * b[0] - a[1] + b[1]) / (m_one - m_two)
    c_y = m_one * (c_x - a[0]) + a[1]

    d_x = (m_two * a[0] - m_one * b[0] - a[1] + b[1]) / (m_two - m_one)
    d_y = m_two * (d_x - a[0]) + a[1]

    return (c_x, c_y), (d_x, d_y)


def is_in_rectangle(rec, p):
    # divde into two triangles
    x = p[0]
    y = p[1]

    x1 = rec[0][0]
    x2 = rec[1][0]
    x3 = rec[2][0]
    x4 = rec[3][0]
    y1 = rec[0][1]
    y2 = rec[1][1]
    y3 = rec[2][1]
    y4 = rec[3][1]

    #calcs for first triangle
    a_1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
    b_1 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
    c_1 = 1 - a_1 - b_1

    #calcs for second triangle
    a_2 = ((y4 - y3) * (x - x3) + (x3 - x4) * (y - y3)) / ((y4 - y3) * (x1 - x3) + (x3 - x4) * (y1 - y3))
    b_2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / ((y4 - y3) * (x1 - x3) + (x3 - x4) * (y1 - y3))
    c_2 = 1 - a_2 - b_2

    #check if in either triangle
    in_first_triangle = 0 <= a_1 <= 1 and 0 <= b_1 <= 1 and 0 <= c_1 <= 1
    in_second_triangle = 0 <= a_2 <= 1 and 0 <= b_2 <= 1 and 0 <= c_2 <= 1

    return in_first_triangle or in_second_triangle


def add_dist(p1,p2,dist):
    radius = R
    bearing = radians(-1 * calc_heading(p1,p2)) - radians(90)


    lat1 = radians(p1[0]) #Current lat point converted to radians
    lon1 = radians(p1[1]) #Current long point converted to radians

    lat2 = asin(sin(lat1)*cos(dist/radius) +
            cos(lat1)*sin(dist/radius)*cos(bearing))

    lon2 = lon1 + atan2(sin(bearing)*sin(dist/radius)*cos(lat1),
                        cos(dist/radius)-sin(lat1)*sin(lat2))

    lat2 = degrees(lat2)
    lon2 = degrees(lon2)

    return lat2, lon2
