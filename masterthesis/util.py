from math import atan2
from math import cos
from math import degrees
from math import radians
from math import sin

from .database import query_neo4j
from gpxpy import geo


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
