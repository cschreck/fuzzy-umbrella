import json

from masterthesis import config

from requests import post
from urllib.parse import urlencode


SINGLE_ROUTE_ENDPOINT = config.SINGLE_ROUTE_ENDPOINT
SEND_QUERY_ENDPOINT = config.SEND_QUERY_ENDPOINT
APPLICATION_JSON_HEADER = {'content-type': 'application/json'}


def upload_route(json_content, retries=5):
    counter = 0
    while counter < retries:
        request = post(
            url=SINGLE_ROUTE_ENDPOINT,
            data=json_content,
            headers=APPLICATION_JSON_HEADER
        )

        if request.content.decode('utf-8') == 'Route processed':
            return True

        counter += 1

    return False


def query_neo4j(query):
    params = {'query': query}
    parsed_query = urlencode(params)

    request = post(
        url="{}?{}".format(SEND_QUERY_ENDPOINT, parsed_query),
        headers=APPLICATION_JSON_HEADER
    )

    return json.loads(request.content.decode('utf-8'))


def get_start_waypoint_ids(user_id):
    query = """
    MATCH (n:User)-[:ROUTE_START]->(m:Waypoint)
    WHERE n.username = '{}'
    RETURN m.waypointID AS waypointID
    """.format(user_id)

    results = query_neo4j(query)
    waypoint_ids = []

    for res in results:
        waypoint_ids.append(res['waypointID'])

    return waypoint_ids


def get_end_waypoint(start_waypoint_id):
    query = """
    MATCH (n:Waypoint {{waypointID: '{}'}})-[p:NEXT_WAYPOINT*]->(m:Waypoint)
    RETURN length(p) AS length, m.waypointID AS waypointID
    """.format(start_waypoint_id)

    results = query_neo4j(query)
    max_length = 0
    end_waypoint_id = None
    for res in results:
        if res['length'] > max_length:
            max_length = max(max_length, res['length'])
            end_waypoint_id = res['waypointID']

    return end_waypoint_id


def get_gps_ids(waypoint_ids):
    query = """
    MATCH (n)-[:WAYPOINT]->(m:Waypoint)
    WHERE m.waypointID = '{}'
    RETURN n.gpsPlusID AS gpsID, n.date AS date
    """

    gps_ids = list()
    date_to_ids = list()

    for waypoint_id in waypoint_ids:
        if waypoint_id is None:
            gps_ids.append(None)
            date_to_ids.append(None)
            continue
        result = query_neo4j(query.format(waypoint_id))[0]
        gps_ids.append(result['gpsID'])
        date_to_ids.append(result['date'])

    return gps_ids, date_to_ids


def get_spot_ids(gps_ids):
    query = """
    MATCH (n:GPS_Plus)-[:MAPPED_TO_SPOT]->(m)
    WHERE n.gpsPlusID = '{}'
    RETURN m
    """

    spot_ids = list()

    for gps_id in gps_ids:
        if gps_id is None:
            spot_ids.append(None)
            continue
        res = query_neo4j(query.format(gps_id))
        spot_ids.append(res[0])

    return spot_ids


def get_spot_position(spot_id):
    query = """
    MATCH (n:Spot)
    WHERE n.spotID = '{}'
    RETURN
        n.latitude AS latitude,
        n.longitude AS longitude
    """.format(spot_id)

    res = query_neo4j(query)

    if len(res) == 1:
        return res[0]
    return {'latitude': None, 'longitude': None}


