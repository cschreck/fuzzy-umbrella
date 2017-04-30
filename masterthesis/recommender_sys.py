import datetime
import requests
import json

from masterthesis import config
from tpm.util.dist import calc_heading
from tpm.util.dist import haversine_distance
from tpm.data_model import Point



def time_selection(data, dt):
    dt = dt + datetime.timedelta(hours=2)
    data[(data['ts_start'].dt.date == dt.date()) & (data['ts_start'] < dt) & (data['ts_end'] > dt)]
    return data


def same_direction(rs, re, ps, pe, threshold):
    bearing1 = calc_heading(rs, re)
    bearing2 = calc_heading(ps, pe)

    return abs(bearing1 - bearing2) < threshold


def spatial_selection(rs, re, ds, de):
    if not same_direction(rs, re, ds, de, 30):
        return False
    lambda_r = haversine_distance(rs, re) / 2 * 1.3
    middle_p = Point((rs.lat + re.lat) / 2, (rs.lon + re.lon) / 2, None)
    if lambda_r > haversine_distance(middle_p, ds) and lambda_r > haversine_distance(middle_p, de):
        return False
    return True

def length_divergence(rs, re, ds, de, travelmode):
    start_route = "geo!{},{}".format(rs.lat, rs.lon)
    end_route = "geo!{},{}".format(re.lat, re.lon)
    start_delivery = "geo!{},{}".format(ds.lat, ds.lon)
    end_delivery = "geo!{},{}".format(de.lat, de.lon)

    params = {'app_id': config.APPID, 'app_code': config.APPCODE, 'mode': 'shortest;{}'.format(travelmode)}
    normal_route_params = params.copy()
    normal_route_params['waypoint0'] = start_route
    normal_route_params['waypoint1'] = end_route

    part1_params = params.copy()
    part1_params['waypoint0'] = start_route
    part1_params['waypoint1'] = start_delivery

    part2_params = params.copy()
    part2_params['waypoint0'] = start_delivery
    part2_params['waypoint1'] = end_delivery

    part3_params = params.copy()
    part3_params['waypoint0'] = end_delivery
    part3_params['waypoint1'] = end_route

    res_normal = requests.get('https://route.cit.api.here.com/routing/7.2/calculateroute.json', normal_route_params)
    res_part1 = requests.get('https://route.cit.api.here.com/routing/7.2/calculateroute.json', part1_params)
    res_part2 = requests.get('https://route.cit.api.here.com/routing/7.2/calculateroute.json', part2_params)
    res_part3 = requests.get('https://route.cit.api.here.com/routing/7.2/calculateroute.json', part3_params)

    len_x = json.loads(res_normal.content.decode('utf-8'))['response']['route'][0]['summary']['distance']
    len_1 = json.loads(res_part1.content.decode('utf-8'))['response']['route'][0]['summary']['distance']
    len_2 = json.loads(res_part2.content.decode('utf-8'))['response']['route'][0]['summary']['distance']
    len_3 = json.loads(res_part3.content.decode('utf-8'))['response']['route'][0]['summary']['distance']
    return len_1 + len_2 + len_3 - len_x


def time_divergence(rs, re, ds, de, travelmode):
    start_route = "geo!{},{}".format(rs.lat, rs.lon)
    end_route = "geo!{},{}".format(re.lat, re.lon)
    start_delivery = "geo!{},{}".format(ds.lat, ds.lon)
    end_delivery = "geo!{},{}".format(de.lat, de.lon)

    params = {'app_id': config.APPID, 'app_code': config.APPCODE, 'mode': 'fastest;{}'.format(travelmode)}
    normal_route_params = params.copy()
    normal_route_params['waypoint0'] = start_route
    normal_route_params['waypoint1'] = end_route

    part1_params = params.copy()
    part1_params['waypoint0'] = start_route
    part1_params['waypoint1'] = start_delivery

    part2_params = params.copy()
    part2_params['waypoint0'] = start_delivery
    part2_params['waypoint1'] = end_delivery

    part3_params = params.copy()
    part3_params['waypoint0'] = end_delivery
    part3_params['waypoint1'] = end_route

    res_normal = requests.get('https://route.cit.api.here.com/routing/7.2/calculateroute.json', normal_route_params)
    res_part1 = requests.get('https://route.cit.api.here.com/routing/7.2/calculateroute.json', part1_params)
    res_part2 = requests.get('https://route.cit.api.here.com/routing/7.2/calculateroute.json', part2_params)
    res_part3 = requests.get('https://route.cit.api.here.com/routing/7.2/calculateroute.json', part3_params)

    time_x = json.loads(res_normal.content.decode('utf-8'))['response']['route'][0]['summary']['travelTime']
    time_1 = json.loads(res_part1.content.decode('utf-8'))['response']['route'][0]['summary']['travelTime']
    time_2 = json.loads(res_part2.content.decode('utf-8'))['response']['route'][0]['summary']['travelTime']
    time_3 = json.loads(res_part3.content.decode('utf-8'))['response']['route'][0]['summary']['travelTime']

    return time_1 + time_2 + time_3 - time_x

