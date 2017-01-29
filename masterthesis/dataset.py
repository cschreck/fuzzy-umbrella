from masterthesis.database import get_start_waypoint_ids
from masterthesis.database import get_gps_ids
from masterthesis.database import get_spot_ids
from masterthesis.database import get_spot_position
from masterthesis.database import get_end_waypoint

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
