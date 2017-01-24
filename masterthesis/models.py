import datetime
from collections import defaultdict
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from gpxpy import geo


class DummyMostFrequentRoute(BaseEstimator, ClassifierMixin):
    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        self.outer_dict_ = defaultdict(lambda: defaultdict(int))
        for start, target in zip(X, y):
            self.outer_dict_[start][target] += 1
        # move maxtarget here
        return self

    def predict(self, X):
        return [self._max_target(x) for x in X['start_spot']]

    def _max_target(self, x):
        target_counts = self.outer_dict_[x]
        return max(target_counts.keys(), key=(lambda key: target_counts[key]))


class TimeSensitiveMostFrequentRoute(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            time_window,
    ):
        self.time_window = time_window
        self.dummydate = datetime.date(1970, 1, 1)
        super().__init__()

    def fit(self, X, y):
        self.data_ = X.copy()

    def predict(self, X):
        res_list = []

        for index, x in X.iterrows():
            res = self.data_[self.data_['start_spot'] == x['start_spot']]
            dummydatetime = datetime.datetime.combine(self.dummydate, datetime.time(index.hour, index.minute))
            lowerbound = dummydatetime - self.time_window
            upperbound = dummydatetime + self.time_window
            lowerbound = '{}:{}'.format(lowerbound.hour, lowerbound.minute)
            upperbound = '{}:{}'.format(upperbound.hour, upperbound.minute)
            res = res.between_time(lowerbound, upperbound)
            try:
                res = res['target_spot'].value_counts().index[0]
            except:
                res = None
            res_list.append(res)

        return res_list


class TimeAndLocSensitiveMostFrequentRoute(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            time_window,
            max_dist_nearest_spot,
            location,
    ):
        self.time_window = time_window
        self.dummydate = datetime.date(1970, 1, 1)
        self.max_dist_nearest_spot = max_dist_nearest_spot
        self.location = location

    def fit(self, X, y=None):
        self.data_ = X.copy()

    def predict(self, X):
        res_list = []

        for index, x in X.iterrows():
            lat = x['lat']
            lon = x['lon']
            nearest_spot = self._get_nearest_known_spot(lat, lon)
            if nearest_spot == None:
                res_list.append(None)
                continue

            res = self.data_[self.data_['start_spot'] == nearest_spot]
            dummydatetime = datetime.datetime.combine(self.dummydate, datetime.time(index.hour, index.minute))
            lowerbound = dummydatetime - self.time_window
            upperbound = dummydatetime + self.time_window
            lowerbound = '{}:{}'.format(lowerbound.hour, lowerbound.minute)
            upperbound = '{}:{}'.format(upperbound.hour, upperbound.minute)
            res = res.between_time(lowerbound, upperbound)

            try:
                res = res['end_spot'].value_counts().index[0]
            except:
                res = None
            res_list.append(res)

        return res_list

    def predict_proba(self, x):
        lat = x['lat'].iloc[0]
        lon = x['lon'].iloc[0]

        nearest_spot = self._get_nearest_known_spot(lat, lon)
        if nearest_spot == None:
            return None

        index = x.index

        res = self.data_[self.data_['start_spot'] == nearest_spot]
        dummydatetime = datetime.datetime.combine(self.dummydate, datetime.time(index.hour, index.minute))

        lowerbound = dummydatetime - self.time_window
        upperbound = dummydatetime + self.time_window
        lowerbound = '{}:{}'.format(lowerbound.hour, lowerbound.minute)
        upperbound = '{}:{}'.format(upperbound.hour, upperbound.minute)
        res = res.between_time(lowerbound, upperbound)

        length = len(res)

        try:
            res = res['end_{}'.format(self.location)].value_counts() / length
        except:
            res = None

        return res

    def _get_nearest_known_spot(self, lat, lon):
        min_dist = self.max_dist_nearest_spot
        spot_id = None
        for index, row in self.data_.iterrows():
            dist = geo.distance(lat, lon, None, float(row['start_lat']), float(row['start_lon']), None)
            if dist < min_dist:
                spot_id = row['start_spot']
                min_dist = dist

        return spot_id
