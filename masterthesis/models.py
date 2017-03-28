import datetime
from collections import defaultdict
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from collections import Counter
from gpxpy import geo
from datetime import timedelta
import pandas as pd
import numpy as np
from masterthesis.util import haversine_distance
from tpm.util.dist import calc_heading
from tpm.data_model import Point


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


##############################
######### Use Case 1 #########
##############################

class FrequentistEstimator(BaseEstimator):
    def __init__(
            self,
            look_ahead=timedelta(hours=1)
    ):
        self.look_ahead = look_ahead

    def fit(self, X, y=None):
        self.data_ = [(date, start, end) for date, start, end in
                      zip(X['start_date'], X['start_cluster'], X['end_cluster'])]
        look_ahead_data = list()
        idx_insert = list()
        if self.look_ahead:
            for i, tup in enumerate(self.data_):
                for j in range(i + 1, len(self.data_)):
                    if self.data_[j][0] - tup[0] < self.look_ahead:
                        look_ahead_data.append([tup[0], tup[1], self.data_[j][2]])
                        idx_insert.append(i + 1)

        idx_insert = sorted(idx_insert, reverse=True)
        for i, idx in enumerate(idx_insert):
            self.data_.insert(idx, look_ahead_data[i])

        return self

    def partial_fit(self):
        # stack data to present data
        pass

    def predict_proba(self, x):
        length = len(self.data_)
        start_end = [(start, end) for _, start, end in self.data_]
        return {k: v / length for k, v in Counter(start_end).items()}


class BayesWeekdayEstimator(BaseEstimator):
    def fit(self, X, y=None):
        self.data_ = X
        return self

    def partial_fit(self, X):
        # stack data to present data
        pass

    def predict_proba(self, x):
        length = len(self.data_)
        start_end = [(start, end) for start, end in zip(self.data_['start_cluster'], self.data_['end_cluster'])]
        priors = {k: v / length for k, v in Counter(start_end).items()}

        dayofweek = x.index.dayofweek

        counts_given_dayofweek = [(row['start_cluster'], row['end_cluster']) for index, row in self.data_.iterrows() if
                                  index.dayofweek == dayofweek]
        prob_given_dayofweek = {k: v / len(counts_given_dayofweek) for k, v in Counter(counts_given_dayofweek).items()}

        res = {key: priors[key] * prob_given_dayofweek[key] / (1 / 7) for key in prob_given_dayofweek}

        return res


class BayesDepartureTimeEstimator(BaseEstimator):
    def fit(self, X, y=None):
        self.data_ = X
        return self

    def partial_fit(self, X):
        # stack data to present data
        pass

    def predict_proba(self, x):
        length = len(self.data_)
        start_end = [(start, end) for start, end in zip(self.data_['start_cluster'], self.data_['end_cluster'])]
        priors = {str(k): v / length for k, v in Counter(start_end).items()}

        dayofweek = x.index.dayofweek

        p_ba = [row['start_time_cluster'] for index, row in self.data_.iterrows() if
                index.dayofweek == dayofweek]
        p_ba = {k: v / len(p_ba) for k, v in Counter(p_ba).items()}

        res = {key: priors[key.split('___')[0]] * p_ba[key] / (1 / 7) for key in p_ba}

        return res

    def resolve_start_time_cluster(self, stc):
        return self.data_[self.data_['start_time_cluster'] == stc]


##############################
######### Use Case 2 #########
##############################

class FuzzyLocationMixin(object):
    def _get_nearest_cluster(self, lat, lon):
        min_dist = self.max_dist_nearest_cluster
        cluster = None
        for index, row in self.data_.iterrows():

            dist = haversine_distance(lat, lon, row['start_lat'], row['start_lon'])
            if dist < min_dist:
                cluster = row['start_cluster']
                min_dist = dist

        return cluster


class TimeSensitiveMostFrequentRoute(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            time_window=timedelta(hours=1),
            look_ahead=timedelta(hours=1)
    ):
        self.time_window = time_window
        self.dummydate = datetime.date(1970, 1, 1)
        self.look_ahead = look_ahead

    def fit(self, X, y=None):
        self.data_ = [(date, start, end) for date, start, end in
                      zip(X['start_date'], X['start_cluster'], X['end_cluster'])]

        look_ahead_data = list()
        idx_insert = list()
        if self.look_ahead:
            for i, tup in enumerate(self.data_):
                for j in range(i + 1, len(self.data_)):
                    if self.data_[j][0] - tup[0] < self.look_ahead:
                        look_ahead_data.append([tup[0], tup[1], self.data_[j][2]])
                        idx_insert.append(i + 1)

        idx_insert = sorted(idx_insert, reverse=True)
        for i, idx in enumerate(idx_insert):
            self.data_.insert(idx, look_ahead_data[i])

        self.data_ = pd.DataFrame(data=self.data_, columns=['start_date', 'start_cluster', 'end_cluster'])
        self.data_ = self.data_.set_index(pd.DatetimeIndex(self.data_['start_date'])).sort_index()

        return self

    def predict(self, X):

        res_list = list()
        for index, row in X.iterrows():
            res = self.data_[self.data_['start_cluster'] == row['start_cluster']]
            dummydatetime = datetime.datetime.combine(self.dummydate, datetime.time(index.hour, index.minute))
            lowerbound = dummydatetime - self.time_window
            upperbound = dummydatetime + self.time_window
            lowerbound = '{}:{}'.format(lowerbound.hour, lowerbound.minute)
            upperbound = '{}:{}'.format(upperbound.hour, upperbound.minute)
            res = res.between_time(lowerbound, upperbound)
            try:
                res = res['end_cluster'].value_counts().index[0]
            except:
                res = None
            res_list.append(res)

        return res_list

    def predict_proba(self, X):
        length = len(self.data_)
        res_list = list()
        for index, row in X.iterrows():
            res = self.data_[self.data_['start_cluster'] == row['start_cluster']]
            dummydatetime = datetime.datetime.combine(self.dummydate, datetime.time(index.hour, index.minute))
            lowerbound = dummydatetime - self.time_window
            upperbound = dummydatetime + self.time_window
            lowerbound = '{}:{}'.format(lowerbound.hour, lowerbound.minute)
            upperbound = '{}:{}'.format(upperbound.hour, upperbound.minute)
            res = res.between_time(lowerbound, upperbound)
            try:
                res = res['end_cluster'].value_counts() / length
            except:
                res = None
            res_list.append(res)

        return res_list


class TimeAndLocSensitiveMostFrequentRoute(BaseEstimator, ClassifierMixin, FuzzyLocationMixin):
    def __init__(
            self,
            time_window=timedelta(hours=1),
            look_ahead=timedelta(hours=1),
            max_dist_nearest_cluster=100,

    ):
        self.time_window = time_window
        self.dummydate = datetime.date(1970, 1, 1)
        self.look_ahead = look_ahead
        self.max_dist_nearest_cluster = max_dist_nearest_cluster

    def fit(self, X, y=None):
        self.data_ = [(date, start, lat, lon, end) for date, start, lat, lon, end in
                      zip(X['start_date'], X['start_cluster'], X['start_lat'], X['start_lon'], X['end_cluster'])]

        look_ahead_data = list()
        idx_insert = list()
        if self.look_ahead:
            for i, tup in enumerate(self.data_):
                for j in range(i + 1, len(self.data_)):
                    if self.data_[j][0] - tup[0] < self.look_ahead:
                        look_ahead_data.append([tup[0], tup[1], tup[2], tup[3], self.data_[j][4]])
                        idx_insert.append(i + 1)

        idx_insert = sorted(idx_insert, reverse=True)
        for i, idx in enumerate(idx_insert):
            self.data_.insert(idx, look_ahead_data[i])

        self.data_ = pd.DataFrame(data=self.data_,
                                  columns=['start_date', 'start_cluster', 'start_lat', 'start_lon', 'end_cluster'])
        self.data_ = self.data_.set_index(pd.DatetimeIndex(self.data_['start_date'])).sort_index()

        return self

    def predict(self, X):
        res_list = []

        for index, x in X.iterrows():
            lat = x['start_lat']
            lon = x['start_lon']
            nearest_cluster = self._get_nearest_cluster(lat, lon)
            if nearest_cluster == None:
                res_list.append(None)
                continue

            res = self.data_[self.data_['start_cluster'] == nearest_cluster]
            dummydatetime = datetime.datetime.combine(self.dummydate, datetime.time(index.hour, index.minute))
            lowerbound = dummydatetime - self.time_window
            upperbound = dummydatetime + self.time_window
            lowerbound = '{}:{}'.format(lowerbound.hour, lowerbound.minute)
            upperbound = '{}:{}'.format(upperbound.hour, upperbound.minute)
            res = res.between_time(lowerbound, upperbound)

            try:
                res = res['end_cluster'].value_counts().index[0]
            except:
                res = None

            res_list.append(res)

        return res_list

    def predict_proba(self, X):
        res_list = []
        length = len(self.data_)

        for index, x in X.iterrows():
            lat = x['start_lat']
            lon = x['start_lon']
            nearest_cluster = self._get_nearest_cluster(lat, lon)
            if nearest_cluster == None:
                res_list.append(None)
                continue

            res = self.data_[self.data_['start_cluster'] == nearest_cluster]
            dummydatetime = datetime.datetime.combine(self.dummydate, datetime.time(index.hour, index.minute))
            lowerbound = dummydatetime - self.time_window
            upperbound = dummydatetime + self.time_window
            lowerbound = '{}:{}'.format(lowerbound.hour, lowerbound.minute)
            upperbound = '{}:{}'.format(upperbound.hour, upperbound.minute)
            res = res.between_time(lowerbound, upperbound)

            try:
                res = res['end_cluster'].value_counts() / length
            except:
                res = None

            res_list.append(res)

        return res_list


class TLWSensitiveMostFrequentRoute(BaseEstimator, ClassifierMixin, FuzzyLocationMixin):
    def __init__(
            self,
            time_window=timedelta(hours=1),
            look_ahead=timedelta(hours=1),
            max_dist_nearest_cluster=100,

    ):
        self.time_window = time_window
        self.dummydate = datetime.date(1970, 1, 1)
        self.look_ahead = look_ahead
        self.max_dist_nearest_cluster = max_dist_nearest_cluster

    def fit(self, X, y=None):
        self.data_ = [(date, start, lat, lon, end) for date, start, lat, lon, end in
                      zip(X['start_date'], X['start_cluster'], X['start_lat'], X['start_lon'], X['end_cluster'])]

        look_ahead_data = list()
        idx_insert = list()
        if self.look_ahead:
            for i, tup in enumerate(self.data_):
                for j in range(i + 1, len(self.data_)):
                    if self.data_[j][0] - tup[0] < self.look_ahead:
                        look_ahead_data.append([tup[0], tup[1], tup[2], tup[3], self.data_[j][4]])
                        idx_insert.append(i + 1)

        idx_insert = sorted(idx_insert, reverse=True)
        for i, idx in enumerate(idx_insert):
            self.data_.insert(idx, look_ahead_data[i])

        self.data_ = pd.DataFrame(data=self.data_,
                                  columns=['start_date', 'start_cluster', 'start_lat', 'start_lon', 'end_cluster'])
        self.data_ = self.data_.set_index(pd.DatetimeIndex(self.data_['start_date'])).sort_index()

        return self

    def predict(self, X):
        res_list = []

        for index, x in X.iterrows():
            lat = x['start_lat']
            lon = x['start_lon']
            nearest_cluster = self._get_nearest_cluster(lat, lon)
            if nearest_cluster == None:
                res_list.append(None)
                continue

            res = self.data_[self.data_['start_cluster'] == nearest_cluster]
            res = res[res.index.dayofweek == index.dayofweek]

            dummydatetime = datetime.datetime.combine(self.dummydate, datetime.time(index.hour, index.minute))
            lowerbound = dummydatetime - self.time_window
            upperbound = dummydatetime + self.time_window
            lowerbound = '{}:{}'.format(lowerbound.hour, lowerbound.minute)
            upperbound = '{}:{}'.format(upperbound.hour, upperbound.minute)
            res = res.between_time(lowerbound, upperbound)

            try:
                res = res['end_cluster'].value_counts().index[0]
            except:
                res = None
            res_list.append(res)

        return res_list

    def predict_proba(self, X):
        res_list = []
        length = len(self.data_)

        for index, x in X.iterrows():
            lat = x['start_lat']
            lon = x['start_lon']
            nearest_cluster = self._get_nearest_cluster(lat, lon)
            if nearest_cluster == None:
                res_list.append(None)
                continue

            res = self.data_[self.data_['start_cluster'] == nearest_cluster]
            res = res[res.index.dayofweek == index.dayofweek]

            dummydatetime = datetime.datetime.combine(self.dummydate, datetime.time(index.hour, index.minute))
            lowerbound = dummydatetime - self.time_window
            upperbound = dummydatetime + self.time_window
            lowerbound = '{}:{}'.format(lowerbound.hour, lowerbound.minute)
            upperbound = '{}:{}'.format(upperbound.hour, upperbound.minute)
            res = res.between_time(lowerbound, upperbound)

            try:
                res = res['end_cluster'].value_counts() / length
            except:
                res = None
            res_list.append(res)

        return res_list


##############################
######### Use Case 3 #########
##############################


class HeadingEstimator(FuzzyLocationMixin):
    def __init__(self, angle_threshold=60):
        self.angle_threshold = angle_threshold

    def fit(self, X, y=None):
        self.data_ = [(lat, lon, end) for lat, lon, end in
                      zip(X['end_lat'], X['end_lon'], X['end_cluster'])]

        self.data_ = pd.DataFrame(data=self.data_,
                                  columns=['end_lat', 'end_lon', 'end_cluster'])
        return self

    def predict(self, traj):
        start_p = traj[0]
        end_p = traj[-1]
        heading1 = calc_heading(start_p, end_p)
        heading_diffs = list()

        for i, row in self.data_.iterrows():
            p = Point(row['end_lat'], row['end_lon'], None)
            heading2 = calc_heading(start_p, p)
            heading3 = calc_heading(end_p, p)

            heading_diff1 = max(heading1, heading2) - min(heading1, heading2)
            heading_diff1 = self._correct_heading_diff(heading_diff1)

            heading_diff2 = max(heading1, heading3) - min(heading1, heading3)
            heading_diff2 = self._correct_heading_diff(heading_diff2)

            # print(heading_diff2)
            if heading_diff2 > self.angle_threshold:
                heading_diffs.append(180)
            else:
                heading_diffs.append(heading_diff1)

        return self.data_['end_cluster'].iloc[np.argmin(heading_diffs)]

    def _correct_heading_diff(self, diff):
        if diff >= 180:
            diff = 360 - diff
        return diff


class HeadingMostFrequentTargetEstimator(FuzzyLocationMixin):
    def __init__(self, angle=25, angle_threshold=30):
        self.angle = angle
        self.angle_threshold = angle_threshold

    def fit(self, X, y=None):
        self.data_ = [(lat, lon, end) for lat, lon, end in
                      zip(X['end_lat'], X['end_lon'], X['end_cluster'])]

        self.data_ = pd.DataFrame(data=self.data_,
                                  columns=['end_lat', 'end_lon', 'end_cluster'])
        return self

    def predict(self, traj):
        try:
            return self.predict_proba(traj).index[0]
        except:
            return None

    def predict_proba(self, traj):
        length = len(self.data_)
        start_p = traj[0]
        end_p = traj[-1]
        heading1 = calc_heading(start_p, end_p)

        potential_clusters = list()
        heading_diffs = list()

        for i, row in self.data_.iterrows():
            p = Point(row['end_lat'], row['end_lon'], None)
            heading2 = calc_heading(start_p, p)
            heading3 = calc_heading(end_p, p)

            heading_diff1 = max(heading1, heading2) - min(heading1, heading2)
            heading_diff1 = self._correct_heading_diff(heading_diff1)

            heading_diff2 = max(heading1, heading3) - min(heading1, heading3)
            heading_diff2 = self._correct_heading_diff(heading_diff2)

            # print(heading_diff2)
            if heading_diff2 < self.angle_threshold and heading_diff1 < self.angle:
                potential_clusters.append(row['end_cluster'])
                heading_diffs.append(heading_diff1)

        args = np.argsort(heading_diffs)
        sorted_clusters = list()
        for arg in args:
            if not potential_clusters[arg] in sorted_clusters:
                sorted_clusters.append(potential_clusters[arg])

        return self.data_['end_cluster'].loc[self.data_['end_cluster'].isin(potential_clusters)].value_counts() / length

    def _correct_heading_diff(self, diff):
        if diff >= 180:
            diff = 360 - diff
        return diff


class KnownStartEstimator(FuzzyLocationMixin):
    def __init__(
            self,
            time_window=timedelta(hours=1),
            look_ahead=timedelta(hours=1),
            max_dist_nearest_cluster=100,

    ):
        self.time_window = time_window
        self.dummydate = datetime.date(1970, 1, 1)
        self.look_ahead = look_ahead
        self.max_dist_nearest_cluster = max_dist_nearest_cluster

    def fit(self, X, y=None):
        self.data_ = [(date, start, lat, lon, end) for date, start, lat, lon, end in
                      zip(X['start_date'], X['start_cluster'], X['start_lat'], X['start_lon'], X['end_cluster'])]

        look_ahead_data = list()
        idx_insert = list()
        if self.look_ahead:
            for i, tup in enumerate(self.data_):
                for j in range(i + 1, len(self.data_)):
                    if self.data_[j][0] - tup[0] < self.look_ahead:
                        look_ahead_data.append([tup[0], tup[1], tup[2], tup[3], self.data_[j][4]])
                        idx_insert.append(i + 1)

        idx_insert = sorted(idx_insert, reverse=True)
        for i, idx in enumerate(idx_insert):
            self.data_.insert(idx, look_ahead_data[i])

        self.data_ = pd.DataFrame(data=self.data_,
                                  columns=['start_date', 'start_cluster', 'start_lat', 'start_lon', 'end_cluster'])
        self.data_ = self.data_.set_index(pd.DatetimeIndex(self.data_['start_date'])).sort_index()

        return self

    def predict(self, traj):
        try:
            return self.predict_proba(traj).index[0]
        except:
            return None

        return res

    def predict_proba(self, traj):
        length = len(self.data_)
        first_point = traj[0]

        lat = first_point.lat
        lon = first_point.lon
        nearest_cluster = self._get_nearest_cluster(lat, lon)
        if nearest_cluster == None:
            return None

        res = self.data_[self.data_['start_cluster'] == nearest_cluster]

        try:
            res = res['end_cluster'].value_counts() / length
        except:
            res = None

        return res


class KnownStartDepartureTimeEstimator(FuzzyLocationMixin):
    def __init__(
            self,
            time_window=timedelta(hours=1),
            look_ahead=timedelta(hours=1),
            max_dist_nearest_cluster=100,

    ):
        self.time_window = time_window
        self.dummydate = datetime.date(1970, 1, 1)
        self.look_ahead = look_ahead
        self.max_dist_nearest_cluster = max_dist_nearest_cluster

    def fit(self, X, y=None):
        self.data_ = [(date, start, lat, lon, end) for date, start, lat, lon, end in
                      zip(X['start_date'], X['start_cluster'], X['start_lat'], X['start_lon'], X['end_cluster'])]

        look_ahead_data = list()
        idx_insert = list()
        if self.look_ahead:
            for i, tup in enumerate(self.data_):
                for j in range(i + 1, len(self.data_)):
                    if self.data_[j][0] - tup[0] < self.look_ahead:
                        look_ahead_data.append([tup[0], tup[1], tup[2], tup[3], self.data_[j][4]])
                        idx_insert.append(i + 1)

        idx_insert = sorted(idx_insert, reverse=True)
        for i, idx in enumerate(idx_insert):
            self.data_.insert(idx, look_ahead_data[i])

        self.data_ = pd.DataFrame(data=self.data_,
                                  columns=['start_date', 'start_cluster', 'start_lat', 'start_lon', 'end_cluster'])
        self.data_ = self.data_.set_index(pd.DatetimeIndex(self.data_['start_date'])).sort_index()

        return self

    def predict(self, traj):
        try:
            return self.predict_proba(traj).index[0]
        except:
            return None

    def predict_proba(self, traj):
        res_list = []
        length = len(self.data_)
        first_point = traj[0]

        lat = first_point.lat
        lon = first_point.lon
        nearest_cluster = self._get_nearest_cluster(lat, lon)
        if nearest_cluster == None:
            return None

        res = self.data_[self.data_['start_cluster'] == nearest_cluster]

        dummydatetime = datetime.datetime.combine(self.dummydate,
                                                  datetime.time(first_point.datetime.hour, first_point.datetime.minute))
        lowerbound = dummydatetime - self.time_window
        upperbound = dummydatetime + self.time_window
        lowerbound = '{}:{}'.format(lowerbound.hour, lowerbound.minute)
        upperbound = '{}:{}'.format(upperbound.hour, upperbound.minute)
        res = res.between_time(lowerbound, upperbound)

        try:
            return res['end_cluster'].value_counts() / length
        except:
            return None


