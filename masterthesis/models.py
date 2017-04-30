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


class LookAheadMixin(object):
    def _look_ahead(self):
        one_second = timedelta(seconds=1)
        look_ahead_data = list()
        idx_insert = list()
        self.data_.reset_index(drop=True, inplace=True)

        if self.look_ahead:
            for i in range(len(self.data_)):
                row_i = self.data_.iloc[i]
                for j in range(i + 1, len(self.data_)):
                    row_j = self.data_.iloc[j]
                    if row_j['start_date'] - row_i['end_date'] < self.look_ahead:
                        look_ahead_data.append([row_i['start_lat'],
                                                row_i['start_lon'],
                                                row_i['start_date'] + one_second,
                                                row_j['end_lat'],
                                                row_j['end_lon'],
                                                row_j['end_date'],
                                                row_i['start_cluster'],
                                                row_j['end_cluster'],
                                                ])

                    else:
                        break

        if len(look_ahead_data) > 0:
            self.data_ = pd.concat([self.data_, pd.DataFrame(look_ahead_data, columns=self.data_.columns.values)], ignore_index=True)

        self.data_ = pd.DataFrame(data=self.data_, columns=self.data_.columns.values)
        self.data_ = self.data_.set_index(pd.DatetimeIndex(self.data_['start_date'])).sort_index()

        return self.data_




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


class BayesWeekdayEstimator(BaseEstimator, LookAheadMixin):
    def __init__(
            self,
            look_ahead=timedelta(hours=1)
    ):
        self.look_ahead = look_ahead

    def fit(self, df):
        self.data_ = df.copy()
        self.data_ = self._look_ahead()
        return self

    def partial_fit(self, X):
        # stack data to present data
        pass

    def predict_proba(self, x):
        length = len(self.data_)
        start_end = [(start, end) for start, end in zip(self.data_['start_cluster'], self.data_['end_cluster'])]
        priors = {k: v / length for k, v in Counter(start_end).items()}

        dayofweek = x.index.dayofweek
        pb = len(self.data_.index.dayofweek == dayofweek)/length

        counts_given_dayofweek = [(row['start_cluster'], row['end_cluster']) for index, row in self.data_.iterrows() if
                                  index.dayofweek == dayofweek]
        prob_given_dayofweek = {k: v / len(counts_given_dayofweek) for k, v in Counter(counts_given_dayofweek).items()}

        res = {key: priors[key] * prob_given_dayofweek[key] / pb for key in prob_given_dayofweek}

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
        pb = len(self.data_.index.dayofweek == dayofweek) / length

        p_ba = [row['start_time_cluster'] for index, row in self.data_.iterrows() if
                index.dayofweek == dayofweek]
        p_ba = {k: v / len(p_ba) for k, v in Counter(p_ba).items()}

        res = {key: priors[key.split('___')[0]] * p_ba[key] / pb for key in p_ba}

        return res

    def resolve_start_time_cluster(self, stc):
        slice = self.data_[self.data_['start_time_cluster'] == stc]
        return min(slice['start_date'].dt.time), max(slice['start_date'].dt.time)



##############################
######### Use Case 2 #########
##############################


class LocationSensitiveEstimator(FuzzyLocationMixin, LookAheadMixin):
    def __init__(
            self,
            max_dist=500,
            look_ahead=timedelta(hours=1)
    ):
        self.max_dist_nearest_cluster = max_dist
        self.look_ahead = look_ahead

    def fit(self, data):
        self.data_ = data.copy()
        self.data_ = self._look_ahead()
        return self

    def predict_proba(self, current_p):
        cluster = self._get_nearest_cluster(current_p.lat, current_p.lon)
        if cluster is None:
            return None
        data_selected = self.data_[self.data_['start_cluster'] == cluster]
        probs = data_selected['end_cluster'].value_counts() / len(data_selected)
        return probs


class LocationAndTimeSensitiveEstimator(FuzzyLocationMixin, LookAheadMixin):
    def __init__(
            self,
            max_dist=500,
            look_ahead=timedelta(hours=1),
            time_window=timedelta(hours=1)
    ):
        self.max_dist_nearest_cluster = max_dist
        self.look_ahead = look_ahead
        self.time_window = time_window
        self.dummy_date = datetime.date(1970, 1, 1)

    def fit(self, data):
        self.data_ = data.copy()
        self.data_ = self._look_ahead()
        return self

    def predict_proba(self, current_p):
        cluster = self._get_nearest_cluster(current_p.lat, current_p.lon)
        if cluster is None:
            return None
        data_selected = self.data_[self.data_['start_cluster'] == cluster]

        dummy_datetime = datetime.datetime.combine(
                                self.dummy_date,
                                datetime.time(current_p.datetime.hour, current_p.datetime.minute)
        )
        lower_bound = dummy_datetime - self.time_window
        upper_bound = dummy_datetime + self.time_window
        lower_bound = '{}:{}'.format(lower_bound.hour, lower_bound.minute)
        upper_bound = '{}:{}'.format(upper_bound.hour, upper_bound.minute)
        data_selected = data_selected.between_time(lower_bound, upper_bound)

        probs = data_selected['end_cluster'].value_counts() / len(data_selected)
        return probs


class LocationAndWeekdaySensitiveEstimator(FuzzyLocationMixin, LookAheadMixin):
    def __init__(
            self,
            max_dist=500,
            look_ahead=timedelta(hours=1),
    ):
        self.max_dist_nearest_cluster = max_dist
        self.look_ahead = look_ahead

    def fit(self, data):
        self.data_ = data.copy()
        self.data_ = self._look_ahead()
        return self

    def predict_proba(self, current_p):
        cluster = self._get_nearest_cluster(current_p.lat, current_p.lon)
        if cluster is None:
            return None
        data_selected = self.data_[self.data_['start_cluster'] == cluster]
        day_of_week = current_p.datetime.weekday()
        data_selected = data_selected[data_selected['start_date'].dt.dayofweek == day_of_week]

        probs = data_selected['end_cluster'].value_counts() / len(data_selected)
        return probs


class LocationAndPeriodSensitiveEstimator(FuzzyLocationMixin, LookAheadMixin):
    def __init__(
            self,
            max_dist=500,
            look_ahead=timedelta(hours=1),
            period=14
    ):
        self.max_dist_nearest_cluster = max_dist
        self.look_ahead = look_ahead
        self.period = period

    def fit(self, data):
        self.data_ = data.copy()
        self.data_ = self._look_ahead()
        return self

    def predict_proba(self, current_p):
        cluster = self._get_nearest_cluster(current_p.lat, current_p.lon)
        if cluster is None:
            return None
        data_selected = self.data_[self.data_['start_cluster'] == cluster]

        selection = list()
        date = datetime.datetime.combine(current_p.datetime.date(), datetime.time(0))
        for i, row in data_selected.iterrows():
            if (row['start_date'] - date).days % self.period == 0:
                selection.append(i)

        if len(selection) > 0:
            data_selected = data_selected.loc[selection]
        else:
            return None

        probs = data_selected['end_cluster'].value_counts() / len(data_selected)
        return probs


class AverageEnsemble():
    def average_proba(self, lists_of_probs):
        potential_ecs = set()
        for list_of_probs in lists_of_probs:
            if list_of_probs is not None:
                potential_ecs = potential_ecs.union(list_of_probs.index)

        result = dict()
        for potential_ec in potential_ecs:
            probs_for_ec = list()
            for list_of_probs in lists_of_probs:
                try:
                    prob = list_of_probs.loc[potential_ec]
                except:
                    prob = 0
                probs_for_ec.append(prob)

            result[potential_ec] = sum(probs_for_ec)/len(probs_for_ec)

        return pd.Series(result)




##############################
######### Use Case 3 #########
##############################



class HeadingEstimator(FuzzyLocationMixin, LookAheadMixin):
    def __init__(
            self,
            max_dist=500,
            angle_threshold=50,
            look_ahead=timedelta(hours=1),
    ):
        self.max_dist_nearest_cluster = max_dist
        self.angle_threshold = angle_threshold
        self.look_ahead = look_ahead

    def fit(self, data):
        self.data_ = data.copy()
        self.data_ = self._look_ahead()
        return self

    def predict_proba(self, traj):
        start_p = traj[0]
        end_p = traj[-1]
        heading_user = calc_heading(start_p, end_p)
        dist_user = haversine_distance(start_p.lat, start_p.lon, end_p.lat, end_p.lon)

        cluster = self._get_nearest_cluster(start_p.lat, start_p.lon)
        if cluster is None:
            return None
        data_selected = self.data_[self.data_['start_cluster'] == cluster]

        selection = list()
        for i, row in data_selected.iterrows():
            scp = Point(row['start_lat'], row['start_lon'], None)
            ecp = Point(row['end_lat'], row['end_lon'], None)

            heading_clusters = calc_heading(scp, ecp)

            diff_heading = abs(heading_user - heading_clusters)
            diff_heading = self._correct_heading_diff(diff_heading)

            if diff_heading > self.angle_threshold:
                continue

            dist_clusters = haversine_distance(scp.lat, scp.lon, ecp.lat, ecp.lon)

            if dist_user > dist_clusters:
                continue

            selection.append(i)

        if len(selection) > 0:
            data_selected = data_selected.loc[selection]
        else:
            return None

        return data_selected['end_cluster'].value_counts()/len(data_selected)

    @staticmethod
    def _correct_heading_diff(diff):
        if diff >= 180:
            diff = 360 - diff
        return diff


class LocationAndHeadingSensitiveEstimator(FuzzyLocationMixin, LookAheadMixin):
    def __init__(self, angle=25, angle_threshold=30):
        self.angle = angle
        self.angle_threshold = angle_threshold

    def fit(self, data):
        self.data_ = data.copy()
        self.data_ = self._look_ahead()
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


