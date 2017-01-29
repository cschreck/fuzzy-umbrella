import datetime
from collections import defaultdict
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from collections import Counter
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

##############################
######### Use Case 1 #########
##############################

class FrequentistEstimator(BaseEstimator):
    def fit(self, X, y=None):
        self.data_ = X
        return self

    def partial_fit(self):
        # stack data to present data
        pass

    def predict_proba(self, x):
        length = len(self.data_)
        start_end = [(start, end) for start, end in zip(self.data_['start_cluster'], self.data_['end_cluster'])]
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

        res = {key: priors[key.split('_')[0]] * p_ba[key] / (1 / 7) for key in p_ba}

        return res


##############################
######### Use Case 2 #########
##############################

class FuzzyLocationMixin(object):
    def _get_nearest_known_spot(self, lat, lon):
        min_dist = self.max_dist_nearest_spot
        spot_id = None
        for index, row in self.data_.iterrows():
            dist = geo.distance(lat, lon, None, float(row['start_lat']), float(row['start_lon']), None)
            if dist < min_dist:
                spot_id = row['start_spot']
                min_dist = dist

        return spot_id



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


class TimeAndLocSensitiveMostFrequentRoute(BaseEstimator, ClassifierMixin, FuzzyLocationMixin):
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


class TLWSensitiveMostFrequentRoute(BaseEstimator, ClassifierMixin, FuzzyLocationMixin):
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


