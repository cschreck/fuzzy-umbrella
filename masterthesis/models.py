import datetime
from collections import defaultdict
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin



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


