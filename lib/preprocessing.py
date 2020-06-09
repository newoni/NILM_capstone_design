from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
import pickle

class MinMax_scaler:
    def __init__(self):
        self.sc = MinMaxScaler()

    def fit_transform(self, data):
        fit_transformed_data = self.sc.fit_transform(data)
        return fit_transformed_data

class MaxAbs_scaler:
    def __init__(self):
        self.sc = MaxAbsScaler()

    def fit_transform(self, data):
        fit_transformed_data = self.sc.fit_transform(data)
        return fit_transformed_data