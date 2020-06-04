from sklearn.preprocessing import MinMaxScaler
import pickle

class MinMax_scaler:
    def __init__(self):
        self.sc = MinMaxScaler()

    def fit_transform(self, data):
        fit_transformed_data = self.sc.fit_transform(data)
        return fit_transformed_data
