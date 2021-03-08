import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


SCALER_COMBINATIONS = ['none2none', 'log2none', 'log2standard', 'standard2standard', 'minmax2minmax']


class Scaler:
    def __init__(self):
        pass

    def fit(self, x):
        pass


class LogScaler(Scaler):
    @staticmethod
    def transform(x):
        return np.log(x+1)
    
    @staticmethod
    def inverse_transform(x):
        return np.exp(x) - 1


class NoneScaler(Scaler):
    @staticmethod
    def transform(x):
        return x

    @staticmethod
    def inverse_transform(x):
        return x
    

SCALER_TYPES = {
    'log': LogScaler,
    'none': NoneScaler,
    'minmax': MinMaxScaler,
    'standard': StandardScaler,
}


class ScalerManager:
    def __init__(self, args, x, y):
        # SET INPUT AND TARGET SCALERS
        x_scaler_type, y_scaler_type = args['scaler_type'].split('2')
        self.x_scaler = SCALER_TYPES[x_scaler_type]()
        self.y_scaler = SCALER_TYPES[y_scaler_type]()
        
        self.fit(x, y)
    
    def fit(self, x, y):
        self.x_scaler.fit(x)
        self.y_scaler.fit(y)
        
    def transform(self, x, y):
        return self.x_scaler.transform(x), self.y_scaler.transform(y)
    
    def inverse_transform(self, x, y):
        return self.x_scaler.inverse_transform(x), self.y_scaler.inverse_transform(y)
