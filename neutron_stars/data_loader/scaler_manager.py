import itertools
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


SCALER_COMBINATIONS = ['none2none', 'log2none', 'log2standard', 'standard2standard', 'minmax2minmax', 'log2exp']


class Scaler:
    def __init__(self):
        pass

    def fit(self, x):
        pass


class ZeroMean(Scaler):
    def __init__(self, mean=None):
        self.mean = mean

    def fit(self, x):
        self.mean = np.mean(x, axis=0)

    def transform(self, x):
        return x - self.mean

    def inverse_transform(self, x):
        return x + self.mean


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
    

class ExpScaler(Scaler):
    @staticmethod
    def transform(x):
        tmp_x = np.copy(x)
        tmp_x[:, 0] = np.exp(x[:, 0])
        tmp_x[:, 1] = np.exp(5 + x[:, 1])

        return tmp_x

    @staticmethod
    def inverse_transform(x):
        tmp_x = np.copy(x)
        tmp_x[:, 0] = np.log(x[:, 0])
        tmp_x[:, 1] = np.log(x[:, 1]) - 5

        return tmp_x


SCALER_TYPES = {
    'zero_mean': ZeroMean,
    'log': LogScaler,
    # 'exp': ExpScaler,
    'none': NoneScaler,
    'minmax': MinMaxScaler,
    'standard': StandardScaler,
}


def scaler_combinations_for_paradigm(paradigm):
    all_scalers = []
    # if 'eos' in paradigm:
    #     del SCALER_TYPES['log']

    scaler_types = list(SCALER_TYPES.keys())
    num_inputs = len(paradigm.split('2')[0].split('+'))
    num_outputs = len(paradigm.split('2')[1].split('+'))

    for i in range(num_inputs+num_outputs):
        if i >= num_inputs and '2eos' in paradigm:
            scaler_types = ['zero_mean']
        all_scalers.append(scaler_types)

    all_combos = list(itertools.product(
        *all_scalers
    ))

    scaler_combinations = []
    for combo in all_combos:
        scaler_combinations.append(
            '+'.join(combo[:num_inputs]) + '2' + '+'.join(combo[num_inputs:])
        )

    return scaler_combinations


class ScalerManager:
    def __init__(self, args, x, y):
        self.args = args
        # SET INPUT AND TARGET SCALERS
        input_scalers, output_scalers = args['scaler_type'].split('2')

        self.input_scalers = {}
        for input_type, scaler_type in zip(args['input_names'], input_scalers.split('+')):
            self.input_scalers[input_type] = SCALER_TYPES[scaler_type]()

        self.output_scalers = {}
        for output_type, scaler_type in zip(args['output_names'], output_scalers.split('+')):
            self.output_scalers[output_type] = SCALER_TYPES[scaler_type]()
        
        self.fit(x, y)
    
    def fit(self, x, y):
        for input_type in x:
            self.input_scalers[input_type].fit(x[input_type])

        for output_type in y:
            self.output_scalers[output_type].fit(y[output_type])

    def transform(self, x, y):
        if self.args['model_type'] == 'transformer':
            _x = {input_type: np.zeros_like(x[input_type])
                  for input_type in x}

            for input_type in x:
                for bidx in range(self.args['batch_size']):
                    _x[input_type][bidx] = self.input_scalers[input_type].transform(x[input_type][bidx])

        else:
            _x = {input_type: self.input_scalers[input_type].transform(x[input_type])
                  for input_type in x}

        _y = {output_type: self.output_scalers[output_type].transform(y[output_type])
              for output_type in y}

        return _x, _y
    
    def inverse_transform(self, x, y):
        if self.args['model_type'] == 'transformer':
            _x = {input_type: np.zeros_like(x[input_type])
                  for input_type in x}

            for input_type in x:
                for bidx in range(self.args['batch_size']):
                    _x[input_type][bidx] = self.input_scalers[input_type].inverse_transform(x[input_type][bidx])

        else:
            _x = {input_type: self.input_scalers[input_type].inverse_transform(x[input_type])
                  for input_type in x}

        _y = {output_type: self.output_scalers[output_type].inverse_transform(y[output_type])
              for output_type in y}

        return _x, _y
