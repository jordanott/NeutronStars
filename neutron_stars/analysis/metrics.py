import numpy as np
import pandas as pd


class MSE:
    def __init__(self):
        self.name = 'mse'
        self.label = r'Squared Error: $(Y - \hat{Y})^2$'

    def __call__(self, df, true_cols, pred_cols):
        errors = (df[true_cols].values - df[pred_cols].values) ** 2
        return np.mean(errors), pd.DataFrame(data=errors, columns=true_cols)


class MAPE:
    def __init__(self):
        self.name = 'mape'
        self.label = r'Absolute Percent Error: $100 * |(Y - \hat{Y}) / Y|$'

    def __call__(self, df, true_cols, pred_cols):
        errors = 100 * (df[true_cols].values - df[pred_cols].values) / df[true_cols].values
        return np.mean(np.abs(errors)), pd.DataFrame(data=errors, columns=true_cols)


AVAILABLE_METRICS = {
    'mse': MSE,
    'mape': MAPE,
}
