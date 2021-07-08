import numpy as np
import pandas as pd
import tensorflow as tf
import neutron_stars as ns


class BlankScaler:
    def inverse_transform(self, x, y): return x, y


class ManyStarsGenerator(tf.keras.utils.Sequence):
    def __init__(self, args, generator, scaler):
        X = []
        Y = []
        self.stars_per_eos = []

        for x, y, _ in generator:
            key = list(x.keys())[0]
            self.stars_per_eos.append(len(x[key]))

            X.append(x)
            Y.append(y)

        self.X = X
        self.Y = Y
        self.args = args
        self.scaler = scaler
        self.num_stars = args['num_stars']
        self.args['batch_size'] = args['batch_size']
        self.seed = 1
        np.random.seed(self.seed)

        self.output_name = args['outputs'][0]['name']
        if self.output_name != 'coefficients':
            self.args['num_coefficients'] = 1

    def on_epoch_end(self):
        self.seed += 1
        np.random.seed(self.seed)

    def __len__(self):
        count = 0
        for y in self.Y:
            count += len(y[self.output_name])
        return count // self.args['batch_size']

    def __getitem__(self, _):
        batch_x = {input_type: np.zeros((self.args['batch_size'],
                                         self.num_stars,
                                         self.X[0][input_type].shape[-1]))
                   for input_type in self.X[0]}

        batch_y = np.zeros((self.args['batch_size'], self.args['num_coefficients']))
        eos_idxs = np.arange(len(self.stars_per_eos))

        for bidx in range(self.args['batch_size']):
            eos_idx = np.random.choice(eos_idxs)
            x = self.X[eos_idx]
            y = self.Y[eos_idx][self.output_name]

            star_idx = np.random.randint(0, self.stars_per_eos[eos_idx],
                                         size=self.num_stars)

            for input_type in x:
                batch_x[input_type][bidx] = np.expand_dims(x[input_type][star_idx], axis=0)

            batch_y[bidx] = np.expand_dims(y[0], axis=0)

        return self.scaler.transform(batch_x, {self.output_name: batch_y})

    def load_all(self, transform=True):
        np.random.seed(123)
        batch_size = self.args['batch_size']
        self.args['batch_size'] = len(self.X)

        x, y = self.__getitem__(None)
        if not transform:
            return self.scaler.inverse_transform(x, y)

        self.args['batch_size'] = batch_size
        return x, y


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, Y, args, scaler=None):
        self.X = X
        self.Y = Y
        self.args = args
        self.scaler = None
        self.batch_size = args['batch_size']

        for k, v in X.items():
            print(k, v.shape)
        for k, v in Y.items():
            print(k, v.shape)

        self.scaler = scaler

    def load_all(self, transform=True):
        x = self.X
        y = self.Y
        if self.scaler is not None and transform:
            x, y = self.scaler.transform(x, y)

        # Commented out to match with Multi-Star Generator above
        # x = [x[opt['name']] for opt in self.args['inputs']]
        # y = [y[opt['name']] for opt in self.args['outputs']]
        return x, y

    def __len__(self):
        return self.Y[list(self.Y.keys())[0]].shape[0] // self.batch_size

    def __getitem__(self, idx):
        x = {input_type: self.X[input_type][idx:idx + self.batch_size]
             for input_type in self.X}
        y = {output_type: self.Y[output_type][idx:idx + self.batch_size]
             for output_type in self.Y}

        return self.scaler.transform(x, y)

    def to_dataframe(self, include_inputs=False):
        x, y = self.load_all(transform=False)
        x = np.concatenate(x, axis=-1).squeeze()
        y = np.concatenate(y, axis=-1).squeeze()

        df = pd.DataFrame(data=np.concatenate(y, axis=-1).squeeze(),
                          columns=self.args['output_columns'])

        if include_inputs:
            df_inputs = pd.DataFrame(data=x, columns=self.args['input_columns'])
            df = pd.concat([df, df_inputs], axis=1)

        return df
