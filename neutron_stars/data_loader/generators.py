import numpy as np
import pandas as pd
import tensorflow as tf
import neutron_stars as ns


class BlankScaler:
    def inverse_transform(self, x, y): return x, y


class ManyStarsGenerator(tf.keras.utils.Sequence):
    def __init__(self, args, generator):
        X = []
        Y = []
        for x, y, _ in generator:
            X.append(x)
            Y.append(y)

        self.X = X
        self.Y = Y
        self.args = args
        self.scaler = BlankScaler()
        self.num_stars = args['num_stars']
        self.batch_size = args['batch_size']
        self.seed = 1
        np.random.seed(self.seed)

    def on_epoch_end(self):
        self.seed += 1
        np.random.seed(self.seed)

    def __len__(self):
        count = 0
        for x in self.X:
            count += len(x['mass-radius'])
        return count // self.batch_size

    def __getitem__(self, _):
        batch_x = np.zeros((self.batch_size, self.num_stars, 2))
        batch_y = np.zeros((self.batch_size, 2))
        eos_idxs = np.arange(len(self.X))

        for bidx in range(self.batch_size):
            eos_idx = np.random.choice(eos_idxs)
            x = self.X[eos_idx]['mass-radius']
            y = self.Y[eos_idx]['coefficients']

            star_idx = np.random.choice(np.arange(len(x)),
                                        size=self.num_stars,
                                        replace=False)

            batch_x[bidx] = np.expand_dims(x[star_idx], axis=0)
            batch_y[bidx] = np.expand_dims(y[0], axis=0)

        x = {'mass-radius': batch_x}    # .reshape(self.batch_size, -1)
        y = {'coefficients': batch_y}

        return x, y

    def load_all(self, transform=True):
        np.random.seed(123)
        size = self.__len__()
        all_x = np.zeros((size, self.num_stars, 2))
        all_y = np.zeros((size, 2))
        for i in range(0, size, self.batch_size):
            batch_x, batch_y = self.__getitem__(i)
            batch_size = all_x[i:i + self.batch_size].shape[0]

            all_x[i:i + batch_size] = batch_x['mass-radius'][:batch_size]
            all_y[i:i + batch_size] = batch_y['coefficients'][:batch_size]

        return all_x, [all_y]


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

        if scaler is None:
            scaler = ns.data_loader.ScalerManager(args, X, Y)
        self.scaler = scaler

    def load_all(self, transform=True):
        x = self.X
        y = self.Y
        if self.scaler is not None and transform:
            x, y = self.scaler.transform(x, y)

        x = [x[opt['name']] for opt in self.args['inputs']]
        y = [y[opt['name']] for opt in self.args['outputs']]
        return x, y

    def __len__(self):
        return self.Y[list(self.Y.keys())[0]].shape[0] // self.batch_size

    def __getitem__(self, idx):
        x = {
            input_type: self.X[input_type][idx:idx + self.batch_size]
            for input_type in self.X
        }
        y = {
            output_type: self.Y[output_type][idx:idx + self.batch_size]
            for output_type in self.Y
        }

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
