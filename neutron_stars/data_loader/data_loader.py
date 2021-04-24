import random
import numpy as np
import pandas as pd
import tensorflow as tf
import neutron_stars as ns
from glob import iglob
from sklearn.model_selection import KFold


class DataLoader:
    def __init__(self, args):
        self.args = args

        # GET ALL FILES WITH CORRECT NUM OF EOS PARAMS
        files = [f for f in list(iglob(ns.DATA_DIR + '*.npz'))
                 if f"{args['num_coefficients']}Param" in f]

        if args['run_type'] == 'sample':
            files = files[:10]

        # INIT PLACEHOLDERS FOR DATA DICTIONARY
        X = {opts['name']: np.zeros((0, len(opts['idxs'])))
             for opts in args['inputs']}
        Y = {opts['name']: np.zeros((0, len(opts['idxs'])))
             for opts in args['outputs']}

        self.eos = np.zeros((0, args['num_coefficients']))

        for file in files:
            np_file = np.load(file)
            mass_threshold = (np_file['details'][:, 0] < args['mass_threshold'])

            for input_opt in args['inputs']:
                name = input_opt['name']                            # USE NAME FOR DATA MAP
                input_file_sample = np_file[input_opt['key']]       # USE KEY TO ACCESS NP FILE

                new_sample = input_file_sample[:, input_opt['idxs']]
                new_sample = new_sample[mass_threshold]

                X[name] = np.concatenate([X[name], new_sample])

            for output_opt in args['outputs']:
                name = output_opt['name']                           # USE NAME FOR DATA MAP
                output_file_sample = np_file[output_opt['key']]     # USE KEY TO ACCESS NP FILE

                new_sample = output_file_sample[:, output_opt['idxs']]
                new_sample = new_sample[mass_threshold]

                Y[name] = np.concatenate([Y[name], new_sample])

            if args['run_type'] == 'sample':
                self.eos = np.concatenate([self.eos, np_file['coefficients'][mass_threshold]])

        if 'spectra' in X:
            X['spectra'] *= 10000.

        num_samples = len(X[list(X.keys())[0]])
        example_input = np.zeros((num_samples,))

        if args['sherpa']:
            np.random.seed(args['num_folds'])
            idxs = np.arange(num_samples)
            np.random.shuffle(idxs)

            num_train = int(num_samples * .8)
            train_idxs = idxs[:num_train]
            val_idxs = idxs[num_train:]
        else:
            kf = KFold(n_splits=10)
            for _ in range(args['fold']):
                train_idxs, val_idxs = next(kf.split(example_input))

        self.X = X
        self.Y = Y

        x_train = {input_type: X[input_type][train_idxs] for input_type in X}
        y_train = {output_type: Y[output_type][train_idxs] for output_type in Y}
        x_test = {input_type: X[input_type][val_idxs] for input_type in X}
        y_test = {output_type: Y[output_type][val_idxs] for output_type in Y}

        self.train_gen = DataGenerator(x_train, y_train, args)
        self.validation_gen = DataGenerator(x_test, y_test, args, self.train_gen.scaler)

    def group_by_eos(self, num_samples=10):
        unique_eos = np.unique(self.eos, axis=0)
        if num_samples != 'all':
            unique_eos = unique_eos[:num_samples]

        for eos in unique_eos:
            # Find idx of matching eos coefficients
            match = np.all(self.eos == eos, axis=1)

            x = {input_type: self.X[input_type][match]
                 for input_type in self.X}
            y = {output_type: self.Y[output_type][match]
                 for output_type in self.Y}

            # Return stars from matching eos and the number of stars
            yield x, y, np.sum(match)

    def sample_nuisance_parameters(self, x, y, idx, num_samples=10, sample_type='empirical'):
        # Repeat indexed data point with tiling
        x = {input_type: np.tile(x[input_type][idx], (num_samples, 1))
             for input_type in x}
        y = {output_type: np.tile(y[output_type][idx], (num_samples, 1))
             for output_type in y}

        # Sample nuisance parameters
        if sample_type == 'uniform':
            np_min = np.min(self.X['nuisance-parameters'], axis=0)
            np_max = np.max(self.X['nuisance-parameters'], axis=0)

            x['nuisance-parameters'] = np.stack([
                np.random.uniform(np_min[i], np_max[i], size=num_samples)
                for i in range(len(np_min))
            ]).T

        elif sample_type == 'small_noise':
            x['nuisance-parameters'] += np.random.uniform(-.01, .01, size=(num_samples, 3))

        elif sample_type == 'empirical':
            x['nuisance-parameters'] = np.stack([np.random.choice(self.X['nuisance-parameters'][:, 0], num_samples),
                                                 np.random.choice(self.X['nuisance-parameters'][:, 1], num_samples),
                                                 np.random.choice(self.X['nuisance-parameters'][:, 2], num_samples)]).T

        return self.train_gen.scaler.transform(x, y)

    def sample_poisson_spectra(self, x, y, idx, num_samples=10, **kwargs):
        # Repeat indexed data point with tiling
        x = {input_type: np.tile(x[input_type][idx], (num_samples, 1))
             for input_type in x}
        y = {output_type: np.tile(y[output_type][idx], (num_samples, 1))
             for output_type in y}

        x['spectra'] = np.random.poisson(x['spectra'] * 100) / 100.

        return self.train_gen.scaler.transform(x, y)


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

    def to_dataframe(self, inputs=False):
        y = [self.Y[opt['name']] for opt in self.args['outputs']]
        columns = [c for c in self.args['output_columns']]

        df = pd.DataFrame(data=np.concatenate(y, axis=-1), columns=columns)

        if inputs:
            x = np.concatenate([
                self.X[opt['name']]
                for opt in self.args['inputs']],
                axis=-1
            )
            df_inputs = pd.DataFrame(data=x, columns=self.args['input_columns'])
            df = pd.concat([df, df_inputs], axis=1)

        return df


if __name__ == '__main__':
    args = ns.parse_args()
    ns.paradigm_settings(args)
    args['fold'] = 1
    args['num_folds'] = 1

    print(
        len(ns.data_loader.scaler_combinations_for_paradigm(args['paradigm']))
    )

    data_loader = DataLoader(args)
    X, Y = data_loader.train_gen[0]

    for k, v in X.items():
        print(k, v.shape)
    for k, v in Y.items():
        print(k, v.shape)

    X, Y = data_loader.train_gen.load_all()
    print(X.shape, Y.shape)
