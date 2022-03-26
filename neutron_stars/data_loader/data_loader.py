import copy
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import neutron_stars as ns
from glob import iglob
from .generators import *
from sklearn.model_selection import KFold


class DataLoader:
    def __init__(self, args, small=False):
        self.args = args

        # GET ALL FILES WITH CORRECT NUM OF EOS PARAMS
        files = [f for f in list(iglob(args['data_dir'] + '*.npz'))
                 if f"{args['num_coefficients']}Param" in f]

        print(len(files))
        # files += [f for f in list(iglob('/baldig/physicstest/NeutronStarsData/res/*.npz'))
        #           if f"{args['num_coefficients']}Param" in f]

        if small:
            files = files[:10]

        # INIT PLACEHOLDERS FOR DATA DICTIONARY
        # X = {opts['name']: np.zeros((0, len(opts['idxs'])))
        #      for opts in args['inputs']}
        # Y = {opts['name']: np.zeros((0, len(opts['idxs'])))
        #      for opts in args['outputs']}
        #
        # self.eos = np.zeros((0, args['num_coefficients']))
        #
        # for file in files:
        #     np_file = np.load(file)
        #     mass_threshold = (np_file['details'][:, 0] < args['mass_threshold'])
        #
        #     for input_opt in args['inputs']:
        #         name = input_opt['name']                            # USE NAME FOR DATA MAP
        #         input_file_sample = np_file[input_opt['key']]       # USE KEY TO ACCESS NP FILE
        #
        #         new_sample = input_file_sample[:, input_opt['idxs']]
        #         new_sample = new_sample[mass_threshold]
        #
        #         if 'spectra' == name and 'res_nonoise10' in file:
        #             new_sample *= 1e5
        #
        #         X[name] = np.concatenate([X[name], new_sample])
        #
        #     for output_opt in args['outputs']:
        #         name = output_opt['name']                           # USE NAME FOR DATA MAP
        #         output_file_sample = np_file[output_opt['key']]     # USE KEY TO ACCESS NP FILE
        #
        #         new_sample = output_file_sample[:, output_opt['idxs']]
        #         new_sample = new_sample[mass_threshold]
        #
        #         if 'spectra' == name and 'res_nonoise10' in file:
        #             new_sample *= 1e5
        #
        #         Y[name] = np.concatenate([Y[name], new_sample])
        #
        #     self.eos = np.concatenate([self.eos, np_file['coefficients'][mass_threshold]])

        X, Y, self.eos = ns.data_loader.faster_data_loader(args)
        
        num_samples = len(X[list(X.keys())[0]])
        example_input = np.zeros((num_samples,))

        np.random.seed(args['num_folds'])
        idxs = np.arange(num_samples)
        np.random.shuffle(idxs)

        X = {input_type: X[input_type][idxs] for input_type in X}
        Y = {output_type: Y[output_type][idxs] for output_type in Y}

        if args['sherpa']:
            num_train = int(num_samples * .8)
            train_idxs = idxs[:num_train]
            val_idxs = idxs[num_train:]
        else:
            kf = KFold(n_splits=5)
            for _ in range(args['fold']):
                train_idxs, val_idxs = next(kf.split(example_input))

        self.X = X
        self.Y = Y

        x_train = {input_type: X[input_type][train_idxs] for input_type in X}
        y_train = {output_type: Y[output_type][train_idxs] for output_type in Y}
        eos_train = self.eos[train_idxs]

        x_test = {input_type: X[input_type][val_idxs] for input_type in X}
        y_test = {output_type: Y[output_type][val_idxs] for output_type in Y}
        eos_test = self.eos[val_idxs]

        train_scaler = ns.data_loader.ScalerManager(args, x_train, y_train)

        if args['model_type'] == 'transformer':
            self.train_gen = ManyStarsGenerator(args, self.group_by_eos(x_train, y_train, eos_train, 'all'),
                                                scaler=train_scaler)
            self.validation_gen = ManyStarsGenerator(args, self.group_by_eos(x_test, y_test, eos_test, 'all'),
                                                     scaler=train_scaler)
            self.all_gen = ManyStarsGenerator(
                args,
                self.group_by_eos(self.X, self.Y, self.eos, 'all'),
                scaler=train_scaler
            )
            print('Lengths:', len(self.train_gen), len(self.validation_gen), self.validation_gen.count_all_stars())
        else:
            self.train_gen = DataGenerator(x_train, y_train, args, scaler=train_scaler)
            self.validation_gen = DataGenerator(x_test, y_test, args, scaler=train_scaler)
            self.all_gen = DataGenerator(self.X, self.Y, args, scaler=train_scaler)

    def group_by_eos(self, X=None, Y=None, EOS=None, num_samples=10):
        if X is None:
            X = self.X; Y = self.Y; EOS = self.eos

        unique_eos = np.unique(EOS, axis=0)
        if num_samples != 'all':
            unique_eos = unique_eos[:num_samples]

        for eos in unique_eos:
            # Find idx of matching eos coefficients
            match = np.all(EOS == eos, axis=1)

            x = {input_type: X[input_type][match]
                 for input_type in X}
            y = {output_type: Y[output_type][match]
                 for output_type in Y}

            # Return stars from matching eos and the number of stars
            yield x, y, np.sum(match)

    def sample(self, x, sample_type='empirical', sample_format=None, logTeff=0.):
        x = copy.deepcopy(x)
        np_min = np.min(self.X['nuisance-parameters'], axis=0)
        np_max = np.max(self.X['nuisance-parameters'], axis=0)

        # Sample nuisance parameters
        if sample_type == "poisson":
            x['spectra'] = np.random.poisson(x['spectra'] * 100) / 100.

        elif sample_type == 'uniform':
            x['nuisance-parameters'] = np.random.uniform(
                np_min, np_max, size=x["nuisance-parameters"].shape
            )

        elif sample_type == "uncertainty":
            # 'nH', 'logTeff', 'dist'
            uncertainty = np.random.uniform(
                [0.5, 1, 0.9],
                [1.5, 1, 1.1],
                size=x["nuisance-parameters"].shape
            )
            x['nuisance-parameters'] = x['nuisance-parameters'] * uncertainty

        elif sample_type == 'small_noise':
            # 'nH', 'logTeff', 'dist'
            uncertainty = np.random.uniform(
                [0.95, 0.95, 0.95],
                [1.05, 1.05, 1.05],
                size=x["nuisance-parameters"].shape
            )
            x['nuisance-parameters'] = x['nuisance-parameters'] * uncertainty

        elif sample_type == 'empirical':
            size = x["nuisance-parameters"].shape[:-1]
            x['nuisance-parameters'] = np.stack([
                np.random.choice(self.X['nuisance-parameters'][:, 0], size=size[::-1]),
                np.random.choice(self.X['nuisance-parameters'][:, 1], size=size[::-1]),
                np.random.choice(self.X['nuisance-parameters'][:, 2], size=size[::-1])
            ]).T

        elif sample_type == 'knn_spectra':
            distances = np.sum(np.abs(self.X['spectra'] - x['spectra'][0]),
                               axis=1)
            idx = np.argsort(distances)[:num_samples]
            x['nuisance-parameters'] = self.X['nuisance-parameters'][idx]

        elif type(sample_type) is float:
            uncertainty = np.random.uniform(
                [1 - sample_type, 1, 1 - sample_type],
                [1 + sample_type, 1, 1 + sample_type],
                size=x["nuisance-parameters"].shape
            )
            x['nuisance-parameters'] = x['nuisance-parameters'] * uncertainty
        elif sample_type == "tight":
            uncertainty = np.random.uniform(
                [0.7, 0.95, 0.95],
                [1.3, 1.05, 1.05],
                size=x["nuisance-parameters"].shape
            )
            uncertainty = np.array([0.3, logTeff, 0.05]) * sample_format
            uncertainty += np.ones(3)

            x['nuisance-parameters'] = np.clip(
                x['nuisance-parameters'] * uncertainty,
                np_min,
                np_max
            )
        elif sample_type == "loose":
            uncertainty = np.random.uniform(
                [0.5, 0.9, 0.8],
                [1.5, 1.1, 1.2],
                size=x["nuisance-parameters"].shape
            )
            uncertainty = np.array([0.5, logTeff, 0.2]) * sample_format
            uncertainty += np.ones(3)

            x['nuisance-parameters'] = np.clip(
                x['nuisance-parameters'] * uncertainty,
                np_min,
                np_max
            )

        return x

    def sample_poisson_spectra(self, x, y, idx, num_samples=10, **kwargs):
        # Repeat indexed data point with tiling
        x = {input_type: np.tile(x[input_type][idx], (num_samples, 1))
             for input_type in x}
        y = {output_type: np.tile(y[output_type][idx], (num_samples, 1))
             for output_type in y}

        x['spectra'] = np.random.poisson(x['spectra'] * 100) / 100.

        return self.train_gen.scaler.transform(x, y)


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
