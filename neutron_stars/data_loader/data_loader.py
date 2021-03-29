import random
import numpy as np
import tensorflow as tf
import neutron_stars as ns
from glob import iglob
from sklearn.model_selection import KFold


class DataLoader:
    def __init__(self, args):
        self.args = args

        files = [f for f in list(iglob(ns.DATA_DIR + '*.npz'))
                 if f"{args['num_coefficients']}Param" in f]

        # INIT PLACEHOLDERS FOR DATA DICTIONARY
        X = {opts['name']: np.zeros((0, len(opts['idxs'])))
             for opts in args['inputs']}
        Y = {opts['name']: np.zeros((0, len(opts['idxs'])))
             for opts in args['outputs']}

        for file in files:
            np_file = np.load(file)

            for input_opt in args['inputs']:
                name = input_opt['name']                            # USE NAME FOR DATA MAP
                input_file_sample = np_file[input_opt['key']]       # USE KEY TO ACCESS NP FILE

                X[name] = np.concatenate([
                    X[name],
                    input_file_sample[:, input_opt['idxs']]
                ])

            for output_opt in args['outputs']:
                name = output_opt['name']                           # USE NAME FOR DATA MAP
                output_file_sample = np_file[output_opt['key']]     # USE KEY TO ACCESS NP FILE

                Y[name] = np.concatenate([
                    Y[name],
                    output_file_sample[:, output_opt['idxs']]
                ])

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

        x_train = {input_type: X[input_type][train_idxs] for input_type in X}
        y_train = {output_type: Y[output_type][train_idxs] for output_type in Y}
        x_test = {input_type: X[input_type][val_idxs] for input_type in X}
        y_test = {output_type: Y[output_type][val_idxs] for output_type in Y}

        self.train_gen = DataGenerator(x_train, y_train, args)
        self.validation_gen = DataGenerator(x_test, y_test, args, self.train_gen.scaler)


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
        if self.scaler is None or not transform:
            x, y = self.scaler.transform(x, y)

        x = np.concatenate([x[input_type] for input_type in self.X], axis=-1)
        y = np.concatenate([y[output_type] for output_type in self.Y], axis=-1)
        return x, y

    def __len__(self):
        return self.Y[list(self.Y.keys())[0]].shape[0] // self.batch_size

    def __getitem__(self, idx):
        x = {
            input_type: self.X[input_type][idx:idx+self.batch_size]
            for input_type in self.X
        }
        y = {
            output_type: self.Y[output_type][idx:idx + self.batch_size]
            for output_type in self.Y
        }

        return self.scaler.transform(x, y)


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
