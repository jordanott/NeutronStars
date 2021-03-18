import random
import numpy as np
import tensorflow as tf
import neutron_stars as ns
from glob import iglob
from sklearn.model_selection import KFold


class DataLoader:
    def __init__(self, args):
        self.args = args
        files = list(iglob(ns.DATA_DIR + '*.npz'))
        files = [
            f for f in files
            if f"{args['num_coefficients']}Param" in f
        ]
        random.Random(args['num_folds']).shuffle(files)

        num_train = int(len(files) * .8)
        num_valid = int(len(files) * .9)

        train_files = files[:num_train]
        validation_files = files[num_train:num_valid]
        test_files = files[num_valid:]

        if not args['sherpa']:
            kf = KFold(n_splits=args['num_folds'])
            for _ in range(args['fold']):
                train_idxs, val_idxs = next(kf.split(files))

            train_files = self.get_indices(files, train_idxs)
            validation_files = self.get_indices(files, val_idxs)

        self.train_gen = DataGenerator(
            args=args,
            files=train_files,
        )
        self.validation_gen = DataGenerator(
            args=args,
            files=validation_files,
            scaler=self.train_gen.scaler
        )
        self.test_gen = DataGenerator(
            args=args,
            files=test_files,
            scaler=self.train_gen.scaler
        )

    def get_indices(self, files, idxs):
        selected_files = []
        for idx in idxs:
            selected_files.append(files[idx])
        return selected_files
        
        
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, args, files, scaler=None):
        self.args = args
        self.files = files
        self.scaler = None
        self.batch_size = args['batch_size']

        self._create_mapping()
        if scaler is None:
            x, y = self.load_all()
            print(x.shape, y.shape)
            scaler = ns.data_loader.ScalerManager(args, x, y)
        self.scaler = scaler

    def __len__(self):
        return len(self.mapping) // self.batch_size

    def __getitem__(self, idx):
        batch = self.mapping[idx:idx+self.batch_size]
        file_ids = np.unique(batch[:, 1]).astype(np.int)

        return self.get_batch(batch, file_ids)

    def load_all(self, transform=True):
        batch = self.mapping
        file_ids = np.unique(batch[:, 1]).astype(np.int)

        return self.get_batch(batch, file_ids, transform)

    def get_batch(self, batch, file_ids, transform=True):
        X = np.empty((0, self.args['input_size']))
        Y = np.empty((0, self.args['output_size']))

        for file_id in file_ids:
            # GET ALL IDXs FROM SPECIFIC FILE
            idxs = batch[batch[:, 1] == file_id, 0].astype(np.int)
            # LOAD SPECIFIC FILE
            np_file = np.load(self.files[file_id])

            inputs = []
            for input_opt in self.args['inputs']:
                input_file_sample = np_file[input_opt['key']][idxs]
                inputs.append(input_file_sample[:, input_opt['idxs']])

            all_inputs = np.concatenate(inputs, axis=-1)

            outputs = []
            for output_opt in self.args['outputs']:
                output_file_sample = np_file[output_opt['key']][idxs]
                outputs.append(output_file_sample[:, output_opt['idxs']])

            all_outputs = np.concatenate(outputs, axis=-1)
            # x_file_samples = np_file[self.args['input_key']][idxs]
            # y_file_samples = np_file[self.args['output_key']][idxs]

            X = np.concatenate([
                X, all_inputs # x_file_samples[:, self.args['input_idxs']]
            ])
            Y = np.concatenate([
                Y, all_outputs # y_file_samples[:, self.args['output_idxs']]
            ])

        if self.scaler is None or not transform:
            return X, Y
        return self.scaler.transform(X, Y)

    def _create_mapping(self):
        """
        Creates a mapping for data samples and file ids
        self.mapping = [ [data sample index, file id] ]
        """
        file_ids = np.array([])
        samples_ids = np.array([])

        for file_id, file in enumerate(self.files):
            np_file = np.load(file)
            num_samples = np_file['star_nums'].shape[0]

            samples_ids = np.concatenate([
                samples_ids, np.arange(num_samples)
            ])

            file_ids = np.concatenate([
                file_ids, np.ones(num_samples) * file_id
            ])

        self.mapping = np.vstack([samples_ids, file_ids]).T


if __name__ == '__main__':
    args = ns.parse_args()
    ns.paradigm_settings(args)

    data_loader = DataLoader(args)
    X, Y = data_loader.train_gen[0]
    print('X:', X.shape)
    print('Y:', Y.shape)
