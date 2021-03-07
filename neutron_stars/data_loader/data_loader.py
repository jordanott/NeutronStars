import random
import numpy as np
import tensorflow as tf
import neutron_stars as ns
from glob import iglob


class DataLoader:
    def __init__(self, args=None):
        self.args = args
        files = list(iglob(ns.DATA_DIR + '*.npz'))
        files = [
            f for f in files
            if f"{args['num_coefficients']}Param" in f
        ]
        random.Random(123).shuffle(files)

        num_train = int(len(files) * .8)
        num_valid = int(len(files) * .9)

        self.train_gen = DataGenerator(
            args=args,
            files=files[:num_train],
        )
        self.validation_gen = DataGenerator(
            args=args,
            files=files[num_train:num_valid],
        )
        self.test_gen = DataGenerator(
            args=args,
            files=files[num_valid:],
        )
        
        
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, args, files):
        self.args = args
        self.files = files
        self.batch_size = args['batch_size']

        self._create_mapping()

    def __len__(self):
        return len(self.mapping) // self.batch_size

    def __getitem__(self, idx):
        batch = self.mapping[idx:idx+self.batch_size]
        file_ids = np.unique(batch[:, 1]).astype(np.int)

        return self.get_batch(batch, file_ids)

    def load_all(self):
        batch = self.mapping
        file_ids = np.unique(batch[:, 1]).astype(np.int)

        return self.get_batch(batch, file_ids)

    def get_batch(self, batch, file_ids):
        X = np.empty((0, self.args['input_size']))
        Y = np.empty((0, self.args['output_size']))

        for file_id in file_ids:
            # GET ALL IDXs FROM SPECIFIC FILE
            idxs = batch[batch[:, 1] == file_id, 0].astype(np.int)
            # LOAD SPECIFIC FILE
            np_file = np.load(self.files[file_id])

            x_file_samples = np_file[self.args['input_key']][idxs]
            y_file_samples = np_file[self.args['output_key']][idxs]

            X = np.concatenate([
                X, x_file_samples[:, self.args['input_idxs']]
            ])
            Y = np.concatenate([
                Y, y_file_samples[:, self.args['output_idxs']]
            ])

        return X, Y

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
