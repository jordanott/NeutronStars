import joblib
import numpy as np
import tensorflow as tf
import neutron_stars as ns
from neutron_stars.predictor import ONNX
from onnxruntime import InferenceSession


class SpectraGenerator(ONNX):
    def __init__(self):
        super().__init__()

        self.mr_scaler = joblib.load('SavedModels/mr+star2spectra/00069/mass-radius')
        self.np_scaler = ns.data_loader.ZeroMean(
            mean=np.load('SavedModels/mr+star2spectra/00069/nuisance-parameters.npy'))
        self.spectra_scaler = ns.data_loader.ZeroMean(
            mean=np.load('SavedModels/mr+star2spectra/00069/spectra.npy'))

        self.session = InferenceSession("SavedModels/mr+star2spectra/00069_retrained/model.onnx",
                                        sess_options=self.sess_options)

    def __call__(self, mass, radius, nH, logTeff, dist):
        mass_radius = np.array([[mass, radius]])
        nuisance_parameters = np.array([[nH, logTeff, dist]])

        mass_radius = tf.convert_to_tensor(self.mr_scaler.transform(mass_radius),
                                           dtype=tf.float32)
        nuisance_parameters = tf.convert_to_tensor(self.np_scaler.transform(nuisance_parameters),
                                                   dtype=tf.float32)

        x = {'mass_radius:0': mass_radius.numpy(),
             'nuisance_parameters:0': nuisance_parameters.numpy()}

        predictions = self.session.run(None, x)[0]

        predictions = self.spectra_scaler.inverse_transform(predictions)
        return np.maximum(predictions, 0)


if __name__ == '__main__':
    spectra_generator = SpectraGenerator()
    spectra = spectra_generator(mass=2.581471, radius=12.089365,
                                nH=0.013734, logTeff=6.273879, dist=6.011103)

    import matplotlib.pyplot as plt
    plt.plot(spectra.T)
    plt.savefig('test.png')
