import joblib
import numpy as np
import tensorflow as tf
import neutron_stars as ns
from onnxruntime import InferenceSession


class SpectraGenerator:
    def __init__(self):
        self.mr_scaler = joblib.load('SavedModels/mr+star2spectra/00011/mr_scaler')
        self.log_scaler = ns.data_loader.scaler_manager.LogScaler()

        self.session = InferenceSession("SavedModels/mr+star2spectra/00011.onnx")

    def __call__(self, mass, radius, nH, logTeff, dist):
        mass_radius = np.array([[mass, radius]])
        nuisance_parameters = np.array([[nH, logTeff, dist]])

        mass_radius = tf.convert_to_tensor(self.mr_scaler.transform(mass_radius),
                                           dtype=tf.float32)
        nuisance_parameters = tf.convert_to_tensor(self.log_scaler.transform(nuisance_parameters),
                                                   dtype=tf.float32)

        x = {'mass_radius:0': mass_radius.numpy(),
             'nuisance_parameters:0': nuisance_parameters.numpy()}

        predictions = self.session.run(None, x)[0]

        predictions = self.log_scaler.inverse_transform(predictions)
        return np.maximum(predictions, 0)


if __name__ == '__main__':
    spectra_generator = SpectraGenerator()
    spectra = spectra_generator(mass=2.581471, radius=12.089365,
                                nH=0.013734, logTeff=6.273879, dist=6.011103)

    import matplotlib.pyplot as plt
    plt.plot(spectra.T)
    plt.savefig('test.png')
