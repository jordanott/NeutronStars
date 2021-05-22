import joblib
import numpy as np
import tensorflow as tf
import neutron_stars as ns


class Predictor:
    def __init__(self):
        self.mr_scaler = joblib.load('SavedModels/mr+star2spectra/00011/mr_scaler')
        self.log_scaler = ns.data_loader.scaler_manager.LogScaler()

        self.model = tf.keras.models.load_model('SavedModels/mr+star2spectra/00011/')

    def __call__(self, mass, radius, nH, logTeff, dist):
        mass_radius = np.array([[mass, radius]])
        nuisance_parameters = np.array([[nH, logTeff, dist]])

        x = {'mass-radius': self.mr_scaler.transform(mass_radius),
             'nuisance-parameters': self.log_scaler.transform(nuisance_parameters)}

        predictions = self.model.predict(x)

        return self.log_scaler.inverse_transform(predictions)


if __name__ == '__main__':
    ns_predictor = Predictor()
    spectra = ns_predictor(mass=2.581471, radius=12.089365,
                           nH=0.013734, logTeff=6.273879, dist=6.011103)

    import matplotlib.pyplot as plt
    plt.plot(spectra.T)
    plt.savefig('test.png')
