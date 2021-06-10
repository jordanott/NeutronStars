import numpy as np
import onnxruntime as rt
from onnxruntime import InferenceSession


class RRegressor:
    def __init__(self):
        self.scaler_mean_ = np.array([ 4.98637894, -1.95408965,  1.96591633])
        self.scaler_scale_ = np.array([0.14018517, 0.05556016, 0.50481213])

        self.session = InferenceSession("SavedModels/mM2R/model_001.onnx")

        # self.sess_options = rt.SessionOptions()
        # self.sess_options.inter_op_num_threads = 1
        # self.sess_options.intra_op_num_threads = 1
        # self.sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        # self.session = InferenceSession("SavedModels/mM2R/model_001.onnx", sess_options=self.sess_options)


    def __call__(self, poi_m, mass):
        m_M = np.concatenate([poi_m, mass.reshape(-1,1)], axis=1)
        m_M = (inputs - scaler_mean_)/scaler_scale_

        x = {onnx_input_name :m_M.astype(np.float32)}

        predictions = self.session.run(None, x)[0]
        return predictions.ravel()


if __name__ == '__main__':
    R_regressor = RRegressor()
    test_poi_m = np.array([[ 5.00485518, -1.95035617],
       [ 5.00485518, -1.95035617],
       [ 5.00485518, -1.95035617],
       [ 5.00485518, -1.95035617],
       [ 5.00485518, -1.95035617],
       [ 5.00485518, -1.95035617],
       [ 5.00485518, -1.95035617],
       [ 5.00485518, -1.95035617],
       [ 5.00485518, -1.95035617],
       [ 5.00485518, -1.95035617]])
    test_mass = np.array([1.2047833, 1.2206364, 1.2365605, 1.2525521, 1.2686073, 1.2847223,
       1.3008932, 1.317116 , 1.3333865, 1.3497006])
    trueR = np.array([12.25506 , 12.253798, 12.252601, 12.251458, 12.250355, 12.249282,
       12.248225, 12.247172, 12.246113, 12.245034])
    predR = RRegressor(test_poi_m, test_mass)

    import matplotlib.pyplot as plt
    plt.scatter(test_mass, trueR,
                label="True", s=12, marker="o", color="r")
    plt.scatter(test_mass, predR,
                label="Pred", s=20, marker="x", color="blue")
    plt.xlabel("M")
    plt.ylabel("R")
    plt.title("True vs Predicted R")
    plt.savefig("test_RRegressor.png")
