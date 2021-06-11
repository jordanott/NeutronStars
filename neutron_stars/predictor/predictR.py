import numpy as np
import onnxruntime as rt


class RRegressor:
    def __init__(self):
        self.scaler_mean_ = np.array([ 4.98636989, -1.95408323,  1.9656553 ])
        self.scaler_scale_ = np.array([0.14012709, 0.055574  , 0.50454881])

        self.sess_options = rt.SessionOptions()
        self.sess_options.inter_op_num_threads = 1
        self.sess_options.intra_op_num_threads = 1
        self.sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        self.session = rt.InferenceSession("SavedModels/mM2R/model_002.onnx", sess_options=self.sess_options)
        
        self.onnx_input_name = self.session.get_inputs()[0].name


    def __call__(self, poi_m, mass):
        m_M = np.concatenate([poi_m, mass.reshape(-1,1)], axis=1)
        m_M = (m_M - self.scaler_mean_)/self.scaler_scale_

        x = {self.onnx_input_name :m_M.astype(np.float32)}

        predictions = self.session.run(None, x)[0]
        return predictions.ravel()


if __name__ == '__main__':
    r_regressor = RRegressor()
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
    predR = r_regressor(poi_m=test_poi_m, mass=test_mass)

    import matplotlib.pyplot as plt
    plt.scatter(test_mass, trueR,
                label="True", s=12, marker="o", color="r")
    plt.scatter(test_mass, predR,
                label="Pred", s=20, marker="x", color="blue")
    plt.xlabel("M")
    plt.ylabel("R")
    plt.xlim(1,1.5) #(1,3.5)
    plt.ylim(8,17)
    plt.title("True vs Predicted R")
    plt.legend()
    plt.savefig("test_RRegressor.png")
