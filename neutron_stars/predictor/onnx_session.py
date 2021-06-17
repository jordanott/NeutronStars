import onnxruntime as rt


class ONNX:
    def __init__(self):
        sess_options = rt.SessionOptions()

        sess_options.inter_op_num_threads = 1
        sess_options.intra_op_num_threads = 1
        sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        # sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess_options = sess_options
