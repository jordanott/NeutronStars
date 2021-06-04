import time
import tensorflow as tf
from onnxruntime import InferenceSession


model = tf.keras.models.load_model('SavedModels/mr+star2spectra/00011')
session = InferenceSession("SavedModels/mr+star2spectra/00011.onnx")

start_time = time.time()
for i in range(100):
    got = session.run(None, {'mass_radius:0': tf.convert_to_tensor([[5, 6.]]).numpy(),
                             'nuisance_parameters:0': tf.convert_to_tensor([[1, 2, 3.]]).numpy()})
print(time.time() - start_time)


start_time = time.time()
for i in range(100):
    model.predict({'mass-radius': tf.convert_to_tensor([[5, 6.]]).numpy(),
                   'nuisance-parameters': tf.convert_to_tensor([[1, 2, 3.]]).numpy()})
print(time.time() - start_time)
