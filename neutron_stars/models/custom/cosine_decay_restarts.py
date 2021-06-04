import math
import tensorflow as tf

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops


class CosineDecayRestarts(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Source:
    https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/python/keras/optimizer_v2/learning_rate_schedule.py#L648-L770
    """
    def __init__(
            self,
            initial_learning_rate,
            first_decay_steps,
            t_mul=2.0,
            m_mul=1.0,
            alpha=0.0,
            name=None):
        """Applies cosine decay with restarts to the learning rate.
        Args:
          initial_learning_rate: A scalar `float32` or `float64` Tensor or a Python
            number. The initial learning rate.
          first_decay_steps: A scalar `int32` or `int64` `Tensor` or a Python
            number. Number of steps to decay over.
          t_mul: A scalar `float32` or `float64` `Tensor` or a Python number.
            Used to derive the number of iterations in the i-th period
          m_mul: A scalar `float32` or `float64` `Tensor` or a Python number.
            Used to derive the initial learning rate of the i-th period:
          alpha: A scalar `float32` or `float64` Tensor or a Python number.
            Minimum learning rate value as a fraction of the initial_learning_rate.
          name: String. Optional name of the operation.  Defaults to 'SGDRDecay'.
        """
        super(CosineDecayRestarts, self).__init__()

        self.initial_learning_rate = initial_learning_rate
        self.first_decay_steps = first_decay_steps
        self._t_mul = t_mul
        self._m_mul = m_mul
        self.alpha = alpha
        self.name = name

    def __call__(self, step):
        with ops.name_scope_v2(self.name or "SGDRDecay") as name:
            initial_learning_rate = ops.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate")
            dtype = initial_learning_rate.dtype
            first_decay_steps = math_ops.cast(self.first_decay_steps, dtype)
            alpha = math_ops.cast(self.alpha, dtype)
            t_mul = math_ops.cast(self._t_mul, dtype)
            m_mul = math_ops.cast(self._m_mul, dtype)

            global_step_recomp = math_ops.cast(step, dtype)
            completed_fraction = global_step_recomp / first_decay_steps

            def compute_step(completed_fraction, geometric=False):
                """Helper for `cond` operation."""
                if geometric:
                    i_restart = math_ops.floor(
                        math_ops.log(1.0 - completed_fraction * (1.0 - t_mul)) /
                        math_ops.log(t_mul))

                    sum_r = (1.0 - t_mul ** i_restart) / (1.0 - t_mul)
                    completed_fraction = (completed_fraction - sum_r) / t_mul ** i_restart

                else:
                    i_restart = math_ops.floor(completed_fraction)
                    completed_fraction -= i_restart

                return i_restart, completed_fraction

            i_restart, completed_fraction = control_flow_ops.cond(
                math_ops.equal(t_mul, 1.0),
                lambda: compute_step(completed_fraction, geometric=False),
                lambda: compute_step(completed_fraction, geometric=True))

            m_fac = m_mul ** i_restart
            cosine_decayed = 0.5 * m_fac * (1.0 + math_ops.cos(
                constant_op.constant(math.pi) * completed_fraction))
            decayed = (1 - alpha) * cosine_decayed + alpha

            return math_ops.multiply(initial_learning_rate, decayed, name=name)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "first_decay_steps": self.first_decay_steps,
            "t_mul": self._t_mul,
            "m_mul": self._m_mul,
            "alpha": self.alpha,
            "name": self.name
        }
