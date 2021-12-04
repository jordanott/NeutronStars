"""
Source: https://www.tensorflow.org/tutorials/text/transformer#scaled_dot_product_attention
"""
import tensorflow as tf
from neutron_stars.models.common import AVAILABLE_ACTIVATIONS


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, args):
        super(TransformerBlock, self).__init__()
        activation = AVAILABLE_ACTIVATIONS[args['activation']]

        self.multi_head_attention = MultiHeadAttention(d_model=args['num_nodes'], num_heads=8)
        self.dropout_1 = tf.keras.layers.Dropout(args['dropout'])
        self.normalization_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.pw_ff = tf.keras.Sequential([
            tf.keras.layers.Dense(args['num_nodes'], activation=activation),  # (batch_size, seq_len, embedding_dim)
            tf.keras.layers.Dense(args['num_nodes'])  # (batch_size, seq_len, embedding_dim)
        ])
        self.dropout_2 = tf.keras.layers.Dropout(args['dropout'])
        self.normalization_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        attn_output, attn = self.multi_head_attention(x, k=x, q=x, mask=None)
        attn_output = self.dropout_1(attn_output)
        norm_output = self.normalization_1(x + attn_output)

        pw_ff_output = self.pw_ff(norm_output)
        pw_ff_output = self.dropout_2(pw_ff_output)
        norm_output = self.normalization_2(norm_output + pw_ff_output)

        return norm_output


class Transformer(tf.keras.Model):
    def __init__(self, args):
        super(Transformer, self).__init__()
        activation = AVAILABLE_ACTIVATIONS[args['activation']]

        self.input_embedding = tf.keras.layers.Dense(args['num_nodes'], activation=activation)
        self.transformer = tf.keras.models.Sequential([TransformerBlock(args)
                                                       for _ in range(args['num_layers'])])

        self.op = {'gather': None,
                   'max': tf.reduce_max,
                   'min': tf.reduce_min}[args['transformer_op']]

    def call(self, x):
        e = self.input_embedding(x)
        x = self.transformer(e)

        if self.op is None:
            x = tf.gather(x, [0], axis=1)
        else:
            x = self.op(x, axis=1)

        return x


if __name__ == '__main__':
    import numpy as np
    import neutron_stars as ns
    import matplotlib.pyplot as plt

    ns.utils.gpu_settings()

    star_dim = 2
    embedding_dim = 32
    num_stars_in_set = 5

    args = {'num_layers': 5,
            'num_nodes': 32,
            'transformer_op': 'max',
            'activation': 'relu',
            'dropout': .25}

    model_input = tf.keras.layers.Input(shape=(num_stars_in_set, star_dim), name='mass-radius')
    output = Transformer(args)(model_input)
    model_output = tf.keras.layers.Dense(2, name='coefficients')(output)
    model = tf.keras.Model(inputs=model_input, outputs=model_output)

    model.compile(loss='mse', optimizer='adam')
    model.summary()

    tf.keras.utils.plot_model(model, expand_nested=True)
    print(output.shape)
    print(model.inputs)

    x = np.ones((100, num_stars_in_set, star_dim))
    y = np.ones((100, 2))
    for i in range(100):
        x[i] *= i / 10
        y[i] *= i / 10

    model.fit(x=x, y=y, epochs=50)
    print(model.predict(x))
    # _, axs = plt.subplots(2, 4)
    # for i, ax in zip(range(8), axs.flatten()):
    #     ax.imshow(attn.numpy()[0, i])
    #
    # plt.savefig('attention_sample_transformer.png')
