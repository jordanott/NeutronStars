import sherpa
import tensorflow as tf


AVAILABLE_ACTIVATIONS = {
    'elu': tf.nn.elu,
    'relu': tf.nn.relu,
    'leaky_relu': tf.nn.leaky_relu,
}


def create_callbacks(args):
    callbacks = []
    if args['sherpa']:
        client, trial = args['sherpa_info']
        sherpa_callback = client.keras_send_metrics(trial, objective_name='val_loss')
        callbacks.append(sherpa_callback)

    def schedule(epoch, lr):
        return lr * args['lr_decay']

    callbacks.extend([
        tf.keras.callbacks.LearningRateScheduler(schedule),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args['patience']),
        tf.keras.callbacks.ModelCheckpoint(args['model_dir'], save_best_only=True),
    ])
    return callbacks


def build_model(args):
    activation = AVAILABLE_ACTIVATIONS[args['activation']]
    model_input = x = tf.keras.layers.Input(shape=(args['input_size']))
    layer_outputs = [x]

    for layer_id in range(args['num_layers']):
        x = tf.keras.layers.Dense(args['num_nodes'])(x)
        x = activation(x)

        if args['batch_norm']:
            x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Dropout(args['dropout'])(x)

        if args['skip_connections'] and len(layer_outputs) > 1:
            x = tf.keras.layers.concatenate([
                layer_outputs[-2], x
            ])
        layer_outputs.append(x)

    x = tf.keras.layers.Dense(args['output_size'])(layer_outputs[-1])
    return tf.keras.models.Model(inputs=model_input, outputs=x)
