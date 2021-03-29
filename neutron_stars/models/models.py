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
        sherpa_callback = client.keras_send_metrics(trial,
                                                    objective_name='val_loss',
                                                    context_names=['loss', 'val_loss', 'mean_absolute_percentage_error',
                                                                   'val_mean_absolute_percentage_error'])
        callbacks.append(sherpa_callback)

    def schedule(epoch, lr):
        return lr * args['lr_decay']

    callbacks.extend([
        tf.keras.callbacks.LearningRateScheduler(schedule),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args['patience']),
        tf.keras.callbacks.ModelCheckpoint(args['model_dir'], save_best_only=True),
    ])
    return callbacks


def build_conv_branch(args, input_opts):
    input_size = len(input_opts['idxs'])
    activation = AVAILABLE_ACTIVATIONS[args['activation']]

    branch_input = x = tf.keras.layers.Input(shape=(input_size,), name=input_opts['name'])
    x = tf.keras.layers.Reshape((input_size, 1))(x)

    for layer_id in range(args['num_layers']):
        x = tf.keras.layers.Conv1D(
            filters=args['num_nodes'] // 8,
            kernel_size=5,
            strides=1,
            padding='same',
        )(x)
        x = activation(x)

        if (layer_id + 1) % 3 == 0:
            x = tf.keras.layers.MaxPool1D()(x)
        if args['batch_norm']:
            x = tf.keras.layers.BatchNormalization()(x)

    return branch_input, tf.keras.layers.Flatten()(x)


def build_dense_branch(args, branch_input=None, input_opts=None):
    activation = AVAILABLE_ACTIVATIONS[args['activation']]
    if branch_input is None:
        branch_input = x = tf.keras.layers.Input(
            shape=(len(input_opts['idxs'])),
            name=input_opts['name'])
    else:
        x = branch_input

    branch_outputs = [x]

    for layer_id in range(args['num_layers']):
        x = tf.keras.layers.Dense(args['num_nodes'])(x)
        x = activation(x)

        if args['batch_norm']:
            x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Dropout(args['dropout'])(x)

        if args['skip_connections'] and len(branch_outputs) > 1:
            x = tf.keras.layers.concatenate([
                branch_outputs[-2], x
            ])
        branch_outputs.append(x)

    return branch_input, branch_outputs[-1]


def build_model(args):
    model_inputs = []
    branch_outputs = []

    for input_opts in args['inputs']:
        if input_opts['name'] == 'spectra' and args['conv_branch']:
            branch_input, branch_output = build_conv_branch(args, input_opts)
        else:
            branch_input, branch_output = build_dense_branch(args, input_opts=input_opts)

        model_inputs.append(branch_input)
        branch_outputs.append(branch_output)

    if len(branch_outputs) > 1:
        branch_outputs = tf.keras.layers.concatenate(branch_outputs)
    else:
        branch_outputs = branch_outputs[0]

    _, model_output = build_dense_branch(args, branch_input=branch_outputs)

    x = tf.keras.layers.Dense(args['output_size'], name=args['outputs'][0]['name'])(model_output)
    return tf.keras.models.Model(inputs=model_inputs, outputs=x)
