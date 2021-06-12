import sherpa
import tensorflow as tf
from .transformer import Transformer
from .common import AVAILABLE_ACTIVATIONS
from .custom import CosineDecayRestarts


class EarlyStoppingByValue(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_mean_absolute_percentage_error', value=2.4):
        super(tf.keras.callbacks.Callback, self).__init__()
        self.monitor = monitor
        self.value = value

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)

        if (current > 4*self.value) and epoch == 1:
            print("Epoch %05d: early stopping by value" % epoch)
            self.model.stop_training = True

        if current > self.value and epoch == 10:
            print("Epoch %05d: early stopping by value" % epoch)
            self.model.stop_training = True


def create_callbacks(args, monitor='val_loss'):
    callbacks = []
    if args['sherpa']:
        client, trial = args['sherpa_info']
        sherpa_callback = client.keras_send_metrics(trial,
                                                    objective_name='val_loss',
                                                    context_names=['loss', 'val_loss', 'mean_absolute_percentage_error',
                                                                   'val_mean_absolute_percentage_error',
                                                                   'val_eos_m1_metric', 'val_eos_m2_metric'])
        callbacks.append(sherpa_callback)

    def schedule(epoch, lr):
        return lr * args['lr_decay']

    callbacks.extend([
        tf.keras.callbacks.LearningRateScheduler(CosineDecayRestarts(args['lr'], first_decay_steps=1000), verbose=1),
        # tf.keras.callbacks.ReduceLROnPlateau(factor=args['lr_decay']),
        tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=args['patience']),
        tf.keras.callbacks.ModelCheckpoint(args['model_dir'], save_best_only=True),
    ])
    
    if '2none' in args['scaler_type']:
        callbacks.append(EarlyStoppingByValue(value=3 if args['num_coefficients'] == 4 else 2.5,
                                              monitor=monitor))

    return callbacks


def build_conv_branch(args, input_opts, branch_input=None, pooling=True):
    input_size = len(input_opts['idxs'])
    activation = AVAILABLE_ACTIVATIONS[args['activation']]

    if branch_input is None:
        branch_input = x = tf.keras.layers.Input(shape=(input_size,), name=input_opts['name'])
    else:
        x = branch_input

    x = tf.keras.layers.Reshape((input_size, 1))(x)

    for layer_id in range(args['num_layers']):
        x = tf.keras.layers.Conv1D(
            filters=args['num_nodes'] // 8,
            kernel_size=5,
            strides=1,
            padding='same',
        )(x)
        x = activation(x)

        if (layer_id + 1) % 3 == 0 and pooling:
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


def build_normal_model(args):
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

    if args['outputs'][0]['name'] == 'spectra' and args['conv_branch']:
        x = tf.keras.layers.Dense(args['output_size'], activation='relu')(branch_outputs)
        _, model_output = build_conv_branch(args, branch_input=x)
    else:
        _, model_output = build_dense_branch(args, branch_input=branch_outputs)
        x = tf.keras.layers.Dense(args['output_size'], name=args['outputs'][0]['name'])(model_output)

    return tf.keras.models.Model(inputs=model_inputs, outputs=x)


def build_model(args):
    if args['model_type'] == 'transformer':
        spectra_input = tf.keras.layers.Input(shape=(args['num_stars'], 250),
                                              name='spectra')
        # np_input = tf.keras.layers.Input(shape=(args['num_stars'], 3),
        #                                  name='nuisance-parameter')

        # model_input = tf.keras.layers.concatenate([spectra_input, np_input])

        output = Transformer(args)(spectra_input)
        model_output = tf.keras.layers.Dense(args['num_coefficients'], name='coefficients')(output)

        model = tf.keras.Model(inputs=spectra_input,
                               outputs=model_output)
    else:
        model = build_normal_model(args)

    return model
