import os
import pandas as pd
import tensorflow as tf


def gpu_settings(args):
    gpu = os.environ.get("SHERPA_RESOURCE", '')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu) or args['gpu']

    tf.get_logger().setLevel(3)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)


def dir_set_up(args):
    args['output_dir'] = os.path.join(args['output_dir'], args['paradigm']) + '/'

    if args['sherpa']:
        args['model_dir'] = args['output_dir'] + 'Models/%05d' % args['trial_id']
    else:
        args['trial_id'] = 1
        if args['run_type'] == 'train':
            while True:
                args['model_dir'] = args['output_dir'] + 'Models/%05d' % args['trial_id']
                if not os.path.isdir(args['model_dir']):
                    break
                args['trial_id'] += 1

    os.makedirs(args['model_dir'], exist_ok=True)
    os.makedirs(args['output_dir'] + 'Training/', exist_ok=True)
    os.makedirs(args['output_dir'] + 'Predictions/', exist_ok=True)


def store_training_history(history, args):
    if not args['sherpa']:
        df = pd.DataFrame(history)
        df.to_csv(args['output_dir'] + 'Training/%05d.csv' % args['trial_id'])


def store_predictions(x, y, predictions, args, data_partition='test'):
    x_df = pd.DataFrame(data=x, columns=args['input_columns'])
    y_df = pd.DataFrame(data=y, columns=args['output_columns'])
    p_df = pd.DataFrame(data=predictions, columns=[f'pred_{c}' for c in args['output_columns']])

    df = pd.concat([x_df, y_df, p_df], axis=1)
    df.to_csv(args['output_dir'] + f'Predictions/{data_partition}_%05d.csv' % args['trial_id'])


def predict_scale_store(generator, model, args, data_partition='test'):
    # LOAD THE PARTITION AND MAKE PREDICTIONS
    x, y = generator.load_all()
    y_hat = model.predict(x)

    # UNSCALE DATA
    x = generator.scaler.x_scaler.inverse_transform(x)
    y = generator.scaler.y_scaler.inverse_transform(y)
    y_hat = generator.scaler.y_scaler.inverse_transform(y_hat)

    # STORE INPUTS, TARGETS, PREDICTIONS
    store_predictions(x, y, y_hat, args, data_partition)
