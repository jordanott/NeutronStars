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
    os.makedirs(args['output_dir'] + 'Models/', exist_ok=True)
    os.makedirs(args['output_dir'] + 'Training/', exist_ok=True)
    os.makedirs(args['output_dir'] + 'Predictions/', exist_ok=True)

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


def store_training_history(history, args):
    if not args['sherpa']:
        df = pd.DataFrame(history)
        df.to_csv(args['output_dir'] + 'Training/%05d.csv' % args['trial_id'])


def store_predictions(x, y, predictions, args):
    x_df = pd.DataFrame(data=x, columns=args['input_columns'])
    y_df = pd.DataFrame(data=y, columns=args['output_columns'])
    p_df = pd.DataFrame(data=predictions, columns=[f'pred_{c}' for c in args['output_columns']])

    df = pd.concat([x_df, y_df, p_df], axis=1)
    df.to_csv(args['output_dir'] + 'Predictions/%05d.csv' % args['trial_id'])
