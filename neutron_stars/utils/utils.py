import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf


def load_settings(args):
    stored_settings = {
        'gpu': args['gpu'],
        'sherpa': False,
        'run_type': args['run_type'],
        'output_dir': 'Results/',
        'load_settings_from': args['load_settings_from'],
    }
    if args['model_dir']:
        stored_settings['model_dir'] = args['model_dir']

    with open(args['load_settings_from'], 'r') as f:
        args.update(json.load(f))

    args.update(stored_settings)


def store_settings(args):
    settings = {}
    keys_to_ignore = ['sherpa_info', 'inputs', 'outputs', 'input_columns', 'output_columns']
    for k, v in args.items():
        if k not in keys_to_ignore:
            settings[k] = v

    with open(args['output_dir'] + 'Settings/%05d.json' % args['trial_id'], 'w') as f:
        f.write(json.dumps(settings, indent=4, sort_keys=True))


def gpu_settings(args={'gpu': '0'}):
    gpu = os.environ.get("SHERPA_RESOURCE", '')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu) or args['gpu']

    tf.get_logger().setLevel(3)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)


def dir_set_up(args):
    extra = f"_{args['model_type']}".replace('_normal', '')
    args['output_dir'] = os.path.join(args['output_dir'], args['paradigm'] + extra) + '/'

    if args['sherpa']:
        args['model_dir'] = args['output_dir'] + 'Models/%05d' % args['trial_id']
    else:
        if 'sample' in args['run_type']:
            pass
        elif args['load_settings_from']:
            args['model_dir'] = args['output_dir'] + 'Models/%05d' % args['trial_id']
        else:
            args['trial_id'] = 1
            if args['run_type'] == 'train':
                while True:
                    args['model_dir'] = args['output_dir'] + 'Models/%05d' % args['trial_id']
                    if not os.path.isdir(args['model_dir']):
                        break
                    args['trial_id'] += 1

    try:
        os.makedirs(args['model_dir'], exist_ok=True)
        os.makedirs(args['output_dir'] + 'Training/', exist_ok=True)
        os.makedirs(args['output_dir'] + 'Settings/', exist_ok=True)
        os.makedirs(args['output_dir'] + 'Predictions/', exist_ok=True)
    except:
        pass


def store_training_history(history, args):
    if not args['sherpa']:
        df = pd.DataFrame(history)
        df['fold'] = args['fold']
        df['Iteration'] = list(range(len(df)))

        history_file = args['output_dir'] + 'Training/%05d_%02d.csv' % (args['trial_id'], args['fold'])
        df.to_csv(history_file)

        print('History written to:', history_file)


def store_predictions(x, y, predictions, args, data_partition='test', save_inputs=False):
    if 'sample' in data_partition:
        columns = [f'pred_{c}' for c in args['output_columns']]
        columns.extend([f'pred_std_{c}' for c in args['output_columns']])
    else:
        columns = [f'pred_{c}' for c in args['output_columns']]

    y_df = pd.DataFrame(data=y, columns=args['output_columns'])
    p_df = pd.DataFrame(data=predictions, columns=columns)
    data_frames = [y_df, p_df]

    if save_inputs:
        x = np.concatenate(x, axis=-1).squeeze()
        x_df = pd.DataFrame(data=x, columns=args['input_columns'])
        data_frames.append(x_df)

    df = pd.concat(data_frames, axis=1)
    prediction_file = args['output_dir'] + f'Predictions/{data_partition}_%05d_%02d.csv' % (args['trial_id'], args['fold'])
    df.to_csv(prediction_file)

    print('Predictions written to:', prediction_file)


def predict_scale_store(generator, model, args, data_partition='test', save_inputs=False):
    # LOAD THE PARTITION AND MAKE PREDICTIONS
    x, _ = generator.load_all(transform=True)
    y_hat = model.predict(x, batch_size=args['batch_size'])
    x, y = generator.load_all(transform=False)

    # :SINGLE OUTPUT ASSUMPTION:
    output_name = args['outputs'][0]['name']
    y_hat = {output_name: y_hat}
    _, y_hat = generator.scaler.inverse_transform({}, y_hat)
    y_hat = y_hat[output_name]
    y = y[output_name]
    
    # STORE INPUTS, TARGETS, PREDICTIONS
    store_predictions(x, y, y_hat, args, data_partition, save_inputs=save_inputs)
