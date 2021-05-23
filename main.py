"""
>>> tf2 main.py --paradigm spectra+star2eos
"""

import os
import tqdm
import glob
import pprint
import numpy as np
import tensorflow as tf
import neutron_stars as ns


# GET COMMAND LINE ARGS
args = ns.parse_args()

# SET UP THE DIRECTORY TO STORE RESULTS
ns.utils.dir_set_up(args)

# SHOW ALL ARG SETTINGS
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(args)

# DETERMINE INPUTS AND TARGETS FOR GIVEN PARADIGM
ns.paradigm_settings(args)

if args['run_type'] == 'train':
    ns.utils.store_settings(args)

    args['num_folds'] = 1
    for fold in range(1, args['num_folds']+1):
        args['fold'] = fold
        
        # BUILD THE DATA LOADER & PARTITION THE DATASET
        data_loader = ns.DataLoader(args)

        # GET tf.keras CALLBACKS
        callbacks = ns.models.create_callbacks(args)
        # BUILD MODEL ARCHITECTURE BASED ON ARGS
        model = ns.models.build_model(args)
        # PLOT MODEL
        tf.keras.utils.plot_model(model, args['model_dir'] + '/model.png', show_shapes=True)

        # COMPILE THE MODEL
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=args['lr']),
                      loss=args['loss_function'],
                      metrics=['mean_absolute_percentage_error', 'mse'])

        # TRAIN THE MODEL
        history = model.fit(x=data_loader.train_gen,
                            epochs=args['epochs'],
                            validation_data=data_loader.validation_gen,
                            callbacks=callbacks,
                            verbose=2 if args['sherpa'] else 1,
                            workers=16,
                            max_queue_size=10).history

        ns.utils.store_training_history(history, args)

        # LOAD THE BEST NETWORK FROM EARLY STOPPING
        model = tf.keras.models.load_model(args['model_dir'])

        ns.utils.predict_scale_store(data_loader.validation_gen, model, args, 'validation')
        # if args['sherpa']:
        #     ns.utils.predict_scale_store(data_loader.train_gen, model, args, 'train')

elif args['run_type'] == 'sample':
    """
    tf2 main.py --run_type sample --paradigm spectra+star2mr \
    --model_dir /baldig/physicstest/NeutronStarsData/SherpaResults/spectra+star2mr/Models/00470/ \
    --load_settings_from /baldig/physicstest/NeutronStarsData/SherpaResults/spectra+star2mr/Settings/00470.json \
    --mass_threshold 3 --gpu 3
    """
    args['fold'] = 1
    args['sherpa'] = True
    args['num_folds'] = 1

    num_eos_to_sample = 25
    num_samples_per_star = 50

    # BUILD THE DATA LOADER & PARTITION THE DATASET
    data_loader = ns.DataLoader(args)
    # LOAD MODEL
    model = tf.keras.models.load_model(args['model_dir'])

    for sample_type in ['knn_spectra', 'small_noise', 'poisson', 'empirical', 'uniform']:

        # :SINGLE OUTPUT ASSUMPTION:
        output_name = args['outputs'][0]['name']
        targets = np.zeros((0, args['output_size']))
        predictions = np.zeros((0, args['output_size'] * 2))

        # HOW ARE WE SAMPLING
        sample_fn = data_loader.sample_nuisance_parameters
        if sample_type == 'poisson':
            sample_fn = data_loader.sample_poisson_spectra

        for x, y, num_stars in tqdm.tqdm(data_loader.group_by_eos(num_samples=num_eos_to_sample),
                                         total=num_eos_to_sample):
            for idx in range(num_stars):

                x_sample, y_sample = sample_fn(x, y, idx,
                                               num_samples=num_samples_per_star,
                                               sample_type=sample_type)

                y_hat_sample = model.predict(x_sample, batch_size=args['batch_size'])

                y_hat_sample = {output_name: y_hat_sample}
                _, y_hat_sample = data_loader.train_gen.scaler.inverse_transform({}, y_hat_sample)
                y_hat_sample = y_hat_sample[output_name]

                # :SINGLE OUTPUT ASSUMPTION:
                y_hat_statistic = np.concatenate([np.mean(y_hat_sample, axis=0),
                                                  np.std(y_hat_sample, axis=0)],
                                                 axis=0).reshape(1, -1)

                # :SINGLE OUTPUT ASSUMPTION:
                targets = np.concatenate([targets,
                                         y[output_name][idx].reshape(1, -1)])

                predictions = np.concatenate([
                    predictions,
                    y_hat_statistic
                ], axis=0)

        ns.utils.store_predictions(x=None, y=targets,
                                   predictions=predictions, args=args,
                                   data_partition=args['run_type'] + '_' + sample_type)

elif args['run_type'] == 'test':
    """
    tf2 main.py --run_type test --paradigm mr2eos \
    --model_dir /baldig/physicstest/NeutronStarsData/SherpaResults/mr2eos/Models/00014 \
    """
    # LOAD MODEL FROM CROSS-VALIDATION
    for model_dir in glob.iglob(args['model_dir']):
        if not os.path.exists(model_dir + '/saved_model.pb'):
            continue

        args['load_settings_from'] = model_dir.replace('Models', 'Settings') + '.json'
        print('Model:', model_dir)
        print('Settings:', args['load_settings_from'])

        ns.utils.load_settings(args)

        args['fold'] = 1
        args['sherpa'] = True
        args['num_folds'] = 1
        args['output_dir'] = 'Results/mr2eos/' # /baldig/physicstest/NeutronStarsData/Sherpa

        # BUILD THE DATA LOADER & PARTITION THE DATASET
        data_loader = ns.DataLoader(args)

        model = tf.keras.models.load_model(model_dir)
        model.summary()

        ns.utils.predict_scale_store(data_loader.train_gen, model, args, 'train')
        ns.utils.predict_scale_store(data_loader.validation_gen, model, args, 'validation')
