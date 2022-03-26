"""
>>> tf2 main.py --paradigm spectra+star2eos
"""

import os
import tqdm
import glob
import pprint
import numpy as np
import pandas as pd
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

args["data_dir"] = "/baldig/physicstest/NeutronStarsData/res_nonoise10*/"

if args['run_type'] == 'train':
    ns.utils.store_settings(args)

    args['num_folds'] = 1
    for fold in range(1, args['num_folds']+1):
        args['fold'] = fold
        
        # BUILD THE DATA LOADER & PARTITION THE DATASET
        data_loader = ns.DataLoader(args)

        # GET tf.keras CALLBACKS
        callbacks = ns.models.create_callbacks(args, monitor='loss')
        # BUILD MODEL ARCHITECTURE BASED ON ARGS
        model = ns.models.build_model(args)
        # PLOT MODEL
        tf.keras.utils.plot_model(model, args['model_dir'] + '/model.png', show_shapes=True)

        metrics = ['mean_absolute_percentage_error', 'mse']
        if '2eos' in args['paradigm']:
            metrics.extend([ns.models.custom.eos_m1_metric,
                            ns.models.custom.eos_m2_metric])

        # COMPILE THE MODEL
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=args['lr']),
                      loss=args['loss_function'],
                      metrics=metrics)
        print(model.summary())

        # TRAIN THE MODEL
        history = model.fit(x=data_loader.train_gen,
                            epochs=args['epochs'],
                            validation_data=data_loader.validation_gen,
                            callbacks=callbacks,
                            verbose=2,# if args['sherpa'] else 1,
                            workers=16,
                            max_queue_size=10).history

        ns.utils.store_training_history(history, args)

        # LOAD THE BEST NETWORK FROM EARLY STOPPING
        model = tf.keras.models.load_model(args['model_dir'],
                                           custom_objects={'eos_m1_metric': ns.models.custom.eos_m1_metric,
                                                           'eos_m2_metric': ns.models.custom.eos_m1_metric})

        ns.utils.predict_scale_store(data_loader.validation_gen, model, args, 'validation')
        # if args['sherpa']:
        #     ns.utils.predict_scale_store(data_loader.train_gen, model, args, 'train')

elif args['run_type'] == 'sample':
    """
    tf2 main.py --run_type sample --paradigm spectra+star2eos \
        --model_dir /baldig/physicstest/NeutronStarsData/SherpaResults/spectra+star2eos_transformer_refine_v3.1_00165/Models/00011/ \
        --load_settings_from /baldig/physicstest/NeutronStarsData/SherpaResults/spectra+star2eos_transformer_refine_v3.1_00165/Settings/00011.json \
        --mass_threshold 3 --gpu 3
        
    tf2 main.py --run_type sample --paradigm spectra+star2mr \
        --model_dir /baldig/physicstest/NeutronStarsData/SherpaResults/spectra+star2mr_refine_v2_00386/Models/00023/ \
        --load_settings_from /baldig/physicstest/NeutronStarsData/SherpaResults/spectra+star2mr_refine_v2_00386/Settings/00023.json \
        --mass_threshold 3 --gpu 3
    """
    args['fold'] = 1
    args['sherpa'] = True
    args['num_folds'] = 1

    number_of_batches = 200
    num_samples_per_star = 100

    # LOAD MODEL
    model = tf.keras.models.load_model(args['model_dir'],
                                       custom_objects={'eos_m1_metric': ns.models.custom.eos_m1_metric,
                                                       'eos_m2_metric': ns.models.custom.eos_m1_metric})

    # BUILD THE DATA LOADER & PARTITION THE DATASET
    data_loader = ns.DataLoader(args, small=False)
    # args["model_type"] = "normal"

    np.random.seed(123)
    for sample_type in ["small_noise", "uncertainty", "none", "poisson", 'empirical', 'uniform']:

        # :SINGLE OUTPUT ASSUMPTION:
        output_name = args['outputs'][0]['name']
        ids = np.zeros((0, 2))
        targets = np.zeros((0, args['output_size']))
        predictions = np.zeros((0, args['output_size']))

        for batch_idx in tqdm.tqdm(range(number_of_batches)):
            # GET A BATCH FROM THE DATA LOADER
            x, y = data_loader.train_gen.__getitem__(np.random.randint(0, len(data_loader.train_gen)), transform=False)

            for sample_number in range(1 if sample_type == "none" else num_samples_per_star):
                # SAMPLE THE DATA WITH NOISE ACCORDING TO THE SAMPLE TYPE
                x_sample = data_loader.sample(x, sample_type=sample_type)
                # TRANSFORM THE DATA AFTER SAMPLING
                x_sample, _ = data_loader.train_gen.scaler.transform(x_sample, {})

                # FEED THE DATA THROUGH THE NETWORK
                y_hat_sample = model.predict(x_sample, batch_size=args['batch_size'])

                y_hat_sample = {output_name: y_hat_sample}
                _, y_hat_sample = data_loader.train_gen.scaler.inverse_transform({}, y_hat_sample)
                y_hat_sample = y_hat_sample[output_name]

                targets = np.concatenate([
                    targets,
                    y[output_name]
                ], axis=0)

                predictions = np.concatenate([
                    predictions,
                    y_hat_sample
                ], axis=0)

                # CREATE UNIQUE IDs TO TRACK WHICH BATCH AND IDX THE SAMPLE CAME FROM
                _ids = np.ones((args["batch_size"], 2))
                _ids[:, 0] = batch_idx
                _ids[:, 1] = np.arange(args["batch_size"])
                ids = np.concatenate([ids, _ids])

        ids_df = pd.DataFrame(data=ids, columns=["batch_number", "index_in_batch"])
        targets_df = pd.DataFrame(data=targets, columns=args['output_columns'])
        predictions_df = pd.DataFrame(data=predictions.squeeze(), columns=[f'pred_{c}' for c in args['output_columns']])

        df = pd.concat([ids_df, targets_df, predictions_df], axis=1)
        prediction_file = args['output_dir'] + f'Predictions/sample_{sample_type}_%05d_%02d.csv' % (args['trial_id'], args['fold'])
        df.to_csv(prediction_file)

        print('Predictions written to:', prediction_file)


elif args['run_type'] == 'test':
    """
    tf2 main.py \
        --run_type test \
        --paradigm spectra+star2eos \
        --model_dir Results/spectra+star2eos_transformer/Models/00047 
    """

    for model_dir in glob.iglob(args['model_dir']):
        if not os.path.exists(model_dir + '/saved_model.pb'):
            continue

        args['load_settings_from'] = model_dir.replace('Models', 'Settings') + '.json'
        ns.utils.load_settings(args)
        args["data_dir"] = "/baldig/physicstest/NeutronStarsData/res_nonoise10*/"

        args['fold'] = 1
        args['sherpa'] = "SherpaResults" in args["model_dir"]
        args['num_folds'] = 1
        args['output_dir'] = "/".join(args["model_dir"].split("/")[:-2]) + "/"

        print('Model:', model_dir)
        print('Settings:', args['load_settings_from'])
        print("Output dir:", args["output_dir"])

        model = tf.keras.models.load_model(model_dir, custom_objects={'eos_m1_metric': ns.models.custom.eos_m1_metric,
                                                                      'eos_m2_metric': ns.models.custom.eos_m2_metric})
        model.summary()

        # BUILD THE DATA LOADER & PARTITION THE DATASET
        data_loader = ns.DataLoader(args)

        # ns.utils.predict_scale_store(data_loader.train_gen, model, args, 'train')
        ns.utils.predict_scale_store(data_loader.validation_gen, model, args, 'testing')

        """
        Temp code for testing sampling NP on the transformer models:
            Spectra + star 2 --> EOS
        """
        # Y_HAT = None
        # for i in range(100):
        #     # LOAD THE PARTITION AND MAKE PREDICTIONS
        #     x, _ = data_loader.validation_gen.load_all(transform=True)
        #     num_samples = x['nuisance-parameters'].shape[0]
        #     idx = np.arange(num_samples)
        #
        #     x['nuisance-parameters'][:, 0] = x['nuisance-parameters'][np.random.choice(idx, num_samples, replace=False), 0]
        #     x['nuisance-parameters'][:, 1] = x['nuisance-parameters'][np.random.choice(idx, num_samples, replace=False), 1]
        #     x['nuisance-parameters'][:, 2] = x['nuisance-parameters'][np.random.choice(idx, num_samples, replace=False), 2]
        #
        #     y_hat = model.predict(x, batch_size=args['batch_size'])
        #     x, y = data_loader.validation_gen.load_all(transform=False)
        #
        #     # :SINGLE OUTPUT ASSUMPTION:
        #     output_name = args['outputs'][0]['name']
        #     y_hat = {output_name: y_hat}
        #     _, y_hat = data_loader.validation_gen.scaler.inverse_transform({}, y_hat)
        #     y_hat = y_hat[output_name]
        #     y = y[output_name]
        #
        #     if Y_HAT is None:
        #         Y_HAT = y_hat
        #     else:
        #         Y_HAT += y_hat
        #     print(i)
        #
        # # STORE INPUTS, TARGETS, PREDICTIONS
        # ns.utils.store_predictions(x, y, Y_HAT / 100.0, args, 'sampling', save_inputs=False)
