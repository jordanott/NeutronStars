"""
>>> tf2 main.py --paradigm spectra+star2eos
"""

import os
import tqdm
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

    for fold in range(1, args['num_folds']+1):
        args['fold'] = fold
        
        # BUILD THE DATA LOADER & PARTITION THE DATASET
        data_loader = ns.DataLoader(args)
        # GET tf.keras CALLBACKS
        callbacks = ns.models.create_callbacks(args)
        # BUILD MODEL ARCHITECTURE BASED ON ARGS
        model = ns.models.build_model(args)
        # PLOT MODEL
        tf.keras.utils.plot_model(model, args['model_dir'] + '/model.png')

        # COMPILE THE MODEL
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=args['lr']),
            loss=args['loss_function'],
            metrics=['mean_absolute_percentage_error', 'mse']
        )
    
        # TRAIN THE MODEL
        history = model.fit(
            x=data_loader.train_gen,
            epochs=args['epochs'],
            validation_data=data_loader.validation_gen,
            callbacks=callbacks,
            verbose=2 if args['sherpa'] else 1,
            workers=16,
            max_queue_size=10,
        ).history

        ns.utils.store_training_history(history, args)

        # LOAD THE BEST NETWORK FROM EARLY STOPPING
        model = tf.keras.models.load_model(args['model_dir'])

        ns.utils.predict_scale_store(data_loader.validation_gen, model, args, 'validation')
        if args['sherpa']:
            ns.utils.predict_scale_store(data_loader.test_gen, model, args, 'test')

elif args['run_type'] == 'uncertain':
    args['fold'] = 10
    
    # BUILD THE DATA LOADER & PARTITION THE DATASET
    data_loader = ns.DataLoader(args)
    # LOAD MODEL FROM CROSS-VALIDATION
    model = tf.keras.models.load_model(args['model_dir'])
    
    X, Y = data_loader.validation_gen.load_all(transform=False)
    
    targets = np.zeros((100*Y.shape[0], Y.shape[1]))
    predictions = np.zeros((100*Y.shape[0], Y.shape[1]))

    # ADDING POISSON NOISE TO SPECTROGRAM
    for i in tqdm.tqdm(range(X.shape[0])):
        x = X[i]
        y = Y[i]

        # POISSON NOISE ON SPECTRUM
        x_poisson = np.random.poisson(x, size=(100, len(x)))
        x_poisson[:, -3:] = x[-3:]

        # TRANSFORM THE DATA
        x_poisson_transform = data_loader.validation_gen.scaler.x_scaler.transform(x_poisson)

        # MAKE A PREDICTION
        y_hat = model.predict(x_poisson_transform)

        # INVERSE TRANSFORM THE PREDICTIONS
        y_hat_transform = data_loader.validation_gen.scaler.y_scaler.inverse_transform(y_hat)
        
        idx = i * 100
        targets[idx:idx+100] = np.tile(y, (100, 1))
        predictions[idx:idx+100] = y_hat_transform

    ns.utils.store_predictions(
        x=None,
        y=targets,
        predictions=predictions,
        args=args,
        data_partition='poisson')
