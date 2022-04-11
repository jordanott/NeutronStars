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
        args["data_dir"] = "/baldig/physicstest/NeutronStarsData/res_nonoise10x/"

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
