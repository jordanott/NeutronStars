import os
import sherpa
import pprint
import tensorflow as tf
import neutron_stars as ns


# GET COMMAND LINE ARGS
args = ns.parse_args()
ns.utils.gpu_settings(args)

# IF CONDUCTING HP SEARCH: GET ARGS
if args['sherpa']:
    client = sherpa.Client()
    trial = client.get_trial()
    args.update(trial.parameters)
    args['trial_id'] = trial.id
    args['sherpa_info'] = (client, trial)

# SET UP THE DIRECTORY TO STORE RESULTS
ns.utils.dir_set_up(args)

# SHOW ALL ARG SETTINGS
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(args)

# DETERMINE INPUTS AND TARGETS FOR GIVEN PARADIGM
ns.paradigm_settings(args)

# BUILD THE DATA LOADER & PARTITION THE DATASET
data_loader = ns.DataLoader(args)

if args['run_type'] == 'train':
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
ns.utils.predict_scale_store(data_loader.test_gen, model, args, 'test')
