"""
tf2 run_scripts/sample.py --run_type sample --paradigm spectra+star+mr2eos \
    --model_dir /baldig/physicstest/NeutronStarsData/SherpaResults/spectra+star2eos_transformer_refine_v3.1_00165/Models/00011/ \
    --load_settings_from /baldig/physicstest/NeutronStarsData/SherpaResults/spectra+star2eos_transformer_refine_v3.1_00165/Settings/00011.json \
    --mass_threshold 3 --gpu 3

tf2 run_scripts/sample.py --run_type sample --paradigm spectra+star+eos2mr \
    --model_dir /baldig/physicstest/NeutronStarsData/SherpaResults/spectra+star2mr_refine_v2_00386/Models/00023/ \
    --load_settings_from /baldig/physicstest/NeutronStarsData/SherpaResults/spectra+star2mr_refine_v2_00386/Settings/00023.json \
    --mass_threshold 3 --gpu 3

tf2 run_scripts/sample.py --run_type sample --paradigm spectra+star+eos2mr \
    --model_dir /baldig/physicstest/NeutronStarsData/SherpaResults/spectra+star2mr_refine_v3_00386/Models/00045/ \
    --load_settings_from /baldig/physicstest/NeutronStarsData/SherpaResults/spectra+star2mr_refine_v3_00386/Settings/00045.json \
    --mass_threshold 3 --gpu 3
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

# args["paradigm"] = "spectra+star+mr2eos"
# args["scaler_type"] = "standard+standard+none2zero_mean"
args["paradigm"] = "spectra+star+eos2mr"
args["scaler_type"] = "log+zero_mean+none2none"

# DETERMINE INPUTS AND TARGETS FOR GIVEN PARADIGM
ns.paradigm_settings(args)

args["data_dir"] = "/baldig/physicstest/NeutronStarsData/res_nonoise10*/"

args['fold'] = 1
args['sherpa'] = True
args['num_folds'] = 1

number_of_batches = 50 # 200
number_of_samples_per_star = 25 # 7
SAMPLE_FORMATS = np.array([
    (0, 0, 0),
    (0, 0, 1),
    (0, 0, -1),
    (0, 1, 0),
    (0, -1, 0),
    (1, 0, 0),
    (-1, 0, 0)
])

# LOAD MODEL
model = tf.keras.models.load_model(args['model_dir'],
                                   custom_objects={'eos_m1_metric': ns.models.custom.eos_m1_metric,
                                                   'eos_m2_metric': ns.models.custom.eos_m1_metric})

# BUILD THE DATA LOADER & PARTITION THE DATASET
data_loader = ns.DataLoader(args, small=False)
print(np.unique(data_loader.validation_gen.Y["mass-radius"], axis=0).shape)
# args["model_type"] = "normal"

np.random.seed(123)

for logTeff in [0.01, 0.025, 0.05, 0.075, 0.1]:
    for scale in [0.1, 0.25, 0.5, 1]:

        sample_formats = SAMPLE_FORMATS * scale

        # for sample_type in ["small_noise", "uncertainty", "none", "poisson", 'empirical']:
        for sample_type in ["tight", "loose"]: #0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.5]:
            # :SINGLE OUTPUT ASSUMPTION:
            output_name = args['outputs'][0]['name']
            storage_size = number_of_batches * (1 if sample_type in ["none", 0] else number_of_samples_per_star) * args["batch_size"]

            ids = np.zeros((storage_size, 2))
            targets = np.zeros((storage_size, args['output_size']))
            predictions = np.zeros((storage_size, args['output_size']))
            mass_radius = np.zeros((storage_size, 30))
            eos_coefficients = np.zeros((storage_size, 2))
            nuisance_parameters = np.zeros((storage_size, 6))

            index = 0
            for batch_idx in tqdm.tqdm(range(number_of_batches)):
                # GET A BATCH FROM THE DATA LOADER
                x, y = data_loader.validation_gen.__getitem__(batch_idx * args["batch_size"], transform=False)

                for sample_number in range(1 if sample_type in ["none", 0] else number_of_samples_per_star):

                    # SAMPLE THE DATA WITH NOISE ACCORDING TO THE SAMPLE TYPE
                    sample_format = np.random.uniform(-1, 1, 3) * scale
                    x_sample = data_loader.sample(x, sample_type=sample_type, sample_format=sample_format, logTeff=logTeff) # sample_formats[sample_number]

                    # TRANSFORM THE DATA AFTER SAMPLING
                    transformed_x_sample, _ = data_loader.validation_gen.scaler.transform(x_sample, {})

                    # FEED THE DATA THROUGH THE NETWORK
                    y_hat_sample = model.predict(transformed_x_sample, batch_size=args['batch_size'])

                    y_hat_sample = {output_name: y_hat_sample}
                    _, y_hat_sample = data_loader.validation_gen.scaler.inverse_transform({}, y_hat_sample)
                    y_hat_sample = y_hat_sample[output_name]

                    batch_size = len(y_hat_sample)
                    targets[index:index+batch_size] = y[output_name]
                    predictions[index:index+batch_size] = y_hat_sample

                    if args["model_type"] != "transformer":
                        eos_coefficients[index:index + batch_size] = x["coefficients"]
                        nuisance_parameters[index:index+batch_size] = np.hstack([
                            x_sample["nuisance-parameters"],
                            x["nuisance-parameters"]
                        ])
                    else:
                        mass_radius[index:index+batch_size] = x_sample["mass-radius"].reshape(batch_size, 30)

                    # CREATE UNIQUE IDs TO TRACK WHICH BATCH AND IDX THE SAMPLE CAME FROM
                    _ids = np.ones((batch_size, 2))
                    _ids[:, 0] = batch_idx
                    _ids[:, 1] = np.arange(batch_size)
                    ids[index:index+batch_size] = _ids

                    index += batch_size

            ids_df = pd.DataFrame(data=ids[:index], columns=["batch_number", "index_in_batch"])
            targets_df = pd.DataFrame(data=targets[:index], columns=args['output_columns'])
            predictions_df = pd.DataFrame(data=predictions[:index].squeeze(), columns=[f'pred_{c}' for c in args['output_columns']])
            mass_radius_df = pd.DataFrame(data=mass_radius[:index].squeeze(), columns=[f"M_{int(i/2)}" if i % 2 else f"R_{int(i/2-1)}" for i in range(1, 31)])

            eos_coefficients_df = pd.DataFrame(
                data=eos_coefficients[:index],
                columns=["m_1", "m_2"]
            )
            nuisance_parameters_df = pd.DataFrame(
                data=nuisance_parameters[:index],
                columns=['sample_nH', 'sample_logTeff', 'sample_dist', 'nH', 'logTeff', 'dist']
            )

            if args["model_type"] != "transformer":
                df = pd.concat([ids_df, targets_df, predictions_df, nuisance_parameters_df, eos_coefficients_df], axis=1)
            else:
                df = pd.concat([ids_df, targets_df, predictions_df, mass_radius_df], axis=1)

            folder = os.path.join(args['output_dir'], f'Predictions/logTeff_value_{logTeff}')
            os.makedirs(folder, exist_ok=True)
            prediction_file = os.path.join(
                folder,
                f'sample_{sample_type}_%05d_%02d_scale_{scale}.csv' % (args['trial_id'], args['fold'])
            )

            # prediction_file = args['output_dir'] + f'Predictions/sample_{sample_type}_%05d_%02d_sigma_{scale}.csv' % (args['trial_id'], args['fold'])
            df.to_csv(prediction_file)

            print('Predictions written to:', prediction_file)
