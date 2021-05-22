import os
import pprint
import tensorflow as tf
import neutron_stars as ns


# Get command line args
args = ns.parse_args()

# Display options selected
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(args)

ns.paradigm_settings(args)

# Create the data loader
data_loader = ns.DataLoader(args)

# Load the saved model
model = tf.keras.models.load_model('SavedModels/mr+star2spectra/00011/')

# Make predictions and save them
args['output_dir'] = './'
os.makedirs('Predictions', exist_ok=True)

ns.utils.predict_scale_store(generator=data_loader.all_gen,
                             model=model, args=args,
                             data_partition='all',
                             save_inputs=True)

