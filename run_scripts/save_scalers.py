"""
tf2 run_scripts/save_scalers.py \
    --paradigm mr+star2spectra \
    --scaler_type standard+zero_mean2zero_mean \
    --output_dir SavedModels/mr+star2spectra/00069
"""
import joblib
import numpy as np
import neutron_stars as ns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


args = ns.parse_args()

ns.paradigm_settings(args)

data_loader = ns.DataLoader(args)


def save(name, scaler):
    scaler_file = f"{args['output_dir']}/{name}"
    if type(scaler) is ns.data_loader.ZeroMean:
        np.save(scaler_file + '.npy', scaler.mean)

    elif type(scaler) is StandardScaler:
        joblib.dump(scaler, scaler_file)


for name, scaler in data_loader.train_gen.scaler.input_scalers.items():
    save(name, scaler)

for name, scaler in data_loader.train_gen.scaler.output_scalers.items():
    save(name, scaler)
