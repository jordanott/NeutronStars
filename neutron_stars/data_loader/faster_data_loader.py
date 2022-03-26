import glob
import time

import numpy as np
import multiprocessing
from concurrent.futures import ThreadPoolExecutor


def faster_data_loader(args):
    start_time = time.time()

    all_files = [
        f for f in list(glob.iglob(args['data_dir'] + '*.npz'))
        if f"{args['num_coefficients']}Param" in f
    ]

    spectra_file_data = [0] * len(all_files)
    details_file_data = [0] * len(all_files)
    eos_coefficients_file_data = [0] * len(all_files)

    def load_file(file_number, file):
        np_file = np.load(file)
        mass_threshold = (np_file['details'][:, 0] < args['mass_threshold'])

        spectra_file_data[file_number] = np_file["spectra"][mass_threshold] #* 1e5
        details_file_data[file_number] = np_file["details"][mass_threshold]
        eos_coefficients_file_data[file_number] = np_file["coefficients"][mass_threshold]

    threads = []
    with ThreadPoolExecutor(40) as executor:
        for file_number, file in enumerate(all_files):
            thread = executor.submit(load_file, file_number, file)
            threads.append(thread)

        for thread in threads:
            thread.result()

    number_of_samples = 0
    for file_data in eos_coefficients_file_data:
        number_of_samples += file_data.shape[0]

    data = {
        "spectra": spectra_file_data,
        "details": details_file_data,
        "coefficients": eos_coefficients_file_data
    }
    # INIT PLACEHOLDERS FOR DATA DICTIONARY
    X = {
        opts['name']: np.zeros((number_of_samples, len(opts['idxs'])))
        for opts in args['inputs']
    }
    Y = {
        opts['name']: np.zeros((number_of_samples, len(opts['idxs'])))
        for opts in args['outputs']
    }

    index = 0
    eos_coefficients = np.zeros((number_of_samples, 2))
    for file_index in range(len(all_files)):
        file_data = data["coefficients"][file_index]
        samples_in_file = file_data.shape[0]

        eos_coefficients[index:index + samples_in_file] = file_data
        index += samples_in_file

    # INPUTS
    for opts in args["inputs"]:

        index = 0

        for file_index in range(len(all_files)):
            key = opts['key']
            name = opts["name"]

            file_data = data[key][file_index]
            samples_in_file = file_data.shape[0]

            X[name][index:index+samples_in_file] = file_data[:, opts['idxs']]

            index += samples_in_file

    for opts in args["outputs"]:

        index = 0

        for file_index in range(len(all_files)):
            key = opts['key']
            name = opts["name"]

            file_data = data[key][file_index]
            samples_in_file = file_data.shape[0]

            Y[name][index:index + samples_in_file] = file_data[:, opts['idxs']]

            index += samples_in_file

    print("Loaded data in:", time.time() - start_time, "seconds")

    for key, value in X.items():
        print(key, value.shape)

    for key, value in Y.items():
        print(key, value.shape)

    return X, Y, eos_coefficients


if __name__ == "__main__":
    import neutron_stars as ns
    args = ns.parse_args()
    ns.paradigm_settings(args)

    X, Y, eos_coefficients = fast_load_data(args)

    print(eos_coefficients.shape)
