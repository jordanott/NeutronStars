import numpy as np

DATA_DIR = '/baldig/physicstest/NeutronStarsData/res/'
PARADIGMS = ['spectra2eos', 'spectra2mr', 'spectra2star', 'star2eos']


def paradigm_settings(args):
    num_coefficients = args['num_coefficients']

    if args['paradigm'] == 'spectra2eos':
        args['input_key'] = 'spectra'
        args['output_key'] = 'coefficients'

        args['input_idxs'] = np.arange(1024)
        args['output_idxs'] = np.arange(num_coefficients)

    elif args['paradigm'] == 'spectra2mr':
        args['input_key'] = 'spectra'
        args['output_key'] = 'details'

        args['input_idxs'] = np.arange(1024)
        args['output_idxs'] = np.arange(2)

    elif args['paradigm'] == 'spectra2star':
        args['input_key'] = 'spectra'
        args['output_key'] = 'details'

        args['input_idxs'] = np.arange(1024)
        args['output_idxs'] = np.arange(5)

    elif args['paradigm'] == 'star2eos':
        args['input_key'] = 'details'
        args['output_key'] = 'coefficients'

        args['input_idxs'] = np.arange(5)
        args['output_idxs'] = np.arange(num_coefficients)

    args['input_size'] = len(args['input_idxs'])
    args['output_size'] = len(args['output_idxs'])