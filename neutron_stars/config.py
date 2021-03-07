import numpy as np

DATA_DIR = '/baldig/physicstest/NeutronStarsData/res/'
PARADIGMS = ['spectra2eos', 'spectra2mr', 'spectra2star', 'star2eos']


def paradigm_settings(args):
    num_coefficients = args['num_coefficients']

    opts = {
        'spectra': {
            'name': 'spectra',
            'idxs': np.arange(1024),
            'columns': np.arange(1024),
        },
        'mr': {
            'name': 'details',
            'idxs': np.arange(2),
            'columns': ['Mass', 'Radius'],
        },
        'star': {
            'name': 'details',
            'idxs': np.arange(5),
            'columns': ['Mass', 'Radius', 'nH', 'logTeff', 'dist'],
        },
        'eos': {
            'name': 'coefficients',
            'idxs': np.arange(num_coefficients),
            'columns': [f'c{c}' for c in range(1, num_coefficients+1)],
        }
    }

    x, y = args['paradigm'].split('2')
    
    args['input_key'] = opts[x]['name']
    args['input_idxs'] = opts[x]['idxs']
    args['input_columns'] = opts[x]['columns']
    args['input_size'] = len(args['input_idxs'])

    args['output_key'] = opts[y]['name']
    args['output_idxs'] = opts[y]['idxs']
    args['output_columns'] = opts[y]['columns']
    args['output_size'] = len(args['output_idxs'])