import numpy as np

DATA_DIR = '/baldig/physicstest/NeutronStarsData/res_nonoise10x/'
PARADIGMS = ['spectra+star2eos', 'spectra+star2mr', 'spectra2eos', 'spectra2mr', 'mr2eos', 'mr+star2spectra']


def get_paradigm_opts(num_coefficients=2):
    return {
        'spectra': {
            'key': 'spectra',
            'name': 'spectra',
            'idxs': np.arange(250),
            'columns': np.arange(250).tolist(),
        },
        'mr': {
            'key': 'details',
            'name': 'mass-radius',
            'idxs': np.arange(2),
            'columns': ['Mass', 'Radius'],
        },
        'star': {
            'key': 'details',
            'name': 'nuisance-parameters',
            'idxs': np.array([2, 3, 4]),
            'columns': ['nH', 'logTeff', 'dist'],
        },
        'eos': {
            'key': 'coefficients',
            'name': 'coefficients',
            'idxs': np.arange(num_coefficients),
            'columns': [f'c{c}' for c in range(1, num_coefficients+1)],
        }
    }


def paradigm_settings(args):
    args['fold'] = 1
    args['num_folds'] = 1 if args['sherpa'] else 10

    num_coefficients = args['num_coefficients']
    opts = get_paradigm_opts(num_coefficients)
    
    def get_options(k):
        return opts[k]

    args['inputs'] = []
    args['input_size'] = 0
    args['input_columns'] = []
    args['input_names'] = []

    args['outputs'] = []
    args['output_size'] = 0
    args['output_columns'] = []
    args['output_names'] = []
    inputs, outputs = args['paradigm'].split('2')

    for x in inputs.split('+'):
        input_opts = get_options(x)
        args['inputs'].append(input_opts)
        args['input_size'] += len(input_opts['idxs'])
        args['input_columns'].extend(input_opts['columns'])
        args['input_names'].append(input_opts['name'])

    for x in outputs.split('+'):
        output_opts = get_options(x)
        args['outputs'].append(output_opts)
        args['output_size'] += len(output_opts['idxs'])
        args['output_columns'].extend(output_opts['columns'])
        args['output_names'].append(output_opts['name'])
