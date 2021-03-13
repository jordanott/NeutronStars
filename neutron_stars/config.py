import numpy as np

DATA_DIR = '/baldig/physicstest/NeutronStarsData/res/'
PARADIGMS = ['spectra+star2eos', 'spectra2eos', 'spectra2mr', 'spectra2star', 'star2eos']


def paradigm_settings(args):
    args['num_folds'] = 1 if args['sherpa'] else 10

    num_coefficients = args['num_coefficients']

    opts = {
        'spectra': {
            'name': 'spectra',
            'idxs': np.arange(250),
            'columns': np.arange(250).tolist(),
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

    def get_options(k):
        return {'key': opts[k]['name'],
                'idxs': opts[k]['idxs'],
                'columns': opts[k]['columns']}

    args['inputs'] = []
    args['input_size'] = 0
    args['input_columns'] = []
    args['outputs'] = []
    args['output_size'] = 0
    args['output_columns'] = []
    inputs, outputs = args['paradigm'].split('2')

    for x in inputs.split('+'):
        input_opts = get_options(x)
        args['inputs'].append(input_opts)
        args['input_size'] += len(input_opts['idxs'])
        args['input_columns'].extend(input_opts['columns'])

    for x in outputs.split('+'):
        output_opts = get_options(x)
        args['outputs'].append(output_opts)
        args['output_size'] += len(output_opts['idxs'])
        args['output_columns'].extend(output_opts['columns'])
