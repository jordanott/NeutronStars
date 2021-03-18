"""
>>> tf2 runner.py --paradigm spectra2eos
"""

import os
import sherpa
import argparse
import itertools
import neutron_stars as ns

parser = argparse.ArgumentParser()
parser.add_argument('--gpus', default='0,1,2,3', type=str)
parser.add_argument('--paradigm', default='spectra+star2eos', choices=ns.PARADIGMS)
parser.add_argument('--max_concurrent',help='Number of concurrent processes', type=int, default=24)
parser.add_argument('--output_dir', default='/baldig/physicstest/NeutronStarsData/SherpaResults/')
args = parser.parse_args()


os.makedirs(args.output_dir, exist_ok=True)

parameters = [
    sherpa.Discrete('num_layers', [3, 25]),
    sherpa.Discrete('num_nodes', [128, 1024]),
    sherpa.Choice('batch_norm', [0,1]),
    sherpa.Continuous('dropout', [0, 1]),
    sherpa.Choice('skip_connections', [0,1]),
    sherpa.Continuous('lr', [0.00001, 0.01], 'log'),
    sherpa.Continuous('lr_decay', [0.8, 1.]),
    sherpa.Choice('activation', list(ns.models.AVAILABLE_ACTIVATIONS.keys())),
    sherpa.Choice('scaler_type', ns.data_loader.SCALER_COMBINATIONS),
    sherpa.Choice('loss_function', ['mse', 'mean_absolute_percentage_error']),
    sherpa.Choice('output_dir', [args.output_dir]),
]


algorithm = sherpa.algorithms.RandomSearch(max_num_trials=500)


gpus = [int(x) for x in args.gpus.split(',')]
processes_per_gpu = args.max_concurrent//len(gpus)
assert args.max_concurrent % len(gpus) == 0
resources = list(itertools.chain.from_iterable(itertools.repeat(x, processes_per_gpu) for x in gpus))
scheduler = sherpa.schedulers.LocalScheduler(resources=resources)


command = f"/home/jott1/tf2_env/bin/python main.py --sherpa --paradigm {args.paradigm}"

sherpa.optimize(
    algorithm=algorithm,
    scheduler=scheduler,
    parameters=parameters,
    lower_is_better=True,
    command=command,
    max_concurrent=args.max_concurrent,
    output_dir=args.output_dir + args.paradigm
)
