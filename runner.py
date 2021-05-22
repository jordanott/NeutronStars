"""
tf2 runner.py --paradigm mr2eos --model_type transformer
"""

import os
import sherpa
import argparse
import itertools
import neutron_stars as ns

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', default='normal')
parser.add_argument('--gpus', default='0,1,2,3', type=str)
parser.add_argument('--paradigm', default='spectra2eos', choices=ns.PARADIGMS)
parser.add_argument('--output_dir', default='/baldig/physicstest/NeutronStarsData/SherpaResults/')
parser.add_argument('--max_concurrent', help='Number of concurrent processes', type=int, default=16)
args = parser.parse_args()


os.makedirs(args.output_dir, exist_ok=True)

parameters = [
    sherpa.Choice('mass_threshold', [3, 6]),
    # sherpa.Choice('augmentation', [1, 0]),
    sherpa.Discrete('num_layers', [1, 12]),
    sherpa.Choice('num_nodes', list(range(32, 1025, 8))),
    sherpa.Choice('batch_norm', [0, 1]),
    sherpa.Continuous('dropout', [0, 1]),
    sherpa.Choice('skip_connections', [0, 1]),
    sherpa.Continuous('lr', [0.0001, 0.01]),
    sherpa.Continuous('lr_decay', [0.8, 1.]),
    sherpa.Choice('activation', list(ns.models.AVAILABLE_ACTIVATIONS.keys())),
    sherpa.Choice('scaler_type', ns.data_loader.scaler_combinations_for_paradigm(args.paradigm)),
    sherpa.Choice('loss_function', ['mse', 'mean_absolute_percentage_error', 'huber']),
]

if 'spectra' in args.paradigm.split('2')[0]:
    parameters.append(
        sherpa.Choice('conv_branch', [0, 1])
    )

if args.model_type == 'transformer':
    parameters.extend([
        sherpa.Discrete('num_stars', [1, 10]),
        sherpa.Choice('transformer_op', ['max', 'min', 'sum']),
    ])
algorithm = sherpa.algorithms.RandomSearch(max_num_trials=1000)


gpus = [int(x) for x in args.gpus.split(',')]
processes_per_gpu = args.max_concurrent//len(gpus)
assert args.max_concurrent % len(gpus) == 0
resources = list(itertools.chain.from_iterable(itertools.repeat(x, processes_per_gpu) for x in gpus))
scheduler = sherpa.schedulers.LocalScheduler(resources=resources)


command = f"/home/jott1/tf2_env/bin/python main.py --sherpa --paradigm {args.paradigm} " \
            f"--output_dir {args.output_dir} --model_type {args.model_type}"

sherpa.optimize(
    algorithm=algorithm,
    scheduler=scheduler,
    parameters=parameters,
    lower_is_better=True,
    command=command,
    max_concurrent=args.max_concurrent,
    output_dir=args.output_dir + args.paradigm
)
