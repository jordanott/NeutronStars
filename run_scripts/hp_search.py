"""
# Changes to run from flight
# Test zero mean scaler
tf2 run_scripts/hp_search.py --paradigm spectra2eos --model_type transformer --max_concurrent 16 --name _0mean_v1

# Test only training m_2
tf2 run_scripts/hp_search.py --paradigm spectra2m_2 --model_type transformer --max_concurrent 16 --name _0mean_v1

# Redo Spectra Generator with conv and relu output
# Test edits in models/models.py
tf2 main.py --paradigm mr+stars2spectra --conv_branch --scaler_type none+none2none
# Then run
tf2 run_scripts/hp_search.py --paradigm mr+star2spectra --max_concurrent 16 --name _v2
"""

import os
import sherpa
import argparse
import itertools
import neutron_stars as ns

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='')
parser.add_argument('--model_type', default='normal')
parser.add_argument('--gpus', default='0,1,2,3', type=str)
parser.add_argument('--paradigm', default='spectra2eos', choices=ns.PARADIGMS)
parser.add_argument('--output_dir', default='/baldig/physicstest/NeutronStarsData/SherpaResults/')
parser.add_argument('--max_concurrent', help='Number of concurrent processes', type=int, default=16)
args = parser.parse_args()


os.makedirs(args.output_dir, exist_ok=True)

parameters = [sherpa.Choice('mass_threshold', [6]),
              sherpa.Discrete('num_layers', [1, 12]),
              sherpa.Choice('num_nodes', list(range(64, 2049, 64))),
              sherpa.Choice('batch_norm', [0, 1]),
              sherpa.Continuous('dropout', [0, 1]),
              sherpa.Choice('skip_connections', [0, 1]),
              sherpa.Continuous('lr', [0.0001, 0.005]),
              sherpa.Continuous('lr_decay', [0.8, 1.]),
              sherpa.Choice('activation', list(ns.models.AVAILABLE_ACTIVATIONS.keys())),
              sherpa.Choice('scaler_type', ns.data_loader.scaler_combinations_for_paradigm(args.paradigm)),
              sherpa.Choice('loss_function', ['mse', 'mean_absolute_percentage_error', 'huber'])]

if args.model_type == 'transformer':
    parameters.extend([sherpa.Choice('num_stars', [15]),
                       sherpa.Choice('transformer_op', ['max', 'min', 'gather'])])

elif 'spectra' in args.paradigm:
    parameters.append(
        sherpa.Choice('conv_branch', [0, 1])
    )

algorithm = sherpa.algorithms.RandomSearch(max_num_trials=1000)


gpus = [int(x) for x in args.gpus.split(',')]
processes_per_gpu = args.max_concurrent//len(gpus)
assert args.max_concurrent % len(gpus) == 0
resources = list(itertools.chain.from_iterable(itertools.repeat(x, processes_per_gpu) for x in gpus))
scheduler = sherpa.schedulers.LocalScheduler(resources=resources)


command = f"/home/jott1/tf2_env/bin/python main.py " \
          f"--sherpa --paradigm {args.paradigm} " \
          f"--output_dir {args.output_dir} " \
          f"--model_type {args.model_type} " \
          f"--name {args.name} " \
          f"--data_dir /baldig/physicstest/NeutronStarsData/res_nonoise10x/ " \
          f"--batch_size 1024"

sherpa.optimize(algorithm=algorithm,
                scheduler=scheduler,
                parameters=parameters,
                lower_is_better=True,
                command=command,
                max_concurrent=args.max_concurrent,
                output_dir=args.output_dir + args.paradigm + f"_{args.model_type}{args.name}".replace('_normal', ''))
