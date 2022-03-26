import sherpa
import argparse
import neutron_stars as ns


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='')
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--sherpa', action='store_true')

    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--patience', default=25, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--augmentation', action='store_true')

    # MODEL ARCHITECTURE OPTIONS
    parser.add_argument('--conv_branch', action='store_true')
    parser.add_argument('--num_layers', default=10, type=int)
    parser.add_argument('--num_nodes', default=256, type=int)
    parser.add_argument('--batch_norm', action='store_true')
    parser.add_argument('--dropout', type=float, default=.25)
    parser.add_argument('--skip_connections', action='store_true')
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--lr_decay', type=float, default=1.)
    parser.add_argument('--loss_function', default='mse')
    parser.add_argument('--activation', default='relu', choices=ns.models.AVAILABLE_ACTIVATIONS)

    # Directory and loading args
    parser.add_argument('--model_dir', default='')
    parser.add_argument('--model_type', default='normal')
    parser.add_argument('--output_dir', default='Results/')
    parser.add_argument('--load_settings_from', default='')

    # Transformer args
    parser.add_argument('--num_stars', default=4, type=int, help='Many stars -> one universe')
    parser.add_argument('--transformer_op', default='max', help='Op at the end of transformer')

    # Data args
    parser.add_argument('--mass_threshold', default=3, type=float)
    parser.add_argument('--num_coefficients', default=2, type=int)
    parser.add_argument('--paradigm', default='spectra2eos', choices=ns.PARADIGMS)
    parser.add_argument('--data_dir', default='/baldig/physicstest/NeutronStarsData/res_nonoise10*/')
    parser.add_argument('--scaler_type', default='standard2standard', help='Same format as paradigm: log+none2standard')
    parser.add_argument('--run_type', choices=['train', 'test', 'sample'], default='train')

    args = vars(parser.parse_args())

    ns.utils.gpu_settings(args)

    if args['load_settings_from']:
        ns.utils.load_settings(args)

    # IF CONDUCTING HP SEARCH: GET ARGS
    if args['sherpa']:
        client = sherpa.Client(host="127.0.0.1")
        trial = client.get_trial()
        args.update(trial.parameters)
        args['trial_id'] = trial.id
        args['sherpa_info'] = (client, trial)
    
    return args
