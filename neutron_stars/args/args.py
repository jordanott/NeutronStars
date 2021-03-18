import sherpa
import argparse
import neutron_stars as ns


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--sherpa', action='store_true')
    parser.add_argument('--load_settings_from', default='')

    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--patience', default=25, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    
    # MODEL ARCHITECTURE OPTIONS
    parser.add_argument('--num_layers', default=10, type=int)
    parser.add_argument('--num_nodes', default=256, type=int)
    parser.add_argument('--batch_norm', action='store_true')
    parser.add_argument('--dropout', type=float, default=.25)
    parser.add_argument('--skip_connections', action='store_true')
    parser.add_argument('--lr', type=float, default=.0001)
    parser.add_argument('--lr_decay', type=float, default=1.)
    parser.add_argument('--loss_function', default='mse')
    parser.add_argument('--activation', default='relu', choices=ns.models.AVAILABLE_ACTIVATIONS)

    parser.add_argument('--run_type', choices=['train', 'test', 'uncertain'], default='train')

    parser.add_argument('--model_dir', default='')
    parser.add_argument('--output_dir', default='Results/')
    parser.add_argument('--num_coefficients', default=2, type=int)
    parser.add_argument('--paradigm', default='spectra2eos', choices=ns.PARADIGMS)
    parser.add_argument('--scaler_type', default='standard2standard', choices=ns.data_loader.SCALER_COMBINATIONS)

    args = vars(parser.parse_args())

    ns.utils.gpu_settings(args)

    if args['load_settings_from']:
        ns.utils.load_settings(args)

    # IF CONDUCTING HP SEARCH: GET ARGS
    if args['sherpa']:
        client = sherpa.Client()
        trial = client.get_trial()
        args.update(trial.parameters)
        args['trial_id'] = trial.id
        args['sherpa_info'] = (client, trial)

    return args
