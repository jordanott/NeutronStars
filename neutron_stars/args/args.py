import argparse
import neutron_stars as ns


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sherpa', action='store_true')

    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--patience', default=25, type=int)
    parser.add_argument('--lr', type=float, default=.0001)
    parser.add_argument('--lr_decay', type=float, default=1.)
    parser.add_argument('--loss_function', default='mse')

    parser.add_argument('--run_type', choices=['train', 'test'], default='train')

    parser.add_argument('--model_dir', default='')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--num_coefficients', default=2, type=int)
    parser.add_argument('--paradigm', default='spectra2eos', choices=ns.PARADIGMS)

    return vars(parser.parse_args())