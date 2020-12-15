from argparse import ArgumentParser, ArgumentTypeError


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def update_parser(parser: ArgumentParser):
    parser.add_argument('--layers', type=int, default=1,
                        help='number of vpnn layers in each part')
    parser.add_argument('--rotations', type=int, default=2,
                        help='number of vpnn layers in each part')
    parser.add_argument('--output_dim', type=int, default=None,
                        help='number of vpnn layers in each part')
    parser.add_argument('--output_layer', choices=["dense", "svd", "vpnn"], default="dense",
                        help='number of vpnn layers in each part')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs in training')
    parser.add_argument('--use_dropout', type=str2bool, nargs='?',
                        const=True, default=False,
                        help='number of epochs in training')
    parser.add_argument('-fake', action='store_true')
    parser.add_argument('--permutation_arrangement', type=int, choices=[1, 2, 3, 4, 5, 6], default=4,
                        help='random = 1  horizontal = 2 vertical = 3 mixed = 4 grid = 5 mixed3 = 6')
    parser.add_argument('--total_runs', type=int, default=28,
                        help='it will run tests from 1 to total_runs')


class CommandLineArgs():
    def __init__(self) -> None:
        parser = ArgumentParser()
        update_parser(parser)
        args = parser.parse_args()
        self.layers = args.layers
        self.rotations = args.rotations
        self.output_dim = args.output_dim
        self.output_layer = args.output_layer
        self.epochs = args.epochs
        self.fake = args.fake
        self.dropout = args.use_dropout
        self.permutation_arrangement = args.permutation_arrangement
        self.total_runs = args.total_runs
        self.args = args


if __name__ == "__main__":
    args = CommandLineArgs()
    print(args.args)
