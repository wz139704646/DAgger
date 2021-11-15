import argparse

# import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--env-name',
        type=str,
        default='MontezumaRevengeNoFrameskip-v0')
    parser.add_argument(
        '--num-stacks',
        type=int,
        default=8)
    parser.add_argument(
        '--num-steps',
        type=int,
        default=160)
    parser.add_argument(
        '--test-steps',
        type=int,
        default=1000)
    parser.add_argument(
        '--num-frames',
        type=int,
        default=3200)

    # general setting
    parser.add_argument(
        '--seed',
        type=int,
        default=1
    )

    ## other parameter
    parser.add_argument(
        '--log-interval',
        type=int,
        default=2,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-img',
        type=bool,
        default=False)
    parser.add_argument(
        '--save-interval',
        type=int,
        default=2,
        help='save interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--play-game',
        type=bool,
        default=False)

    # test setting
    parser.add_argument(
        '--test',
        action='store_true',
        help='only test the model'
    )
    parser.add_argument(
        '--test-times',
        type=int,
        default=10,
        help='the total times for testing'
    )
    parser.add_argument(
        '--test-render',
        action='store_true',
        help='render the game when testing'
    )

    # parameters for CNN actor (or other neural network models)
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='learning rate of the Adam optimizer'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='device for CNN actor tensor computing'
    )
    parser.add_argument(
        '--hidden-sizes',
        type=int,
        nargs='*',
        default=[512],
        help='sizes of the hidden layers in the CNN actor'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='the size of each batch samples for training CNN actor'
    )
    # parser.add_argument(
    #     '--update-steps',
    #     type=int,
    #     nargs='*',
    #     default=[50, 10, 200],
    #     help='the number of optimization steps for each update of CNN actor (start, delta, end)'
    # )
    parser.add_argument(
        '--update-epochs',
        type=int,
        nargs='*',
        default=[25, -1, 15],
        help='the number of epochs for each update of CNN actor (start, delta, end)'
    )

    # file related parameters
    parser.add_argument(
        '--checkpoints-base',
        type=str,
        default='./checkpoints',
        help='the base of the checkpoint files'
    )
    parser.add_argument(
        '--load-base',
        type=str,
        default='./checkpoints',
        help='the base of the checkpoint files for reloading'
    )
    parser.add_argument(
        '--log-base',
        type=str,
        default='./run',
        help='the base of the log files'
    )
    parser.add_argument(
        '--reload',
        action='store_true',
        help='whether to reload data and model from load-base'
    )

    args = parser.parse_args()
    return args