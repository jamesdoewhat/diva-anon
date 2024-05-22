from qd_metarl.utils.env_utils import bool_arg
from qd_metarl.config.alchemy.common_args_alchemy_varibad import get_common_arg_parser
from qd_metarl.config.alchemy.common_args_alchemy_rl2 import get_common_rl2_arg_parser


def get_parser(use_rl2):
    if use_rl2:
        parser = get_common_rl2_arg_parser()
    else:
        parser = get_common_arg_parser()

    # --- GENERAL ---
    parser.add_argument('--exp-label', default='hyperx', help='label (typically name of method)')
    parser.add_argument('--env-name', default='AlchemyRandom-v0', help='environment to train on')

    # which exploration bonus(es) to use
    parser.add_argument('--exploration-bonus-hyperstate', type=bool_arg, default=True, help='bonus on (s, b)')
    parser.add_argument('--exploration-bonus-vae-error', type=bool_arg, default=True)

    # --- EXPLORATION (default values from "rooms" env ---

    # weights for the rewards bonuses
    parser.add_argument('--weight-exploration-bonus-hyperstate', type=float, default=10.0)
    parser.add_argument('--weight-exploration-bonus-state', type=float, default=10.0)
    parser.add_argument('--weight-exploration-bonus-belief', type=float, default=10.0)
    parser.add_argument('--weight-exploration-bonus-vae-error', type=float, default=1.0)
    parser.add_argument('--anneal-exploration-bonus-weights', type=bool_arg, default=True)

    # hyperparameters for the random network
    parser.add_argument('--rnd-lr', type=float, default=1e-4, help='learning rate ')
    parser.add_argument('--rnd-batch-size', type=int, default=128)
    parser.add_argument('--rnd-update-frequency', type=int, default=1)
    parser.add_argument('--rnd-buffer-size', type=int, default=10000000)
    parser.add_argument('--rnd-output-dim', type=int, default=128)
    parser.add_argument('--rnd-prior-net-layers', nargs='+', type=int, default=[256, 256])
    parser.add_argument('--rnd-predictor-net-layers', nargs='+', type=int, default=[256, 256])
    parser.add_argument('--rnd-norm-inputs', type=bool_arg, default=False, help='normalise inputs by dividing by var and clipping values')
    parser.add_argument('--rnd-init-weight-scale', type=float, default=10.0, help='by how much to scale the random network weights')

    # other settings
    parser.add_argument('--intrinsic-rew-clip-rewards', type=float, default=10.)
    parser.add_argument('--state-expl-idx', nargs='+', type=int, default=None, help='which part of the state space to do exploration on, None does all')

    return parser