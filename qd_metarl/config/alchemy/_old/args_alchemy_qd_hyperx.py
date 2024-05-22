from qd_metarl.utils.env_utils import bool_arg, int_arg
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

    # --- QD ---
    parser.add_argument('--use-plr', type=bool_arg, default=True, help="Use PLR objective")
    parser.add_argument('--plr-env-generator', type=str, default='gen', choices=['sb', 'gen'], help='Environment generator used as input for PLR. ' 
                             '\'sb\' is for manicured seed-based distribution ' 
                             'and \'gen\' is for random generation using same '
                             'genotype method as QD.')
    parser.add_argument('--use-qd', type=bool_arg, default=True, help="Use QD instead of pre-defined distribution")
    parser.add_argument('--qd-update-interval', type=int, default=10, help="Update interval for QD; one QD update per n meta-RL updates")
    parser.add_argument('--qd-updates-per-iter', type=int, default=1, help="Number of QD updates per iteration")
    parser.add_argument('--qd-log-interval', type=int, default=4, help="Log interval for QD; one QD log per n QD updates")
    parser.add_argument('--qd-warm-start-updates', type=int_arg, default=10_000)
    # parser.add_argument('--qd-measures', nargs='+', default=['num_filled', 'max_dist'])  # TODO: change to alchemy measures
    parser.add_argument('--qd-archive-dims', nargs='+', type=int, default=[50, 50])
    parser.add_argument('--qd-batch-size', type=int, default=4)
    parser.add_argument('--qd-initial-population', type=int_arg, default=10)
    parser.add_argument('--qd-mutation-percentage', type=float, default=0.1)
    parser.add_argument('--qd-emitter-type', type=str, default='me', choices=['es', 'me'])
    parser.add_argument('--qd-measure-selector-use-all-measures', type=bool_arg, default=True)

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