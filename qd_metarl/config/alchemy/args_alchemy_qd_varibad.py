from qd_metarl.utils.env_utils import bool_arg, int_arg
from qd_metarl.config.alchemy.common_args_alchemy_varibad import get_common_arg_parser
from qd_metarl.config.alchemy.common_args_alchemy_rl2 import get_common_rl2_arg_parser


def get_parser(use_rl2):
    if use_rl2:
        parser = get_common_rl2_arg_parser()
    else:
        parser = get_common_arg_parser()

    # --- GENERAL ---
    parser.add_argument('--exp-label', default='varibad', help='label (typically name of method)')
    parser.add_argument('--env-name', default='AlchemyRandomQD-v0', help='environment to train on')

    # --- QD ---
    # NOTE: We 'use_plr' but with a random objective, so the sample weights we get 
    # from the PLR buffer are uniform
    parser.add_argument('--use-plr',                    type=bool_arg,  default=True, help="Use PLR objective")
    parser.add_argument('--plr-env-generator', type=str, default='gen', choices=['sb', 'gen'], help='Environment generator used as input for PLR. ' 
                             '\'sb\' is for manicured seed-based distribution ' 
                             'and \'gen\' is for random generation using same '
                             'genotype method as QD.')
    parser.add_argument("--plr-buffer-size-from-qd",    type=bool_arg,  default=True,      help="Match reply buffer size to QD archive size.")
    parser.add_argument('--use-qd',                     type=bool_arg,  default=True, help="Use QD instead of pre-defined distribution")
    # Updates and logging
    # NOTE: The only active argumment here is the log-interval, which pertains
    # the warm starts in this case. Because we're not using a simlution-based
    # objective, there are no updates after warm-start.
    parser.add_argument('--qd-update-interval',         type=int,       default=10, help="Update interval for QD; one QD update per n meta-RL updates")
    parser.add_argument('--qd-updates-per-iter',        type=int,       default=1, help="Number of QD updates per iteration")
    parser.add_argument('--qd-log-interval',            type=int,       default=5000, help="Log interval for QD; one QD log per n QD updates")
    # Warm-start 
    parser.add_argument('--qd-use-two-stage-ws',        type=bool_arg,  default=True)
    parser.add_argument('--qd-init-warm-start-updates', type=int_arg,   default=50_000)
    parser.add_argument('--qd-warm-start-updates',      type=int_arg,   default=30_000)
    # Archive
    parser.add_argument('--qd-measures', nargs='+', 
                        default=['stone_reflection', 'stone_rotation', 
                                 'parity_first_stone', 'parity_first_potion', 
                                 'latent_state_diversity', 'average_manhattan_to_optimal'])
    parser.add_argument('--qd-init-archive-dims', nargs='+', type=int,  default=[1, 1, 1, 1, 300, 300])
    parser.add_argument('--qd-archive-dims', nargs='+', type=int,       default=[1, 1, 5, 1, 150, 150])
    parser.add_argument('--qd-initial-population',      type=int_arg,   default=1_000)
    # QD hyperparams
    parser.add_argument('--qd-batch-size',              type=int,       default=5)
    parser.add_argument('--qd-emitter-type',            type=str,       default='me', choices=['es', 'me'])
    parser.add_argument('--qd-mutation-percentage',     type=float,     default=0.02)
    # Objectives
    parser.add_argument('--qd-no-sim-objective',        type=bool_arg,  default=True)
    parser.add_argument('--qd-gt-diversity-objective',  type=bool_arg,  default=False)
    # Sample mask
    parser.add_argument('--qd-update-sample-mask',      type=bool_arg,  default=True)
    parser.add_argument('--qd-sample-mask-min-solutions', type=int_arg, default=40)
    parser.add_argument('--qd-bias-new-solutions',      type=bool_arg,  default=True,  help='Slightly bias newer solutions by adding a slowy incrementing constant float to the objective')

    return parser
