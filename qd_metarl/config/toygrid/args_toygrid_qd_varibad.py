from qd_metarl.utils.env_utils import bool_arg, int_arg
from qd_metarl.config.toygrid.common_args_toygrid_varibad import get_common_arg_parser
from qd_metarl.config.toygrid.common_args_toygrid_rl2 import get_common_rl2_arg_parser


def get_parser(use_rl2):
    if use_rl2:
        parser = get_common_rl2_arg_parser()
    else:
        parser = get_common_arg_parser()

    # --- GENERAL ---
    parser.add_argument('--exp-label',                  default='varibad',                  help='label (typically name of method)')
    parser.add_argument('--env-name',                   default='ToyGridQD-v0',             help='environment to train on')
    
    # --- QD ---
    # We 'use_plr' but with a random objective, so the sample weights we get 
    # from the PLR buffer are uniform
    parser.add_argument('--use-plr',                    type=bool_arg,  default=True,       help="Use PLR objective")
    parser.add_argument('--plr-env-generator', type=str, default='gen', choices=['sb', 'gen'], 
                        help='Environment generator used as input for PLR. ' 
                             '\'sb\' is for manicured seed-based distribution ' 
                             'and \'gen\' is for random generation using same '
                             'genotype method as QD.')
    parser.add_argument("--plr-buffer-size-from-qd",    type=bool_arg,  default=True,       help="Match reply buffer size to QD archive size.")
    parser.add_argument('--use-qd',                     type=bool_arg,  default=True,       help="Use QD instead of pre-defined distribution")
    # Updates and logging
    # NOTE: The only active argumment here is the log-interval, which pertains
    # the warm starts in this case. Because we're not using a simlution-based
    # objective, there are no updates after warm-start.
    parser.add_argument('--qd-update-interval',         type=int,       default=10,         help="Update interval for QD; one QD update per n meta-RL updates")
    parser.add_argument('--qd-updates-per-iter',        type=int,       default=1,          help="Number of QD updates per iteration")
    parser.add_argument('--qd-log-interval',            type=int,       default=5_000,       help="Log interval for QD; one QD log per n QD updates")
    # Warm-start
    parser.add_argument('--qd-use-two-stage-ws',        type=bool_arg,  default=False)
    parser.add_argument('--qd-warm-start-updates',      type=int_arg,   default=100_000)
    # Archive
    parser.add_argument('--qd-measures',     nargs='+', default=['x_pos', 'y_pos'])
    parser.add_argument('--qd-archive-dims', nargs='+', type=int,       default=[5, 9])
    parser.add_argument('--qd-initial-population',      type=int_arg,   default=1_000)
    # QD hyperparams
    parser.add_argument('--qd-batch-size',              type=int,       default=8)
    parser.add_argument('--qd-emitter-type',            type=str,       default='me',       choices=['es', 'me'])
    parser.add_argument('--qd-mutation-percentage',     type=float,     default=0.1)
    # Objectives
    parser.add_argument('--qd-no-sim-objective',        type=bool_arg,  default=True)
    parser.add_argument('--qd-gt-diversity-objective',  type=bool_arg,  default=False)
    parser.add_argument('--qd-bias-new-solutions',      type=bool_arg,  default=True,       help='Slightly bias newer solutions by adding a slowy incrementing constant float to the objective')
    # NOTE: No sample mask needed for this environment

    return parser
