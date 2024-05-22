from qd_metarl.utils.env_utils import bool_arg, int_arg
from qd_metarl.config.racing.common_args_racing_varibad import get_common_arg_parser
from qd_metarl.config.racing.common_args_racing_rl2 import get_common_rl2_arg_parser


def get_parser(use_rl2):
    if use_rl2:
        parser = get_common_rl2_arg_parser()
    else:
        parser = get_common_arg_parser()

    # --- GENERAL ---
    parser.add_argument('--exp-label', default='varibad', help='label (typically name of method)')
    parser.add_argument('--env-name', default='CarRacing-BezierQD-v0', help='environment to train on')
    # In Oracle-populated archive QD case, we use a different train/archive environment
    parser.add_argument('--archive-env-name', default='CarRacing-Bezier-v0', help='environment to use for archive')

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
    # parser.add_argument('--qd-measures', nargs='+', default=['num_filled', 'max_dist'])  # TODO: change to alchemy measures
    parser.add_argument('--qd-archive-dims', nargs='+', type=int, default=[50, 50])
    parser.add_argument('--qd-batch-size', type=int, default=5)
    parser.add_argument('--qd-initial-population', type=int_arg, default=400)
    parser.add_argument('--qd-emitter-type', type=str, default='es', choices=['es', 'me'])
    # parser.add_argument('--qd-measure-selector-use-all-measures', type=bool_arg, default=True)
    parser.add_argument('--qd-no-sim-objective', type=bool_arg, default=True)
    parser.add_argument('--qd-gt-diversity-objective', type=bool_arg, default=False)

    return parser
