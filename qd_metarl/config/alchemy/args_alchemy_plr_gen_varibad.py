from qd_metarl.utils.env_utils import bool_arg
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

    # --- PLR ---
    parser.add_argument('--use-plr', type=bool_arg, default=True, help='Use prioritized level replay')
    parser.add_argument('--plr-env-generator', type=str, default='gen', choices=['sb', 'gen'], help='Environment generator used as input for PLR. ' 
                             '\'sb\' is for manicured seed-based distribution ' 
                             'and \'gen\' is for random generation using same '
                             'genotype method as QD.')
    # Placeholder---we need the QD archive to generate bounds (for now)
    parser.add_argument('--qd-measures', nargs='+', default=['num_stones'])
    parser.add_argument('--qd-archive-dims', nargs='+', type=int, default=[1])
    
    return parser
