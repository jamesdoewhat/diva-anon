from qd_metarl.utils.env_utils import bool_arg
from qd_metarl.config.toygrid.common_args_toygrid_varibad import get_common_arg_parser
from qd_metarl.config.toygrid.common_args_toygrid_rl2 import get_common_rl2_arg_parser


def get_parser(use_rl2):
    if use_rl2:
        parser = get_common_rl2_arg_parser()
    else:
        parser = get_common_arg_parser()

    # --- GENERAL ---
    parser.add_argument('--exp-label', default='varibad', help='label (typically name of method)')
    parser.add_argument('--env-name', default='ToyGrid-v0', help='environment to train on')

    return parser
