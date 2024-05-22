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
    parser.add_argument('--env-name',                   default='ToyGridQD-v0',             help='environment to train on')

        # --- PLR ---
    parser.add_argument('--use-plr', type=bool_arg, default=True, help='Use prioritized level replay')
    parser.add_argument('--plr-level-replay-strategy', type=str, default='random', help="Level replay scoring strategy")
    parser.add_argument('--plr-env-generator', type=str, default='gen', choices=['sb', 'gen'], help='Environment generator used as input for PLR. ' 
                             '\'sb\' is for manicured seed-based distribution ' 
                             'and \'gen\' is for random generation using same '
                             'genotype method as QD.')
    
    # Placeholder---we need the QD archive to generate bounds (for now)
    parser.add_argument('--qd-measures',     nargs='+', default=['x_pos', 'y_pos'])
    parser.add_argument('--qd-archive-dims', nargs='+', type=int,       default=[5, 9])
    parser.add_argument("--plr-seed-buffer-size",type=int,default=150, help="Number of levels to store in our buffer, after which we replace")



    return parser
