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
    # All the new levels come from the QD archive, so we don't need PLR to produce them for us
    parser.add_argument("--plr-replay-prob", type=float,default=1.0, help="Probability of sampling a new level instead of a replay level.")
    parser.add_argument("--plr-level-replay-rho",type=float, default=0.0, help="Minimum size of replay set relative to total number of levels before sampling replays.")
    
    parser.add_argument('--use-qd',                     type=bool_arg,  default=True,       help="Use QD instead of pre-defined distribution")
    parser.add_argument('--qd-use-plr-for-training',    type=bool_arg,  default=True,       help="Use PLR to sample levels for training instead of sampling from archive.")

    # Updates and logging
    # NOTE: The only active argumment here is the log-interval, which pertains
    # the warm starts in this case. Because we're not using a simlution-based
    # objective, there are no updates after warm-start.
    parser.add_argument('--qd-update-interval',         type=int,       default=5, help="Update interval for QD; one QD update per n meta-RL updates")
    # NOTE: max_steps=20, trials_per_episode=10, policy_num_steps=40, num_processes=8
    #       So many environments per QD update do we need? This is how we can
    #       determine how to set qd_update_interval and qd_updates_per_iter.
    #       
    #       It takes (20*10)/40 = 5 meta-RL updates to use up each environment.
    #       With four environments used in parallel, we use 1 environments per
    #       meta-RL update on average.
    #       
    #       In every QD update, batch_size * updates_per_iter * num_emitters
    #       new solutions are evaluated = (5*1*1) = 5 new solutions per QD update.
    #
    #       Thus, to match, we set qd_update_interval = 5, so ever 5 meta-RL
    #       updates (where 4 environments are consumed, we produce 5 new solutions)
    # 
    # NOTE: While this is the minimum, we may also choose to produce an excess,
    #       so we should also test with either (1) more QD updates per iter,
    #       or (2) a smaller qd_update_interval.
    
    parser.add_argument('--qd-updates-per-iter',        type=int,       default=1, help="Number of QD updates per iteration")
    parser.add_argument('--qd-log-interval',            type=int,       default=50, help="Log interval for QD; one QD log per n QD updates")
    # Warm-start 
    parser.add_argument('--qd-warm-start-updates',      type=int_arg,   default=2)
    # Archive
    parser.add_argument('--qd-use-flat-archive',        type=bool_arg,  default=True)
    parser.add_argument('--qd-measures', nargs='+', 
                        default=['stone_reflection', 'stone_rotation', 
                                 'parity_first_stone', 'parity_first_potion', 
                                 'latent_state_diversity', 'average_manhattan_to_optimal'])
    parser.add_argument('--qd-archive-dims', nargs='+', type=int,       default=[112_500])
    parser.add_argument('--qd-initial-population',      type=int_arg,   default=10e000)
    # QD hyperparams
    parser.add_argument('--qd-batch-size',              type=int,       default=5)
    parser.add_argument('--qd-num-emitters',            type=int,       default=1)
    parser.add_argument('--qd-emitter-type',            type=str,       default='me', choices=['es', 'me'])
    parser.add_argument('--qd-mutation-percentage',     type=float,     default=0.02)
    # Objectives
    parser.add_argument('--qd-no-sim-objective',        type=bool_arg,  default=False)

    return parser
