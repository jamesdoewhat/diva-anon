import argparse
from qd_metarl.utils.env_utils import bool_arg, int_arg

parser = argparse.ArgumentParser()

parser.add_argument("wb_label", help="Name of experiment (local)")
parser.add_argument('--env-type', type=str, default='gridworld_varibad')
parser.add_argument('--use-wb', default=False, action='store_true')
parser.add_argument("--wb-notes", "-n", type=str, default="Untitled", help="Experimental notes (W&B)")
parser.add_argument("--wb-tags", "-t", nargs="+", default=list(), help="Experimental tags (W&B)")
parser.add_argument("--wb-entity", type=str, default="CHOOSE", help="W&B entity")
parser.add_argument("--wb-project", type=str, default="CHOOSE", help="W&B project")
parser.add_argument("--profile", action="store_true", help="Profile algorithm with cProfile to determine performance bottlenecks")
parser.add_argument("--debug", action="store_true", help="Debug mode (few warm starts, no wandb, etc.)")
parser.add_argument('--seed',  nargs='+', type=int, default=[73])
parser.add_argument('--deterministic-execution', type=bool_arg, default=False, help='Make code fully deterministic. Expects 1 process and uses deterministic CUDNN')
parser.add_argument('--dense-rewards', action='store_true', default=False)
parser.add_argument('--skip-eval', action='store_true', default=False)
parser.add_argument('--qd-supported-env', type=bool_arg, default=True, help='Environment has QD support implemented')

# Misc
parser.add_argument('--use-popart',                 type=bool_arg,      default=False)

# HyperX reward bonuses default arguments: which exploration bonus(es) to use
parser.add_argument('--exploration-bonus-hyperstate',type=bool_arg,     default=False, help='bonus on (s, b)')
parser.add_argument('--exploration-bonus-state',    type=bool_arg,      default=False, help='bonus only on (s)')
parser.add_argument('--exploration-bonus-belief',   type=bool_arg,      default=False, help='bonus only on (b)')
parser.add_argument('--exploration-bonus-vae-error',type=bool_arg,      default=False)

## QD default parameters
parser.add_argument('--use-qd',                     default=False,      action='store_true', help="Use QD instead of pre-defined distribution")
parser.add_argument('--qd-log-interval',            type=int,           default=100,    help="Log interval for QD; one QD log per n QD updates")
# QD hyperparams
parser.add_argument('--qd-batch-size',              type=int,           default=5,      help="Batch size for each QD emitter on ask()")
parser.add_argument('--qd-initial-population',      type=int_arg,       default=10)
parser.add_argument('--qd-num-emitters',            type=int,           default=5)
parser.add_argument('--qd-emitter-type',            type=str,           default='es',   choices=['es', 'me'])
parser.add_argument('--qd-mutation-percentage',     type=float,         default=0.05)
parser.add_argument('--qd-use-constant-mutations',  type=bool_arg,      default=False)
parser.add_argument('--qd-mutations-constant',      type=int_arg,       default=4)
parser.add_argument('--qd-es-sigma0',               type=float,         default=1.0)
parser.add_argument('--qd-stepwise-mutations',      type=bool_arg,      default=False)
# Warm start
parser.add_argument('--qd-use-two-stage-ws',        type=bool_arg,      default=False)
parser.add_argument('--qd-warm-start-updates',      type=int_arg,       default=20_000)
parser.add_argument('--qd-init-warm-start-updates', type=int_arg,       default=20_000)
parser.add_argument('--qd-warm-start-no-sim-objective', type=bool_arg,  default=True,   help="Don't use simulation-based objective for warm start.")
parser.add_argument('--qd-warm-start-only',         type=bool_arg,      default=False)
# Objectives
parser.add_argument('--qd-no-sim-objective',        type=bool_arg,      default=True,   help="Don't use simulation-based objective at all.")
parser.add_argument('--qd-gt-diversity-objective',  type=bool_arg,      default=False,  help="Use genotype-based diversity objective.")
parser.add_argument('--qd-meas-diversity-objective',type=bool_arg,      default=False,  help="Use measure-based diversity objective.")
parser.add_argument('--qd-meas-diversity-measures', nargs='+', type=str,default=[],     help="Measure(s) to use for measure-based diversity objective.")
parser.add_argument('--qd-meas-alignment-objective',type=bool_arg,      default=False,  help="Use measure-based alignment objective.")
parser.add_argument('--qd-meas-alignment-measures', nargs='+', type=str,default=[],     help="Measure(s) to use for measure-based alignment objective.")
parser.add_argument('--qd-randomize-objective',     type=bool_arg,      default=False,  help="Use random objective (better than constant because *stochasticity*). Will not use any other objective!")
parser.add_argument('--qd-bias-new-solutions',      type=bool_arg,      default=False,  help='Slightly bias newer solutions by adding a slowy incrementing constant float to the objective')
# Archive construction
parser.add_argument('--qd-measures',                nargs='+',          default=[])
parser.add_argument('--qd-archive-dims', nargs='+', type=int,           default=[],     help="Dimensions of final archive (used for WS2).")
parser.add_argument('--qd-init-archive-dims', nargs='+', type=int,      default=[],     help="Dimensions of archive for WS1.")
parser.add_argument('--qd-use-flat-archive',        type=bool_arg,      default=False,  help="Do we use a flat archive?")
# Refreshing stale solutions
parser.add_argument('--qd-refresh-archive',         type=bool_arg,      default=False,  help="Do we periodically update stale solutions in the archive?")
parser.add_argument('--qd-sslu-threshold',          type=int,           default=100,    help="How many steps we allow solutions to without their objectives being updated.")
# Sample mask
parser.add_argument('--qd-update-sample-mask',      type=bool_arg,      default=False,  help="Do we periodically update the sample mask to bring it closer to the target region?")
parser.add_argument('--qd-sample-mask-min-solutions', type=int_arg,     default=20)
parser.add_argument('--qd-sparsity-reweighting',    type=bool_arg,      default=False)
parser.add_argument('--qd-sparsity-reweighting-sigma', type=float,      default=0.1)
# Sampling
parser.add_argument('--qd-use-plr-for-training',    type=bool_arg,      default=False,  help="Use PLR to sample levels for training instead of sampling from archive.")
# Loading archives
parser.add_argument('--qd-load-archive-from',       type=str,           default='')
parser.add_argument('--qd-load-archive-run-index',  type=int,           default=0)
parser.add_argument('--qd-plr-integration',         type=bool_arg,      default=False,  help="Do we integrate PLR into QD training?")

## Deprecated QD arguments
# No longer using measure selector
parser.add_argument('--qd-use-measure-selector',    type=bool_arg, default=False)
parser.add_argument('--qd-measure-selector-num-dims', type=int, default=2)
parser.add_argument('--qd-measure-selector-range', nargs='+', type=float, default=[-0.2, 1.2])
parser.add_argument('--qd-measure-selector-num-samples', type=int, default=500)
parser.add_argument('--qd-measure-selector-resolution', type=int, default=50)
parser.add_argument('--qd-measure-selector-dim-red-method', type=str, default='vae')
parser.add_argument('--qd-measure-selector-use-all-measures', type=bool_arg, default=False)
parser.add_argument('--qd-measure-selector-use-neg-samples', type=bool_arg, default=False)
parser.add_argument('--qd-unique-measure-selector', type=bool_arg, default=False)
# We don't currently have any use for inserting downstream samples into the archive
parser.add_argument('--qd-num-downstream-samples-to-use', type=int, default=0, help="Number of downstream samples to use for QD")
# KD arguments (deprecated)
parser.add_argument('--qd-kd-smoothing-coef', type=float, default=1.0, help="Smoothing coefficient for KD")
parser.add_argument('--qd-use-kd', type=bool_arg, default=False, help="Use KD for QD")
parser.add_argument('--qd-kd-num-samples', type=int, default=100, help="Number of samples for KD")
parser.add_argument('--qd-update-interval', type=int, default=5, help="Update interval for QD; one QD update per n meta-RL updates")
# We no longer update QD over the course of training (for this project)
parser.add_argument('--qd-async-updates', type=bool_arg, default=False)
parser.add_argument('--qd-updates-per-iter', type=int, default=1, help="Number of QD updates per iteration")

# Prioritized Level Replay arguments.
parser.add_argument('--use-plr', default=False, action='store_true', help="Use prioritized level replay")
parser.add_argument("--plr-buffer-size-from-qd", type=bool_arg, default=False, help="Match reply buffer size to QD archive size.")
parser.add_argument("--plr-level-replay-score-transform", type=str, default='softmax', choices=['constant', 'max', 'eps_greedy', 'rank', 'power', 'softmax'], help="Level replay scoring strategy")
parser.add_argument("--plr-level-replay-temperature-start", type=float,default=1.0, help="Level replay temperature")
parser.add_argument("--plr-level-replay-temperature-end", type=float,default=1.0, help="Level replay temperature")
parser.add_argument("--plr-level-replay-strategy", type=str, default='value_l1',
                    choices=['off', 'random', 'sequential', 'policy_entropy', 
                            'least_confidence', 'min_margin', 'gae', 'value_l1', 
                            'one_step_td_error', 'uniform', 'signed_value_loss',
                            'positive_value_loss', 'grounded_signed_value_loss',
                            'grounded_positive_value_loss', 'alt_advantage_abs', 
                            'tscl_window'], help="Level replay scoring strategy")
parser.add_argument("--plr-level-replay-eps", type=float,default=0.05, help="Level replay epsilon for eps-greedy sampling")
parser.add_argument("--plr-level-replay-schedule",type=str,default='proportionate', help="Level replay schedule for sampling seen levels")
parser.add_argument("--plr-level-replay-rho",type=float, default=1.0, help="Minimum size of replay set relative to total number of levels before sampling replays.")
parser.add_argument("--plr-replay-prob", type=float,default=0.20, help="Probability of sampling a new level instead of a replay level.")
parser.add_argument("--plr-level-replay-alpha",type=float, default=0.0, help="Level score EWA smoothing factor")
parser.add_argument("--plr-staleness-coef",type=float, default=0.3, help="Staleness weighing")
parser.add_argument("--plr-staleness-transform",type=str, default='power',
                    choices=['max', 'eps_greedy', 'rank', 'power', 'softmax'], help="Staleness normalization transform")
parser.add_argument("--plr-staleness-temperature",type=float, default=1.0, help="Staleness normalization temperature")
parser.add_argument("--plr-weight-log-interval", type=int, default=1, help="Save level weights every this many updates")
parser.add_argument("--plr-seed-buffer-size",type=int,default=2500, help="Number of levels to store in our buffer, after which we replace")
parser.add_argument('--plr-env-generator', type=str, default='sb', choices=['sb', 'gen'], help='Environment generator used as input for PLR. ' 
                    '\'sb\' is for manicured seed-based distribution ' 
                    'and \'gen\' is for random generation using same '
                    'genotype method as QD.')

# --- VAE ---
parser.add_argument('--vae-use-ensemble', default=False, type=bool_arg, help='use ensemble of VAEs')

# --- Alchemy ---
parser.add_argument('--alchemy-mods', nargs='+', type=str, default=['fix-pt-map', 'fix-graph'])
# NOTE: only Alchemy and Racing use this
parser.add_argument('--max-steps', type=int, default=20)
parser.add_argument('--alchemy-use-dynamic-items', type=bool_arg, default=True)

# --- XLand ---
parser.add_argument('--xland-grid-size', type=int, default=13)
parser.add_argument('--xland-grid-type', type=str, default='R1', 
                    choices=['R1', 'R2', 'R4', 'R6', 'R9'])

# --- Racing ---
parser.add_argument('--racing-fps', type=int, default=15)
parser.add_argument('--racing-obs-type', type=str, default='f15')