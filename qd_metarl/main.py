"""
Main scripts to start experiments.
Takes a flag --env-type (see below for choices) and loads the parameters from 
the respective config file.
"""
import os
import sys
import warnings
from copy import deepcopy
from numba import NumbaDeprecationWarning
import subprocess
import atexit

# Suppress specific warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Disables annoying TF warnings
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=NumbaDeprecationWarning)

display_number = "1"
lock_file = f"/tmp/.X{display_number}-lock"

def start_xvfb(display):
    return subprocess.Popen(["Xvfb", display, "-screen", "0", "1024x768x24"])

def cleanup():
    if 'xvfb_process' in locals():
        xvfb_process.terminate()

if not os.path.exists(lock_file):
    xvfb_process = start_xvfb(f":{display_number}")
atexit.register(cleanup)
os.environ["DISPLAY"] = f":{display_number}"

import argparse
from argparse import Namespace
from arguments import parser

import numpy as np
import cProfile
import pstats
import pprint
from io import StringIO


def main(): 
    from qd_metarl.utils.torch_utils import select_device
    from qd_metarl.utils.env_parsers import ENV_PARSERS, get_vec_env_kwargs
    from qd_metarl.environments.parallel_envs import make_vec_envs
    from qd_metarl.metalearner import MetaLearner
    from qd_metarl.utils.torch_utils import DeviceConfig
    from qd_metarl.utils.plr_utils import get_plr_args_dict   

    pp = pprint.PrettyPrinter(indent=4)

    # Directly collect all command-line arguments, just to get env_type
    all_args, unknown_args = parser.parse_known_args()
    base_parser = parser

    env = all_args.env_type
    if 'rl2' in env:
        use_rl2 = True
    else:
        use_rl2 = False

    # TODO: FIX FIX FIX THIS! It's so messy
    # Remove rest of string after final _ (to get rid of "varibad")
    if 'hyperx' not in env and \
       'belief_oracle' not in env and \
       'evaribad' not in env:
        env = env[:env.rfind('_')]

    print('Environment name:', env)
    assert env in ENV_PARSERS, f'Environment {env} not found in ENV_PARSERS'
    config_parser = ENV_PARSERS[env](use_rl2)
    
    cmd_args = sys.argv[1:]
    config_args, remaining_args = config_parser.parse_known_args(cmd_args)
    base_args, _ = parser.parse_known_args(remaining_args)
    merged_args_dict = {**vars(base_args), **vars(config_args)}
     
    # Create the final Namespace object
    args = Namespace(**merged_args_dict)
    
    # Print final args
    print('\nFinal args:')
    pp.pprint(vars(args))

    # For dubugging
    if args.debug:
        args.qd_warm_start_updates = 20
        args.qd_warm_start_no_sim_objective = True
        args.num_processes = 3
        args.num_frames = 10_000

    # If only warm-start for QD, set num-processes to 3 to preserve memory
    if args.use_qd and args.qd_warm_start_only:
        args.num_processes = 3

    # check if we're adding an exploration bonus
    args.add_exploration_bonus = args.exploration_bonus_hyperstate or \
                                 args.exploration_bonus_state or \
                                 args.exploration_bonus_belief or \
                                 args.exploration_bonus_vae_error

    # warning for deterministic execution
    if args.deterministic_execution:
        print('Envoking deterministic code execution.')
        if torch.backends.cudnn.enabled:
            warnings.warn('Running with deterministic CUDNN.')
        if args.num_processes > 1:
            raise RuntimeError(
                'If you want fully deterministic, run with num_processes=1.'
                'Warning: This will slow things down and might break A2C if '
                'policy_num_steps < env._max_trial_steps.')

    os.environ['TF_TENSORRT_DISABLED'] = '1'
    
    # Select device
    # GPU selection
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:  # If not defined we use GPU by default
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = os.environ['CUDA_VISIBLE_DEVICES']
    if len(device) == 0:
        # -1 is CPU
        select_device(-1)
        print("\nDevice: CPU")
    else:
        # Non-negative integer is the index of GPU
        select_device(0)
        print("\nDevice: GPU", device)

    # if we're normalising the actions, we have to make sure that the env 
    # expects actions within [-1, 1]
    if args.norm_actions_pre_sampling or args.norm_actions_post_sampling:
        # NOTE: We don't actually use these environments; just for assertions
        envs, _ = make_vec_envs(
            env_name=args.env_name, seed=0, num_processes=args.num_processes,
            gamma=args.policy_gamma, device='cpu',
            trials_per_episode=args.trials_per_episode,
            normalise_rew=args.norm_rew_for_policy, ret_rms=None,
            plr=args.use_plr,
            tasks=None,
            )
        assert np.unique(envs.action_space.low) == [-1]
        assert np.unique(envs.action_space.high) == [1]
        envs.close()
        del envs

    # clean up arguments
    if args.disable_metalearner or args.disable_decoder:
        args.decode_reward = False
        args.decode_state = False
        args.decode_task = False

    if hasattr(args, 'decode_only_past') and args.decode_only_past:
        args.split_batches_by_elbo = True

    # Init profiler
    if args.profile:
        args.num_frames = 1000
        pr = cProfile.Profile()
        pr.enable()

    # Weights and Biases
    if args.use_wb:
        assert args.wb_label is not None
        args.wb_exp_index = args.wb_label.split('_')[0]

    # We only use one seed for the main training loop
    assert isinstance(args.seed, int) or isinstance(args.seed[0], int)
    args.seed = args.seed if isinstance(args.seed, int) else args.seed[0]
    print('Training with seed: ', args.seed)

    # Prepare environment arguments
    args.action_space = None  # In case it wasn't defined (?)
    args.vec_env_kwargs = get_vec_env_kwargs(args)

    print('vec_env_kwargs:', args.vec_env_kwargs)


    # Prepare some things before QD
    try:
        # Account for two-stage warm-start
        if args.qd_use_two_stage_ws:
            init_archive_dims = args.qd_init_archive_dims
            reg_archive_dims = args.qd_archive_dims
            init_total_cells = np.prod(init_archive_dims)
            reg_total_cells = np.prod(reg_archive_dims)
            # We initialize PLR with maximum total cells between the two archives
            total_cells = max(init_total_cells, reg_total_cells)
            # Set current archive dims to initial archive dims
            archive_dims = init_archive_dims
        else:
            # Set PLR buffer size normally
            archive_dims = args.qd_archive_dims # Default
            total_cells = np.prod(archive_dims)

        if args.plr_buffer_size_from_qd:
            print('Setting PLR seed buffer size based on QD archive: ', total_cells, '+ 10_000')
            args.plr_seed_buffer_size = int(total_cells) + 10_000
        elif args.use_qd and args.use_plr:
            print('Adding 1000 to PLR seed buffer size to account for temporary QD excess during eval: ', total_cells, '+ 1000')
            args.plr_seed_buffer_size = args.plr_seed_buffer_size + 10_000
        else:
            print('Not setting PLR seed buffer size based on QD archive.')
        print('args.plr_seed_buffer_size:', args.plr_seed_buffer_size)
        # Adding 10_000 to account for when archive gets too full---we need
        # this extra space because otherwise, when we find new solutions
        # and our archive/buffer is full, when we try to insert, the
        # fully plr buffer will prevent us from being able to insert these
        # solutions. This is a temporary fix!
    except Exception as e:
        # TODO: We should remove this try: except if possible
        print('Exception with QD preparation:', e)
        print('Continuing. Assuming unnecessary for this experiment.')
        archive_dims = np.prod(args.qd_archive_dims)

    # Initialize environments here for memory efficiency
    if args.use_plr:
        if args.use_qd and args.qd_no_sim_objective:
            args.plr_level_replay_strategy = 'random'  # Ignore objectives
        plr_num_actors = args.qd_batch_size * 5 if args.use_qd else args.num_processes 
        plr_level_sampler_args = get_plr_args_dict(args, plr_num_actors)
    else:
        plr_level_sampler_args = None
    # Initialise environments and plr level sampler (if applicable)
    envs, plr_components = make_vec_envs(
        env_name=args.env_name, 
        seed=args.seed, 
        num_processes=args.num_processes,
        gamma=args.policy_gamma, 
        device=DeviceConfig.DEVICE,
        trials_per_episode=args.trials_per_episode,
        normalise_rew=args.norm_rew_for_policy, 
        ret_rms=None,
        tasks=None, 
        qd=args.use_qd,
        qd_tasks=None,
        plr=args.use_plr,
        plr_level_sampler_args=plr_level_sampler_args,
        plr_level_sampler=None,  # We initialize inside the function for now
        plr_env_generator=args.plr_env_generator,
        dense_rewards=args.dense_rewards, 
        **args.vec_env_kwargs)

    # Meta-learn
    learner_fn = MetaLearner
    learner = learner_fn(args, envs, plr_components, archive_dims=archive_dims)
    learner.train()
    
    # Profiling output
    if args.profile:
        s = StringIO()
        pr.disable()
        sortby = 'cumtime'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())


if __name__ == '__main__':
    import torch
    torch.multiprocessing.set_start_method('spawn')
    main()
