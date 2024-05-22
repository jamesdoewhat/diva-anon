"""
Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
"""

from typing import Any, Sequence, Optional
from functools import partial

import gym
import torch
import random
from gym import spaces
import numpy as np
import time

from qd_metarl.environments.env_utils.vec_env import VecEnvWrapper
from qd_metarl.environments.env_utils.vec_env.dummy_vec_env import DummyVecEnv
from qd_metarl.environments.env_utils.vec_env.subproc_vec_env import SubprocVecEnv
from qd_metarl.environments.env_utils.vec_env.vec_normalize import VecNormalize
from qd_metarl.environments.wrappers import TimeLimitMask, VariBadWrapper
from qd_metarl.environments.mujoco import rand_param_envs

from qd_metarl.level_replay.envs import SeededSubprocVecEnv
from qd_metarl.level_replay.level_sampler import LevelSampler
from qd_metarl.level_replay.level_store import LevelStore
from qd_metarl.environments.wrappers import make_env

from qd_metarl.utils.torch_utils import tensor


def make_vec_envs(
        env_name, 
        seed: int, 
        num_processes: int, 
        gamma: float,
        device: torch.device, 
        trials_per_episode: int,
        normalise_rew: bool, 
        ret_rms: Optional[Any], 
        tasks: Optional[Any],
        rank_offset: int = 0,
        add_done_info: Optional[Any] = None,
        qd: bool = False,
        qd_tasks: Optional[Any] = None,
        plr: bool = False,
        plr_level_sampler: Optional[Any] = None,
        plr_level_sampler_args: Optional[Any] = None,
        plr_env_generator: Optional[Any] = None,
        **kwargs):
    """
    :param ret_rms: running return and std for rewards
    """
    assert seed is not None  # Let's prevent this altogether, for now
    if qd_tasks is None or plr_env_generator == 'sb':
        qd_tasks = [None] * num_processes

    # 1) Create environments and apply (Seeded)SubprocVecEnv wrapper
    env_fns = [
        make_env(
            env_id=env_name, 
            seed=seed,
            rank=rank_offset + i,
            trials_per_episode=trials_per_episode,
            tasks=tasks,
            add_done_info=add_done_info,
            qd_task=qd_tasks[i],  # Used to set QD genotype
            **kwargs) 
        for i in range(num_processes)]
    if plr:
        # PLR: Create SeededSubprocVecEnv
        envs = SeededSubprocVecEnvWrapper(env_fns)
    else:
        # non-PLR: Create (vanilla) SubprocVecEnv
        envs = SubprocVecEnv(env_fns) if len(env_fns) > 1 else DummyVecEnv(env_fns)
    
    init_seeds = [seed + rank_offset + i for i in range(num_processes)]

    # 2) Add functional wrappers such as VecNormalize
    if isinstance(envs.observation_space, (spaces.Box, rand_param_envs.gym.spaces.box.Box)):
        # Only perform VecNormalization if observation space is shape 1
        # Why? Maybe for Mujoco: https://github.com/dannysdeng/dqn-pytorch/blob/master/env.py
        if len(envs.observation_space.shape) == 1:
            envs = VecNormalize(envs, normalise_rew=normalise_rew, ret_rms=ret_rms, gamma=gamma)
    elif isinstance(envs.observation_space, spaces.Dict):
        envs = VecNormalize(envs, normalise_rew=normalise_rew, ret_rms=ret_rms, gamma=gamma)
    else:
        raise NotImplementedError

    # 3) Create VecPyTorch envs (the most high-level wrapper)
    if plr:
        if plr_level_sampler is None:  # Otherwise we assume it is passed in
            # Initialize sampler
            plr_level_sampler = LevelSampler(
                [],  # init_seeds: we don't need these since we're using full distribution
                envs.observation_space, 
                envs.action_space,
                sample_full_distribution=True,
                **(plr_level_sampler_args or {'strategy': 'random'}))
        # Create level store to store genotypes (if applicable)
        plr_level_store = LevelStore() if (plr_env_generator == 'gen' or qd) else None
        envs = VecPyTorch(envs, device, level_sampler=plr_level_sampler, level_store=plr_level_store)
    else:
        plr_level_store = None
        envs = VecPyTorch(envs, device, start_seeds=init_seeds)

    # 4) Return envs and PLR-specific objects (if applicable)
    return envs, (plr_level_sampler, plr_level_store)


class SeededSubprocVecEnvWrapper(SeededSubprocVecEnv):
    """ Used because there was an issue with using SeededSubprocVecEnv directly. """
    def __init__(self, env_fns):
        super(SeededSubprocVecEnv, self).__init__(env_fns)


def solution_from_seed(seed, level_store, lower_bounds, upper_bounds, solution_dim, env_size, gt_type):
    """ Generate or retrieve (if already generated) level from seed. """
    # TODO: Should be able to find environment automatically
    # env_class = MazeEnv
    print('solution from seed:', seed)
    global CarRacingBezier, ExtendedSymbolicAlchemy
    from qd_metarl.environments.box2d.car_racing_bezier import CarRacingBezier
    from qd_metarl.environments.alchemy.alchemy_qd import ExtendedSymbolicAlchemy
    from qd_metarl.environments.toygrid.toygrid import ToyGrid

    # TODO: SOOOO MESSY
    if 'CP' in gt_type:
        env_class = CarRacingBezier
    elif gt_type == 'natural' or gt_type[0] == 'a':
        env_class = ToyGrid
    else:
        env_class = ExtendedSymbolicAlchemy

    # First, check if seed in level store
    if seed in level_store.seed2level:
        # NOTE: For now, we are ignoring the whole encoding thing, as we are
        # not using it (see level_store.get_level for detailes)
        sol = level_store.seed2level[seed]
    # Otherwise, generate level    
    else:
        sol = env_class.genotype_from_seed_static(
            seed, 
            gt_type=gt_type, 
            genotype_lower_bounds=lower_bounds,
            genotype_upper_bounds=upper_bounds,
            genotype_size=solution_dim)  # TODO unify naming
    return sol


def solutions_from_seeds(seeds, level_store, lower_bounds, upper_bounds, solution_dim, env_size, gt_type):
    """ Convert list of seeds to list of generated levels. """
    solutions = []
    for seed in seeds:
        solution = solution_from_seed(seed, level_store, lower_bounds, upper_bounds, solution_dim, env_size, gt_type)
        # Add solution to list
        solutions.append(solution)
    return solutions


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device, level_sampler=None, level_store=None, start_seeds=None):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        self.level_sampler = level_sampler
        self.level_store = level_store
        self.start_seeds = np.array(start_seeds)
        self.cur_seeds = self.start_seeds

    def reset_mdp(self, index=None):
        if index is None: 
            # SubprocVecEnv does not have index---we cannot assume we're getting
            # VecNormalize'd envs
            obs = self.venv.reset_mdp()
        else:
            obs = self.venv.reset_mdp(index=index)
        obs = tensor(obs, device=self.device)
        return obs

    def reset(self, index=None, task=None):
        if index is not None:
            raise NotImplementedError("index: {}".format(index))
        # 1) PLR: Sample level and send seed to environment, if applicable
        if self.level_sampler and task is None:
            # We shouldn't be supplying a task here if we're using PLR;
            # (for GEN or BT); Task will only be supplied if we're using
            # QD with PLR (level_sampler will still be defined)
            # NOTE: we can bypass using QD samples if we set
            #       args.qd_use_plr_for_training; in that case, tasks will
            #       not be passed in.
            # TODO: Is using torch necessary here? This is what PLR code does...
            if index is not None:
                raise NotImplementedError
            else:
                tasks = []
                seeds = torch.zeros(self.venv.num_envs, dtype=torch.int)
                for e in range(self.venv.num_envs):
                    seed = self.level_sampler.sample('gae')
                    seeds[e] = seed
                    # This calls SeededSubprocVecEnv.seed()
                    self.venv.seed(seed, e)
                    if self.level_store is not None:  # for PLR-gen
                        task = solution_from_seed(
                            seed, self.level_store, self.lower_bounds, 
                            self.upper_bounds, self.solution_dim, env_size=self.env_size,
                            gt_type=self.gt_type)
                        tasks.append(task)
                    else: 
                        tasks.append(seed)
                task = tasks
        else:
            # DEBUG: set dummy seeds for QD
            if index is not None:
                seed = 0  # Provide just one seed if we're using index
            else:
                seeds = torch.zeros(self.venv.num_envs, dtype=torch.int)

        # 2) Reset the environment
        if task is None:
            # Increase seeds by multiple of how many environments there are
            if index is not None:
                self.cur_seeds[index] += len(self.start_seeds)
                task = list(self.cur_seeds[index])
            else:
                self.cur_seeds += len(self.start_seeds)
                task = list(self.cur_seeds)
        else: 
            assert isinstance(task, list) or isinstance(task, np.ndarray)
        # SubprocVecEnv does not have index---we cannot assume we're getting
        # VecNormalize'd envs
        if index is None:
            state = self.venv.reset(task=task)
        else:
            state = self.venv.reset(index=index, task=task)
        state = tensor(state, device=self.device)

        # 3) PLR: Return the level seed, if applicable
        if self.level_sampler:
            return state, seeds
        else:
            return state

    def step_async(self, actions):
        # actions = actions.squeeze(1).cpu().numpy()
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        st0 = time.time()
        state, reward, done, info = self.venv.step_wait()
        et0 = time.time()
        info[0]['time/ES-VecPyTorch.step_wait;self.venv.step_wait'] = et0 - st0
        st1 = time.time()
        state = tensor(state, device=self.device)
        # NOTE: We can't just use the `tensor` function below since the
        # unsqueeze is necessary for later dimensionality expectations
        if isinstance(reward, list):  # raw + normalised
            reward = [torch.from_numpy(r).unsqueeze(dim=1).float().to(self.device) for r in reward]
        else:
            reward = torch.from_numpy(reward).unsqueeze(dim=1).float().to(self.device)
        et1 = time.time()
        info[0]['time/ES-VecPyTorch.step_wait;REST'] = et1 - st1
        return state, reward, done, info
    
    def set_genotype_bounds_info(self, lower_bounds, upper_bounds, solution_dim, env_size, gt_type):
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.solution_dim = solution_dim
        self.env_size = env_size
        self.gt_type = gt_type

    def __getattr__(self, attr):
        """ If env does not have the attribute then call the attribute in the wrapped_env """

        if attr in ['_max_trial_steps', 'task_dim', 'belief_dim', 
                    'num_states', 'bit_map_size', 'genotype_size', 'qd_bounds',
                    'genotype_bounds', 'size', 'bit_map_shape', 'gt_type',
                    'compute_measures', 'compute_measures_static', 'get_measures_info',
                    'process_genotype', 'is_valid_genotype']:
            # This will get the attribute value for the first env in the env list.
            # We assume all envs have the same value for this attribute.
            return self.unwrapped.get_env_attr(attr)

        if attr in ['genotype', 'level_rendering', 'get_belief']:
            # These are attributes that have different values for each env
            # NOTE: self.unwrapped is just the first env? (no? I don't think so
            #       --otherwise genotype would be wrong)
            attributes = self.unwrapped.get_env_attrs(attr)
            attributes = [attr[0] for attr in attributes]  # remove list wrapper
            # NOTE: The same "callable" trick here doesn't work; for some
            # reason (still not understood...) it re-initializes the envs
            # before retrieving the attribute. For level_rendering, I turned
            # these methods into properties for now; alternatively I could
            # have called the method in here, but I figured that would be
            # messier.
            return attributes

        try:
            orig_attr = self.__getattribute__(attr)
        except AttributeError:
            orig_attr = self.unwrapped.__getattribute__(attr)

        if callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                return result

            return hooked
        else:
            return orig_attr
