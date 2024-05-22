from typing import Tuple, Any
import logging
from argparse import Namespace
import os
import time
from scipy.sparse import csgraph


import itertools
import math
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torch.nn import functional as F
import gym
import gym_minigrid.minigrid as minigrid
from gym import spaces
import numpy as np
import torch
from collections import deque

from qd_metarl.environments.maze.envs.multigrid import MultiGridEnv, Grid
from qd_metarl.utils import env_utils as utl
from qd_metarl.utils.torch_utils import DeviceConfig
from qd_metarl.qd.measures.toygrid_measures import ToyGridMeasures

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


logger = logging.getLogger(__name__)


def sample_grid_location(
        size=None,
        region=None,
        sample_mode='uniform',
        border_size=0,
        seed=None,
        prng=None, 
        prevs=[],
):
    """
    Sample maze location given various specifications and constraints.

    NOTE: Either size or region must be provided.

    Args:
        size: Number of cells in (presumed square) maze.
        region: Region within maze to sample from (all inclusive). The region
            is defined by (i_1, j_1, i_2, j_2), where 1 is top left point and
            2 is bottom right.
        sample_mode: Method for sampling points within region.
        border_size: How many cells inwards to exclude from the boundary.
        seed: Seed used for RNG.
        prng: Random number generator.
        prevs: List of previously sampled points to avoid.
    Returns (tuple): (i, j) sampled location; (y, x).
    """
    if prng is None:
        prng = np.random.RandomState(seed)
    if size is None and region is None:
        raise ValueError('Need to provide either region or size.')

    # Define region if only size passed in
    if region is None:
        # If no region provided, allow sampling over full size of maze
        region = (0, 0, size-1, size-1)  # TODO: is this right?
    else:
        assert len(region) == 4
    
    # Potentially shrink region to account for maze border
    r = region
    bs = border_size
    r = (r[0]+bs, r[1]+bs, r[2]-bs, r[3]-bs)
    if sample_mode == 'uniform':
        while True:
            i = prng.randint(r[0], r[2]+1)
            j = prng.randint(r[1], r[3]+1)
            if (i, j) in prevs:
                continue
            else:
                return (i, j), prng
    else:
        raise NotImplementedError(f'Unknown sample mode: {sample_mode}')


class ToyGrid(MultiGridEnv):
    """Single-agent maze environment specified via a bit map."""

    def __init__(
        self,                       #
        size:                       int = 11,
        # num_y_pos_terms:            int = 16,
        minigrid_mode:              bool = True,
        max_steps:                  int = None,
        reward_type:                str = 'sparse',
        initial_goal_visibility:    str = 'visible',
        goal_pos:                   Tuple[int] = None,
        goal_sampler:               str = 'uniform',
        goal_sampler_region:        Tuple[int] = None,
        variable_trial_lengths:     bool = False,
        seed:                       int = None,
        distribution_type:          str = 'SB',
        gt_type:                    str = 'a16',
        visualize:                  bool = False,
        dense_rewards:              bool = False,
    ):
        """ Maze environment initialization.
        
        Args:
        - size (int): Size of maze (odd number)--includes outer walls.
        - num_y_pos_terms (int): Number of y position terms in genotype.
        - agent_view_size (int): Size of agent's view.
        - minigrid_mode (bool): Whether to use minigrid mode (not multigrid).
        - max_steps (int): Maximum number of steps in episode.
        - reward_type (str): Type of reward. Options are:
            - 'sparse': 1 if agent reaches goal, 0 otherwise.
            - 'dense': (proximity-based reward)
        - initial_goal_visibility (str): Whether goal is visible at start.
          Options are:
            - 'visible': Goal is visible at start.
            - 'invisible': Goal is not visible at start.
        - goal_pos (Tuple[int]): Goal position of agent.
        - goal_sampler (str): How to sample goal position. Options are:
            - 'uniform': Uniformly sample from all locations.
            - 'edges': Sample from edges of maze.
        - goal_sampler_region (Tuple[int]): Region to sample goal position
            from. If None, sample from entire maze.
        - variable_trial_lengths (bool): If we don't care about trajectories being the
          same length, we can set this to True. If False, when agent 
          reaches goal or dies, we keep episode running.
        - seed (int): Seed for RNG.
        - distribution_type: How to generate maze. Options are:
            - 'SB': Use seeded backtracking algorithm
                        and location specifications to generate maze.
            - 'QD': Use QD genotype to generate maze.
        - gt_type: How to interpret genotype. Options are:
            - 'aN': genotype is X, and N locations summed for Y
            - 'natural': genotype is X and Y
        - visualize (bool): Whether to visualize environment.
        - dense_rewards (bool): Whether to use dense rewards.
        """
        del visualize  # Unused for this environmnent
        self.size = size
        assert size == 11  # Hardcoded for now
        self.reward_type = reward_type
        self.rng_seed = seed
        self.distribution_type = distribution_type
        self.gt_type = gt_type
        self.dense_rewards = dense_rewards

        self.start_pos = (0, 0)
        self.goal_pos = goal_pos
        self._init_goal_pos = goal_pos
        self.goal_sampler = goal_sampler
        self.goal_sampler_region = goal_sampler_region

        if type(self.goal_sampler_region) is str:
            if self.goal_sampler_region == 'left':
                # i1 (y1), j1 (x1), i2 (y2), j2 (x2)
                # 'i' is top left point and 'j' is bottom right
                self.goal_sampler_region = (0, 0, size-1, size//2)
            elif self.goal_sampler_region == 'right':
                self.goal_sampler_region = (0, size//2, size-1, size-1)
            else:
                raise NotImplementedError
        
        self.goal_visibility = initial_goal_visibility

        self.bit_map = np.zeros((size - 2, size - 2))
        self.bit_map_padded = np.pad(self.bit_map, pad_width=1, mode='constant', constant_values=1)
        self.bit_map_shape = self.bit_map.shape
        self.bit_map_size = self.bit_map.size

        # Generate level from specifications
        if self.distribution_type == 'SB':
            self.generate_level_from_seed()
        elif self.distribution_type == 'QD':
            self.genotype_set = False
            self.generate_level_from_genotype(
                genotype=None, gt_type=self.gt_type)
        else:
            raise NotImplementedError(
                f'Unknown distribution type: {self.distribution_type}')

        # Set max_steps as function of size of maze, by default
        if max_steps is None:
            max_steps = 2 * size * size

        # For QD+MetaRL
        self._max_trial_steps = max_steps
        self.num_states = (size - 2)**2
        self.variable_trial_lengths = variable_trial_lengths

        super().__init__(
            n_agents=1,
            grid_size=size,
            agent_view_size=3,  # NOTE: doesn't matter
            max_steps=max_steps,
            see_through_walls=True,  # Set this to True for maximum speed
            minigrid_mode=minigrid_mode,
            seed=seed
        )

        self.grid_size = size
        
        # Observation space is just agent position
        self.coords_obs_space = gym.spaces.Box(
            low=0,
            high=self.grid_size,
            shape=(3,),  # also hardcoded
            dtype="uint8")

        self.observation_space = self.coords_obs_space

        # For QD+MetaRL
        self.genotype = None
        self.set_genotype_info()
        self.success = False

    def step(self, action):
        """ Step function. """

        success_prev = self.success

        obs, reward, done, info = super().step(action)

        if reward == 0:
            reward -= 0.05  # Whenever we have zero reward, we instead penalize
        
        if self.dense_rewards:
            # Compute dense reward
            agent_pos = np.array(self.agent_pos[0])
            goal_pos = np.array(self.goal_pos)
            dist = np.linalg.norm(agent_pos - goal_pos)
            
            # Normalize distance by the diagonal of the maze
            max_distance = np.sqrt(self.size**2 + self.size**2)
            normalized_dist = dist / max_distance
            
            # Scale reward with respect to max_steps
            scale_factor = 0.1 / self.max_steps
            dense_reward = -scale_factor * self.max_steps * (np.exp(normalized_dist) - 1)
            
            # Add dense reward
            reward += dense_reward

        if self.success:
            info['success'] = True
        else:
            info['success'] = False

        # If we've just succeeded, log as event
        if self.success and not success_prev:
            info['event'] = 'goal reached'

        return obs, reward, done, info
        
    def seed(self, seed=None):
        """ Set seed. """
        if seed is not None: 
            seed = int(seed)
        super().seed(seed=seed)
        self.rng_seed = seed

    def _gen_grid(self, width, height):
        """Generate grid from start/goal locations."""
        if self.distribution_type == 'QD' and not self.genotype_set:
            return
        # Create an empty grid
        self.grid = Grid(width, height)
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        # Goal
        self.put_obj(minigrid.Goal(), self.goal_pos[0], self.goal_pos[1])
        # Agent
        self.place_agent_at_pos(0, self.start_pos)
    
    def set_genotype_info(self):
        """ Extract information from genotype and genotype type. """

        self.measures = ToyGridMeasures.get_all_measures()

        if self.gt_type[0] == 'a':
            num_y_pos_terms = int(self.gt_type[1:])
            # Genotype is goal location, with unnecessary y position terms
            # Length is [ 1    +    num_y_terms ] 
            #           [ x pos  +  y pos terms ]
            self.genotype_size = 1 + num_y_pos_terms
            # Lower bound for goal in x position is 1, since 0 is wall; 
            # For y positions, each term is either -1, 0, or 1, and we just
            # clip the sum to be within the appropriate range
            self.genotype_lower_bounds = np.array(
                [1]             + [-1] * num_y_pos_terms)
            self.genotype_upper_bounds = np.array(
                [self.size - 2] + [+1] * num_y_pos_terms)                                    

        elif self.gt_type == 'natural':
            # Genotype is just direct goal location
            # Length is [ 1      +      1 ] 
            #           [  x pos + y pos  ]
            self.genotype_size = 2
            self.genotype_lower_bounds = np.array([1, 1])
            self.genotype_upper_bounds = np.array([self.size - 2, self.size - 2])
        else:
            raise ValueError('Unknown genotype type: {}'.format(self.gt_type))
        
        self.genotype_bounds = [(l, u) for l, u in
                                zip(list(self.genotype_lower_bounds), 
                                    list(self.genotype_upper_bounds))]

    def pos_to_flat_pos(self, pos):
        """ Convert position tuple to flattened index. """
        return (pos[0]-1) * (self.size - 2) + (pos[1]-1)
    
    def get_measures_info(self, env_name):
        """ Get info about measures. """
        return ToyGridMeasures.get_measures_info(env_name)

    def compute_measures(self, genotype=None, measures=None, return_pg=False):
        """" Compute the measures for the given maze. """
        # Use current maze measures and genotype if none provided
        if measures is None:
            measures = self.measures
        if genotype is None:
            genotype = self.genotype

        # Extract useful properties of genotype
        pg = self.process_genotype_new(genotype)

        # Compute measures
        meas = ToyGridMeasures.compute_measures(
            genotype=genotype,
            goal_pos = pg.goal_pos,
            measures=measures
        )

        if return_pg:
            return meas, pg
        else:
            return meas
    
    @staticmethod
    def compute_measures_static(genotype=None, size=None, measures=None, gt_type=None, return_pg=False):
        """" Compute the measures for the given maze. """

        # Extract useful properties of genotype
        pg = ToyGrid.process_genotype(genotype, size, gt_type=gt_type)

        # Compute measures
        meas = ToyGridMeasures.compute_measures(
            genotype=genotype,
            goal_pos = pg.goal_pos,
            measures=measures
        )

        if return_pg:
            return meas, pg
        else:
            return meas

    def genotype_from_seed(self, seed, level_store=None):
        """ Generate or retrieve (if already generated) level from seed. """
        # First, check if seed in level store
        if level_store is not None and seed in level_store.seed2level:
            # NOTE: For now, we are ignoring the whole encoding thing, as we are
            # not using it (see level_store.get_level for detailes)
            # TODO: Why is this never happening? Before the variable was named
            # "solution" instead of "sol", and it was not crashing the code.
            sol = level_store.seed2level[seed]
        # Otherwise, generate level    
        else:
            num_attempts = 0
            rng = np.random.default_rng(seed)
            while True:
                # Keep using this seed to generate environments until one is valid
                num_attempts += 1
                sol = rng.integers(
                    low=self.genotype_lower_bounds,
                    high=self.genotype_upper_bounds + 1,
                    size=(self.genotype_size))

                # Check if solution is valid
                pg = self.process_genotype_new(sol)
                valid, reason = self.is_valid_genotype(
                    pg, gt_type=self.gt_type)
                del reason  # TODO: if failed, print most common reason
                if valid:
                    # print('Found valid in {} attempts'.format(num_attempts))
                    break
                if num_attempts == 100:
                    print('WARNING: Could not sample a valid solution after 100 attempts')
                if num_attempts > 100_000:
                    raise RuntimeError("Could not sample a valid solution")
        return sol
    
    @staticmethod
    def genotype_from_seed_static(
            seed, gt_type='a16', genotype_lower_bounds=None, 
            genotype_upper_bounds=None, genotype_size=None):
        num_attempts = 0
        rng = np.random.default_rng(seed)
        while True:
            # Keep using this seed to generate environments until one is valid
            num_attempts += 1
            sol = rng.integers(
                low=genotype_lower_bounds,
                high=genotype_upper_bounds + 1,
                size=(genotype_size))

            # Check if solution is valid
            pg = ToyGrid.process_genotype(sol, size=11, gt_type=gt_type)  # Hardcoded for now!
            valid, reason = ToyGrid.is_valid_genotype(
                pg, gt_type=gt_type)
            del reason  # TODO: if failed, print most common reason
            if valid:
                # print('Found valid in {} attempts'.format(num_attempts))
                break
            if num_attempts == 100:
                print('WARNING: Could not sample a valid solution after 100 attempts')
            if num_attempts > 100_000:
                raise RuntimeError("Could not sample a valid solution")
        return sol


    def process_genotype_new(self, genotype):
        """Extract information from genotype and genotype type."""
        gisn = genotype is None

        assert gisn or len(genotype) == self.genotype_size, \
            f'Genotype length {len(genotype)} != {self.genotype_size}'
                    
        # print('\nProcess genotype: {}'.format(genotype))

        if self.gt_type[0] == 'a':
            # Genotype is goal location, with unnecessary y position terms
            # Length is [ 1    +    num_y_terms ]
            #           [ x pos  +  y pos terms ]
            # E.g. if size = 11, then we get 4; this is how many blank cells
            # above and below goal position
            num_y_pos_terms = int(self.gt_type[1:])
            y_num = (self.size - 2 - 1) // 2 
            # Ensure that the number of terms we're using for y position is a
            # multiple of y_num, so division is clean
            assert num_y_pos_terms % y_num == 0
            # y_div is what we will divide our sum by to get final y position
            y_div = num_y_pos_terms // y_num
            # Get raw x position and y position terms
            x_pos = None if gisn else genotype[0]
            y_pos_terms = None if gisn else genotype[1:]
            # Get scaled y sum
            y_unshifted = None if gisn else np.sum(y_pos_terms) // y_div
            # For e.g. size = 11 for, we will get values between -4 and 4, so 
            # we shift by 5 to get values between 1 and 9 (but do for 
            # general case)
            y_shift = (self.size - 2 - 1) // 2 + 1
            # Get final y position
            y_pos = None if gisn else y_unshifted + y_shift

        elif self.gt_type == 'natural':
            # Genotype is just direct goal location
            # Length is [ 1      +      1 ] 
            #           [  x pos + y pos  ]
            x_pos = None if gisn else genotype[0]
            y_pos = None if gisn else genotype[1]

        goal_pos = None if gisn else (y_pos, x_pos)

        # Return processed genotype
        processed_genotype = {
            'grid_size': self.size,
            'start_pos': (self.size // 2, self.size // 2),
            'goal_pos': goal_pos,
            'genotype': genotype,
            'genotype_bounds': self.genotype_bounds,
            'genotype_size': self.genotype_size,
            'genotype_lower_bounds': self.genotype_lower_bounds,
            'genotype_upper_bounds': self.genotype_upper_bounds,
        }
        
        return Namespace(**processed_genotype)
        
    @staticmethod
    def process_genotype(genotype, size=None, gt_type='a64'):
        """Extract information from genotype and genotype type."""
        gisn = genotype is None

        # print('\nProcess genotype: {}'.format(genotype))

        if gt_type[0] == 'a':
            # Genotype is goal location, with unnecessary y position terms
            # Length is [ 1    +    num_y_terms ]
            #           [ x pos  +  y pos terms ]
            # E.g. if size = 11, then we get 4; this is how many blank cells
            # above and below goal position
            num_y_pos_terms = int(gt_type[1:])     # E.g. 128
            y_num = (size - 2 - 1) // 2  # For 11, this is 8//2 = 4
            genotype_size = 1 + num_y_pos_terms    # a128: = 1 + 128
            # Ensure that the number of terms we're using for y position is a
            # multiple of y_num, so division is clean
            assert num_y_pos_terms % y_num == 0    # 128 % 4 = 0
            # y_div is what we will divide our sum by to get final y position
            y_div = num_y_pos_terms // y_num       # 128 // 4 = 32
            # Get raw x position and y position terms
            x_pos = None if gisn else genotype[0]
            y_pos_terms = None if gisn else genotype[1:]
            # Get scaled y sum
            y_unshifted = None if gisn else np.sum(y_pos_terms) // y_div    
            # E.g. min = (-1 * 128)//32 = 4; max = (1 * 128)//32 = 34

            # For e.g. size = 11 for, we will get values between -4 and 4, so
            # we shift by 5 to get values between 1 and 9 (but do for
            # general case)
            y_shift = (size - 2 - 1) // 2 + 1
            # Get final y position
            y_pos = None if gisn else y_unshifted + y_shift

            # Bounds
            genotype_lower_bounds = np.array(
                [1]        + [-1] * num_y_pos_terms)
            genotype_upper_bounds = np.array(
                [size - 2] + [+1] * num_y_pos_terms)
            
        elif gt_type == 'natural':
            # Genotype is just direct goal location
            # Length is [ 1      +      1 ] 
            #           [  x pos + y pos  ]
            genotype_size = 2
            x_pos = None if gisn else genotype[0]
            y_pos = None if gisn else genotype[1]

            # Bounds
            genotype_lower_bounds = np.array([1, 1])
            genotype_upper_bounds = np.array([size - 2, size - 2])
        else:
            raise NotImplementedError(
                'Unknown genotype type: {}'.format(gt_type))

        goal_pos = None if gisn else (y_pos, x_pos)

        genotype_bounds = [(l, u) for l, u in
                            zip(list(genotype_lower_bounds),
                                list(genotype_upper_bounds))]

        # Return processed genotype
        processed_genotype = {
            'grid_size': size,
            'start_pos': (size // 2, size // 2),
            'goal_pos': goal_pos,
            'genotype': genotype,
            'genotype_bounds': genotype_bounds,
            'genotype_size': genotype_size,
            'genotype_lower_bounds': genotype_lower_bounds,
            'genotype_upper_bounds': genotype_upper_bounds,
        }
        
        return Namespace(**processed_genotype)
    
    @staticmethod
    def is_valid_genotype(processed_genotype, gt_type=None):
        """Check if genotype is valid"""
        del gt_type
        pg = processed_genotype
        # Check if genotype within bounds
        if pg.genotype is None:
            return False, 'none_genotype'
        if np.any(pg.genotype < pg.genotype_lower_bounds):
            return False, 'lower_bound_violation'
        if np.any(pg.genotype > pg.genotype_upper_bounds):
            return False, 'upper_bound_violation'
        # Check goal does not overlap with agent start position
        # Start position is at center of maze (size // 2, size // 2)
        start_pos = (pg.grid_size // 2, pg.grid_size // 2)
        if pg.goal_pos == start_pos:
            return False, 'goal_start_overlap'
        
        return True, None
    
    def set_genotype_from_current_grid(self):
        """Set genotype from current grid.
        
        This method is called after a level is generated by a generator
        (i.e. not a genotype), and we want to set the genotype from the current
        grid information.

        Returns:
            Namespace: Processed genotype.
        """

        # print('\nSet genotype from current grid')
        if self.gt_type[0] == 'a':
            # There is not a deterministic mapping from grid to genotype, so
            # we just set values to one possible genotype
            # print('goal_pos: {}'.format(self.goal_pos))
            # Goal pos of form (y, x); that is (i, j)
            num_y_pos_terms = int(self.gt_type[1:])

            x_pos = self.goal_pos[1]
            y_pos = self.goal_pos[0]
            
            # From y_pos, we first want to compute shifted y_pos, which is
            # y_pos - y_shift
            y_shift = (self.size - 2 - 1) // 2 + 1
            # print('y_shift: {}'.format(y_shift))
            y_unshifted = y_pos - y_shift
            # print('y_unshifted: {}'.format(y_unshifted))
            # Now, we we set the magnitude of y_unshifted number of y_pos_terms
            # to the sign of y_unshifted
            y_pos_terms = np.zeros(num_y_pos_terms)
            num_terms = np.abs(y_unshifted) * 4
            y_pos_terms[:num_terms] = np.sign(y_unshifted)
            # print('y_pos_terms: {}'.format(y_pos_terms))
            genotype = np.concatenate(([x_pos], y_pos_terms))
            # print('genotype: {}'.format(genotype))

        elif self.gt_type == 'natural':
            # goal_pos of form (y, x); that is (i, j)
            genotype = np.array([self.goal_pos[1], self.goal_pos[0]])
        else:
            raise NotImplementedError(
                'Unknown genotype type: {}'.format(self.gt_type))
        
        self.genotype = genotype.astype(int)

        return self.process_genotype_new(genotype)

    def generate_level_from_genotype(self, genotype, gt_type='a16'):
        """Generate level from genotype, which is a sequence of ints"""
        if genotype is not None:
            genotype = np.array(genotype).astype(int)
            self.genotype = genotype
        
        if genotype is None and self.genotype_set:
            self._gen_grid(self.size, self.size)
            return

        # Process genotype
        pg = self.process_genotype(
            genotype, self.size, gt_type=gt_type)
        
        # Set common variables
        self.genotype_size = pg.genotype_size
        self.genotype_lower_bounds = pg.genotype_lower_bounds
        self.genotype_upper_bounds = pg.genotype_upper_bounds
        self.genotype_bounds = pg.genotype_bounds
        self.genotype = pg.genotype
        self.start_pos = pg.start_pos
        self.goal_pos = pg.goal_pos

        # Indicate that genotype is set
        if genotype is None:
            # NOTE: This is important because we might pass in a None genotype
            # here after we've already set one
            self.genotype_set = False
        else:
            self.genotype_set = True
            self._gen_grid(self.size, self.size)

    def generate_level_from_seed(self, seed=None):
        """
        We assume self.rng_seed is already set from __init__, so no need to
        pass in seed here, unless we're changing the task. We assume self.seed
        has already been called.
        """
        if seed is not None:
            self.seed(seed=seed)

        prevs = []
        self.start_pos = np.array((self.size // 2, self.size // 2))
        prevs.append(tuple(self.start_pos))

        # Set goal location
        if self._init_goal_pos is None:
            # Sample goal location
            # goal_pos is of form (x, y); that is (j, i)
            self.goal_pos, _ = sample_grid_location(
                size=self.size,
                region=self.goal_sampler_region,
                sample_mode=self.goal_sampler,
                border_size=1,
                seed=self.rng_seed,
                prevs=prevs
            )
            self.goal_pos = np.array(self.goal_pos)
        else:
            self.goal_pos = np.array(self.goal_pos)


    def gen_obs(self):
        """Add goal loc to observations"""
        # Observation is just agent current position plus its direction as an int
        pos = np.array(self.agent_pos[0])
        dr =  self.agent_dir[0]
        obs = np.concatenate((pos, [dr]))
        return obs
    
    def _reward(self):
        """
        Fixing the reward to make it markovian... commented out version is not
        """
        # return 1 - 0.9 * (self.step_count / self.max_steps)
        return 1

    def agent_is_done(self, agent_id):
        """
        Overwriting MultiGridEnv functionality so that we can choose if
        episode ends when agent reaches a goal state.
        """
        # If we want all trajectories to be the same length, we do not allow
        # ending the episode early
        if not self.variable_trial_lengths:
            self.success = True
            return
        else:
            # Otherwise, we use the parent method
            self.success = True
            super().agent_is_done(agent_id)

    def reset(self):
        """ Copied from MultiGridEnv.reset() and modified for QD """
        if self.distribution_type == 'QD' and not self.genotype_set:
            # NOTE: If we're using QD, genotype needs to be set before we
            # generate the grid, etc. For now, return placeholder obs by
            # generating maze from seed
            return None
        
        if self.fixed_environment:
            self.seed(self.seed_value)

        # Current position and direction of the agent
        self.agent_pos = [None] * self.n_agents
        self.agent_dir = [None] * self.n_agents
        self.done = [False] * self.n_agents

        # Generate the grid. Will be random by default, or same environment if
        # 'fixed_environment' is True.
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        for a in range(self.n_agents):
            assert self.agent_pos[a] is not None
            assert self.agent_dir[a] is not None

            # Check that the agent doesn't overlap with an object
            start_cell = self.grid.get(*self.agent_pos[a])
            assert (start_cell.type == "agent" or start_cell is None or
                    start_cell.can_overlap()), \
                   "Invalid starting position for agent: {}\n".format(start_cell) + \
                   "Agent pos: {}\n".format(self.agent_pos[a]) + \
                   "Agent dir: {}\n".format(self.agent_dir[a]) + \
                   "Genotype: {}\n".format(None if not self.distribution_type 
                                         == 'QD' else self.genotype) + \
                   "Is valid genotype: {}, {}\n".format(*self.is_valid_genotype(
                        self.process_genotype(self.genotype, self.size, self.gt_type),
                        self.gt_type))

        # Item picked up, being carried, initially nothing
        self.carrying = [None] * self.n_agents

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        obs = self.gen_obs()

        self.success = False

        return obs

    def reset_task(self, task=None) -> None:
        """
        Reset current task (i.e. seed, genotype, etc.).

        Reset the task, either at random (if task=None) or the given task.
        Should *not* reset the environment. Returns the coordinates of a new
        goal state. 
        TODO: Should the task be more than just the goal state, 
        since we have walls (unlike the VariBAD gridworld)?
        """
        # Generate level from specifications
        if self.distribution_type == 'SB':
            # If we're using a seed-based distribution, we need to generate
            # a new seed and then generate the level from that seed
            self.seed(task)
            self.generate_level_from_seed(seed=task)
        elif self.distribution_type == 'QD':
            # Convert genotype to all int array
            self.generate_level_from_genotype(genotype=task, gt_type=self.gt_type)
        else:
            raise ValueError(
                f'Unknown distribution type: {self.distribution_type}')
        self._gen_grid(self.width, self.height)

        if self.distribution_type == 'SB':
            # Set genotype from current task (generated from seed)
            _ = self.set_genotype_from_current_grid()

        if self.distribution_type == 'QD':
            return self.genotype
        else:
            # TODO: Shouldn't we always return genotype, either way?
            return self.goal_pos
    
    def get_task(self):
        """ Return the ground truth task. """
        # TODO: more thoughtful implementation
        if hasattr(self, 'genotype') and self.genotype is not None:
            return np.asarray(self.genotype).copy()
        else:
            return np.array((0.0,))

    def task_to_id(self, goals):
        """
        MazeEnv can be enumerated as easily as VariBAD's gridworld environment, 
        so instead of using a separate head for each state in reward prediction,
        we pass it in as input. Thus, we do not need this function (I think).
        """
        raise NotImplementedError
    
    def id_to_task(self, classes):
        """ Undefined for same reason as `task_to_id` (see docstring). """
        raise NotImplementedError

    def goal_to_onehot_id(self, pos):
        """ Undefined for same reason as `task_to_id` (see docstring). """
        raise NotImplementedError

    def onehot_id_to_goal(self, pos):
        """ Undefined for same reason as `task_to_id` (see docstring). """
        raise NotImplementedError
    
    def _reset_belief(self) -> np.ndarray:
        raise NotImplementedError('Oracle not implemented for MazeEnv.')

    def update_belief(self, state, action) -> np.ndarray:
        raise NotImplementedError('Oracle not implemented for MazeEnv.')

    def get_belief(self):
        raise NotImplementedError('Oracle not implemented for MazeEnv.')

    @property
    def level_rendering(self):
        """Render high-level view of level"""
        return self.render(mode='human')
    
    @staticmethod
    def visualise_behaviour(env,
                            args,
                            policy,
                            iter_idx,
                            encoder=None,
                            reward_decoder=None,
                            image_folder=None,
                            **kwargs
                            ):
        """
        Visualises the behaviour of the policy, together with the latent state 
        and belief. The environment passed to this method should be a 
        SubProcVec or DummyVecEnv, not the raw env!
        """

        num_episodes = args.trials_per_episode
        unwrapped_env = env.venv.unwrapped.envs[0]

        # --- initialise things we want to keep track of ---

        episode_returns = []
        episode_lengths = []

        if args.pass_belief_to_policy and (encoder is None):
            episode_beliefs = [[] for _ in range(num_episodes)]
        else:
            episode_beliefs = None

        if encoder is not None:
            # keep track of latent spaces
            episode_latent_samples = [[] for _ in range(num_episodes)]
            episode_latent_means = [[] for _ in range(num_episodes)]
            episode_latent_logvars = [[] for _ in range(num_episodes)]
        else:
            episode_latent_samples = episode_latent_means = \
                episode_latent_logvars = None

        curr_latent_sample = curr_latent_mean = curr_latent_logvar = None

        # --- roll out policy ---

        env.reset_task()
        [state, belief, task, level_seeds] = utl.reset_env(env, args)
        start_obs = dict()
        for k, v in state.items():
            start_obs[k] = v.clone()

        episode_img_obs = [[] for _ in range(num_episodes)]
        episode_all_obs = {k: [[] for _ in range(num_episodes)] for k in state.keys()}
        episode_prev_obs = {k: [[] for _ in range(num_episodes)] for k in state.keys()}
        episode_next_obs = {k: [[] for _ in range(num_episodes)] for k in state.keys()}
        episode_actions = [[] for _ in range(num_episodes)]
        episode_rewards = [[] for _ in range(num_episodes)]

        for episode_idx in range(args.trials_per_episode):
            curr_goal = env.get_task()
            curr_rollout_rew = []
            curr_rollout_goal = []

            # Get first image observation
            img_obs = unwrapped_env.render()
            episode_img_obs[episode_idx].append(img_obs)

            if encoder is not None:

                if episode_idx == 0:
                    # reset to prior
                    curr_latent_sample, curr_latent_mean, curr_latent_logvar, \
                        hidden_state = encoder.prior(1)
                    curr_latent_sample = curr_latent_sample[0].to(DeviceConfig.DEVICE)
                    curr_latent_mean = curr_latent_mean[0].to(DeviceConfig.DEVICE)
                    curr_latent_logvar = curr_latent_logvar[0].to(DeviceConfig.DEVICE)

                episode_latent_samples[episode_idx].append(
                    curr_latent_sample[0].clone())
                episode_latent_means[episode_idx].append(
                    curr_latent_mean[0].clone())
                episode_latent_logvars[episode_idx].append(
                    curr_latent_logvar[0].clone())

            for k, v in start_obs.items():
                episode_all_obs[k][episode_idx].append(v.clone())
            if args.pass_belief_to_policy and (encoder is None):
                episode_beliefs[episode_idx].append(belief)

            for step_idx in range(1, env._max_trial_steps + 1):

                if step_idx == 1:
                    for k, v in start_obs.items():
                        episode_prev_obs[k][episode_idx].append(v.clone())
                else:
                    for k, v in state.items():
                        episode_prev_obs[k][episode_idx].append(v.clone())
                
                state_view = dict()
                img_dims = len(state['image'].shape)
                for k, v in state.items():
                    if k == 'image':
                        state_view[k] = v.view(v.shape[img_dims-3:])
                    else:
                        state_view[k] = v.view(-1)

                # act
                _, action = utl.select_action(
                    args=args,
                    policy=policy,
                    state=state_view,
                    belief=belief,
                    task=task,
                    deterministic=True,
                    latent_sample=curr_latent_sample.view(-1) if (
                        curr_latent_sample is not None) else None,
                    latent_mean=curr_latent_mean.view(-1) if (
                        curr_latent_mean is not None) else None,
                    latent_logvar=curr_latent_logvar.view(-1) if (
                        curr_latent_logvar is not None) else None,
                )

                # observe reward and next obs
                [state, belief, task], rewards, done, infos \
                    = utl.env_step(env, action, args)
                img_obs = unwrapped_env.render()
                episode_img_obs[episode_idx].append(img_obs)

                if len(rewards) == 2:
                    # Using vector norm wrapper on rewards
                    rew_raw, rew_normalised = rewards
                else: 
                    # Not using vector norm wrapper on rewards
                    rew_raw = rewards

                if encoder is not None:
                    # update task embedding
                    curr_latent_sample, curr_latent_mean, curr_latent_logvar, \
                    hidden_state = \
                        encoder(
                            action.float().to(DeviceConfig.DEVICE),
                            state,
                            rew_raw.reshape((1, 1)).float().to(DeviceConfig.DEVICE),
                            hidden_state,
                            return_prior=False
                        )

                    episode_latent_samples[episode_idx].append(
                        curr_latent_sample[0].clone())
                    episode_latent_means[episode_idx].append(
                        curr_latent_mean[0].clone())
                    episode_latent_logvars[episode_idx].append(
                        curr_latent_logvar[0].clone())

                for k, v in state.items():
                    episode_all_obs[k][episode_idx].append(v.clone())
                    episode_next_obs[k][episode_idx].append(v.clone())
                episode_rewards[episode_idx].append(rew_raw.clone())
                episode_actions[episode_idx].append(action.clone())

                curr_rollout_rew.append(rew_raw.clone())
                curr_rollout_goal.append(env.get_task().copy())

                if args.pass_belief_to_policy and (encoder is None):
                    episode_beliefs[episode_idx].append(belief)

                if infos[0]['done_mdp'] and not done:
                    start_obs = infos[0]['start_state']
                    for k, v in start_obs.items():
                        start_obs[k] = torch.from_numpy(np.array(v)).float().reshape((1, *np.array(v).shape)).to(DeviceConfig.DEVICE)
                    break

            episode_returns.append(sum(curr_rollout_rew))
            episode_lengths.append(step_idx)

        # Process episode data for return
        if encoder is not None:
            episode_latent_means = [
                torch.stack(e) for e in episode_latent_means]
            episode_latent_logvars = [
                torch.stack(e) for e in episode_latent_logvars]
        episode_prev_obs = {k: [torch.cat(e) for e in v] for k, v in episode_prev_obs.items()}
        episode_next_obs = {k: [torch.cat(e) for e in v] for k, v in episode_next_obs.items()}
        episode_actions = [torch.cat(e) for e in episode_actions]
        episode_rewards = [torch.cat(e) for e in episode_rewards]

        # Plot the behavior in the environment
        rew_pred_means, rew_pred_vars = plot_behavior(
            env, args, episode_all_obs, episode_img_obs, reward_decoder,
            episode_latent_means, episode_latent_logvars,
            image_folder, iter_idx, episode_beliefs)

        return episode_latent_means, episode_latent_logvars, \
               episode_prev_obs, episode_next_obs, episode_actions, \
               episode_rewards, episode_returns



def plot_behavior(env, args, episode_all_obs, episode_img_obs, reward_decoder,
            episode_latent_means, episode_latent_logvars, image_folder, 
            iter_idx, episode_beliefs):
    """
    Plot episode image observations.
    """


    plt.figure(figsize=(1.5 * env._max_trial_steps, 
                        1.5 * args.trials_per_episode))

    if isinstance(episode_all_obs, dict):
        num_episodes = len(episode_all_obs[list(episode_all_obs.keys())[0]])
        num_steps = len(episode_all_obs[list(episode_all_obs.keys())[0]][0])
    else:
        num_episodes = len(episode_all_obs)
        num_steps = len(episode_all_obs[0])
    

    rew_pred_means = [[] for _ in range(num_episodes)]
    rew_pred_vars = [[] for _ in range(num_episodes)]

    for k, v in episode_all_obs.items():
        print('k', k, 'v', v[0][0].shape, v[0][1].shape, v[0][-1].shape)
        print('len(v)', len(v))

    print('num_episodes', num_episodes)

    # loop through the experiences
    for episode_idx in range(num_episodes):
        for step_idx in range(num_steps):
            curr_obs = {k: v[episode_idx][:step_idx + 1] for k, v in episode_all_obs.items()}
            curr_goal = episode_goals[episode_idx]

            if episode_latent_means is not None:
                curr_means = episode_latent_means[episode_idx][:step_idx + 1]
                curr_logvars = \
                    episode_latent_logvars[episode_idx][:step_idx + 1]

            # choose correct subplot
            plt.subplot(args.trials_per_episode,
                        math.ceil(env._max_trial_steps) + 1,
                        1 + episode_idx * \
                        (1 + math.ceil(env._max_trial_steps)) + step_idx),

            # plot the behaviour
            plot_observations(env, curr_obs, curr_goal)

            # TODO: If time, we can bring back functionality to plot the
            #       beliefs (see gridworld code).
            rew_pred_means = rew_pred_vars = None

            if episode_idx == 0:
                plt.title('t = {}'.format(step_idx))

            if step_idx == 0:
                plt.ylabel('Episode {}'.format(episode_idx + 1))

    # TODO: see above
    # if reward_decoder is not None:
    #     rew_pred_means = [torch.stack(r) for r in rew_pred_means]
    #     rew_pred_vars = [torch.stack(r) for r in rew_pred_vars]

    # save figure that shows policy behaviour
    plt.tight_layout()
    if image_folder is not None:
        plt.savefig('{}/{}_behaviour'.format(image_folder, iter_idx))
        plt.close()
    else:
        plt.show()

    return None, None
    # TODO: see above
    # return rew_pred_means, rew_pred_vars


def plot_observations(env, observations, goal):
    num_cells = int(env.observation_space.high[0] + 1)

    # draw grid
    for i in range(num_cells):
        for j in range(num_cells):
            pos_i = i
            pos_j = j
            rec = Rectangle((pos_i, pos_j), 1, 1, facecolor='none', alpha=0.5,
                            edgecolor='k')
            plt.gca().add_patch(rec)

    # shift obs and goal by half a stepsize
    # TODO: Finish implementing this
    # if isinstance(observations, tuple) or isinstance(observations, list):
    #     observations = torch.cat(observations)
    # elif isinstance(observations, dict):
    #     if observation
    observations = observations.cpu().numpy() + 0.5
    goal = np.array(goal) + 0.5

    # visualise behaviour, current position, goal
    plt.plot(observations[:, 0], observations[:, 1], 'b-')
    plt.plot(observations[-1, 0], observations[-1, 1], 'b.')
    plt.plot(goal[0], goal[1], 'kx')

    # make it look nice
    plt.xticks([])
    plt.yticks([])
    plt.xlim([0, num_cells])
    plt.ylim([0, num_cells])

