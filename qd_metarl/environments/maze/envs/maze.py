# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Code from https://github.com/ucl-dark/paired.

Implements single-agent manually generated Maze environments.

Humans provide a bit map to describe the position of walls, the starting
location of the agent, and the goal location.
"""

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
from qd_metarl.environments.maze.envs.maze_generation import CustomMazeGenerator
from qd_metarl.environments.maze.level import OBJ_TYPES_TO_INT
from qd_metarl.utils import env_utils as utl
from qd_metarl.utils.torch_utils import DeviceConfig
from qd_metarl.qd.measures.maze_measures import MazeMeasures

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


logger = logging.getLogger(__name__)


DEFAULT_BITMAPS = [
    np.array([
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1],
        [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
        [1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    ])
]


def sample_maze_location(
        size=None,
        region=None,
        sample_mode='uniform',
        border_size=0,
        seed=None,
        bit_map=None,
        prng=None, 
        prevs=[],
):
    """
    Sample maze location given various specifications and constraints.

    NOTE: Either size or region must be provided.

    Args:
        size: Number of cells in (presumed square) maze.
        region: Region within maze to sample from (all inclusive). The region
            is defined by (i_1, j_1, i_2, j_2), where i is top left point and
            j is bottom right.
        sample_mode: Method for sampling points within region.
        border_size: How many cells inwards to exclude from the boundary.
        seed: Seed used for RNG.
        bit_map: Wall locations---if not None, we keep resampling until we find
        a location that does not exist along a wall.
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
            if bit_map is not None and bit_map[i][j] == 1 or (i, j) in prevs:
                continue
            else:
                return (i, j), prng
    elif sample_mode == 'edges':
        # TODO
        raise NotImplementedError
    else:
        raise NotImplementedError(f'Unknown sample mode: {sample_mode}')


class MazeEnv(MultiGridEnv):
    """Single-agent maze environment specified via a bit map."""

    def __init__(
        self,                       #
        size:                       int = 11,
        agent_view_size:            int = 5,
        minigrid_mode:              bool = True,
        max_steps:                  int = None,
        bit_map:                    Any = None,
        use_info_cell:              bool = False,
        reward_type:                str = 'sparse',
        initial_goal_visibility:    str = 'visible',
        start_pos:                  Tuple[int] = None,
        goal_pos:                   Tuple[int] = None,
        info_pos:                   Tuple[int] = None,
        start_sampler:              str = 'uniform',
        goal_sampler:               str = 'uniform',
        info_sampler:               str = 'uniform',
        start_sampler_region:       Tuple[int] = None,
        goal_sampler_region:        Tuple[int] = None,
        info_sampler_region:        Tuple[int] = None,
        variable_trial_lengths:          bool = False,
        default_bitmap_index:       int = None,
        seed:                       int = None,
        distribution_type:          str = 'SB',
        gt_type:                    str = 'BM+SGL',
        visualize:                  bool = False,
        dense_rewards:              bool = False,
    ):
        """ Maze environment initialization.
        
        Args:
        - size (int): Size of maze (odd number).
        - agent_view_size (int): Size of agent's view.
        - minigrid_mode (bool): Whether to use minigrid mode (not multigrid).
        - max_steps (int): Maximum number of steps in episode.
        - bit_map (Any): Bit map of maze. If None, we generate a random maze.
        - use_info_cell (bool): Whether to use an info cell.
        - reward_type (str): Type of reward. Options are:
            - 'sparse': 1 if agent reaches goal, 0 otherwise.
            - 'dense': TODO (proximity-based reward)
        - initial_goal_visibility (str): Whether goal is visible at start.
          Options are:
            - 'visible': Goal is visible at start.
            - 'invisible': Goal is not visible at start.
        - start_pos (Tuple[int]): Starting position of agent.
        - goal_pos (Tuple[int]): Goal position of agent.
        - info_pos (Tuple[int]): Info position of agent.
        - start_sampler (str): How to sample start position. Options are:
            - 'uniform': Uniformly sample from all locations.
            - 'edges': Sample from edges of maze.
        - goal_sampler (str): How to sample goal position. Options are the same
            as start_sampler.
        - info_sampler (str): How to sample info position. Options are the same
            as start_sampler.
        - start_sampler_region (Tuple[int]): Region to sample start position
            from. If None, sample from entire maze.
        - goal_sampler_region (Tuple[int]): Region to sample goal position
            from. If None, sample from entire maze.
        - info_sampler_region (Tuple[int]): Region to sample info position 
            from. If None, sample from entire maze.
        - variable_trial_lengths (bool): If we don't care about trajectories being the
          same length, we can set this to True. If False, when agent 
          reaches goal or dies, we keep episode running.
        - default_bitmap_index (int): Index of default bitmap to use. If None,
            we generate a random maze.
        - seed (int): Seed for RNG.
        - distribution_type: How to generate maze. Options are:
            - 'SB': Use seeded backtracking algorithm
                        and location specifications to generate maze.
            - 'QD': Use QD genotype to generate maze.
        - gt_type: How to interpret genotype. Options are:
            - 'BM+SGL': Genotype expressed as flattened bitmap + start/goal locations
            - 'BM+SGIL': Genotype expressed as flattened bitmap + start/goal/info locations
        - visualize (bool): Whether to visualize environment.
        - dense_rewards (bool): Whether to use dense rewards.
        """
        del visualize  # Unused for this environmnent
        self.size = size
        self.use_info_cell = use_info_cell
        self.reward_type = reward_type
        self.default_bitmap_index = default_bitmap_index
        self.rng_seed = seed
        self.distribution_type = distribution_type
        self.gt_type = gt_type
        self.dense_rewards = dense_rewards

        self.start_pos = start_pos
        self._init_start_pos = start_pos
        self.goal_pos = goal_pos
        self._init_goal_pos = goal_pos
        self.info_pos = info_pos
        self._init_info_pos = info_pos
        self.start_sampler = start_sampler
        self.goal_sampler = goal_sampler
        self.info_sampler = info_sampler
        self.start_sampler_region = start_sampler_region
        self.goal_sampler_region = goal_sampler_region
        self.info_sampler_region = info_sampler_region

        if type(self.start_sampler_region) is str:
            if self.start_sampler_region == 'top-left':
                self.start_sampler_region = (0, 0, size//2, size//2)
            elif self.start_sampler_region == 'bottom-right':
                self.start_sampler_region = (size//2, size//2, size-1, size-1)
            elif self.start_sampler_region == 'top-right':
                self.start_sampler_region = (0, size//2, size//2, size-1)
            elif self.start_sampler_region == 'bottom-left':
                self.start_sampler_region = (size//2, 0, size-1, size//2)
            else:
                raise NotImplementedError

        if type(self.goal_sampler_region) is str:
            if self.goal_sampler_region == 'top-left':
                self.goal_sampler_region = (0, 0, size//2, size//2)
            elif self.goal_sampler_region == 'bottom-right':
                self.goal_sampler_region = (size//2, size//2, size-1, size-1)
            elif self.goal_sampler_region == 'top-right':
                self.goal_sampler_region = (0, size//2, size//2, size-1)
            elif self.goal_sampler_region == 'bottom-left':
                self.goal_sampler_region = (size//2, 0, size-1, size//2)
            else:
                raise NotImplementedError
        
        if type(self.info_sampler_region) is str:
            raise NotImplementedError
        
        # If using info cell, should not be able to see goal at start
        if self.use_info_cell:
            self.goal_visibility = False
        else:
            self.goal_visibility = initial_goal_visibility

        # If bitmap provided, ensure it is of correct format
        self.maze_generator = None
        if default_bitmap_index is not None:
            bit_map = DEFAULT_BITMAPS[default_bitmap_index]
            self.new_bitmap_fn = lambda: bit_map
        else:
            if bit_map is not None:
                # If bit_map provided, use it!
                bit_map = np.array(bit_map)
                assert bit_map.shape == (size - 2, size - 2)
                self.new_bitmap_fn = lambda: bit_map
            else:
                # Otherwise, generate random maze from distribution
                assert size % 2 == 1  # We assume size is odd
                self.maze_generator = CustomMazeGenerator(
                    w=size//2, h=size//2, seed=self.rng_seed)
                bit_map = self.maze_generator.generate_bitmap()
        self.bit_map = bit_map
        self.bit_map_padded = np.pad(bit_map, pad_width=1, mode='constant', constant_values=1)
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
            agent_view_size=agent_view_size,
            max_steps=max_steps,
            see_through_walls=True,  # Set this to True for maximum speed
            minigrid_mode=minigrid_mode,
            seed=seed
        )

        self.grid_size = size

        # Instead of pure bool, we use ints
        self.bool_obs_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(1,),  # hardcode as 1 since n_agents=1
            dtype="uint8")
        
        self.coords_obs_space = gym.spaces.Box(
            low=0,
            high=self.grid_size,
            shape=(2,),  # also hardcoded
            dtype="uint8")

        self.observation_space = gym.spaces.Dict({
            "image": self.image_obs_space,
            "direction": self.direction_obs_space,
            "goal_known": self.bool_obs_space, 
            "info_known": self.bool_obs_space,
            "goal_loc": self.coords_obs_space,
            "info_loc": self.coords_obs_space,
            "x": self.pos_obs_space,
            "y": self.pos_obs_space,
        })

        # For QD+MetaRL
        self.genotype = None
        self.set_genotype_info()
        self.success = False

    def step(self, action):
        """ Step function. """

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

        return obs, reward, done, info
        
    def seed(self, seed=None):
        """ Set seed. """
        if seed is not None: 
            seed = int(seed)
        super().seed(seed=seed)
        self.rng_seed = seed
        # If we're using a maze_generator, we set its seed as well
        if self.maze_generator is not None:
            self.maze_generator.reset_seed(seed)

    def _gen_grid(self, width, height):
        """Generate grid from self.bit_map, and start/goal/info locations."""
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
        # Info 
        if self.use_info_cell:
            raise NotImplementedError
        # Walls
        for x in range(self.bit_map.shape[0]):
            for y in range(self.bit_map.shape[1]):
                if self.bit_map[y, x]:
                    # Add an offset of 1 for the outer walls
                    self.put_obj(minigrid.Wall(), x + 1, y + 1)

    @staticmethod
    def _get_adj(level):
        """
        Converts the level into an adjacency matrix that can be used by scipy's
        graph methods.

        Args:
            level: Array with ints corresponding to elements in the grid

        Returns:
            2D Array with the shortest distances between each cell
                (np.inf if it is a wall or if there is no path)
        """
        n_cells = level.size
        adj = np.zeros((n_cells, n_cells))

        # Set edges to distance 1
        for i in range(level.shape[0]):
            for j in range(level.shape[1]):
                if level[i, j] == OBJ_TYPES_TO_INT[" "]:  # Empty
                    neighbors = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
                    valid_neighbors = [(i, j)]
                    for n in neighbors:
                        if 0 <= n[0] < level.shape[0] and 0 <= n[
                                1] < level.shape[1]:
                            if level[n] == OBJ_TYPES_TO_INT[" "]:  # Empty
                                valid_neighbors.append(n)

                    # Get flattened idx
                    neighbor_idx = np.ravel_multi_index(
                        list(zip(*valid_neighbors)), level.shape)
                    cell_idx = neighbor_idx[0]
                    if len(valid_neighbors) >= 2:  # At least one neighbor
                        adj[cell_idx, neighbor_idx[1:]] = 1
        return adj
    
    @staticmethod
    def is_reachable(bit_map_padded, start, target):
        """
        Check if target is reachable from the starting location given
        the bitmap. Reachability is computed using breadth-first search.

        Args:
            bit_map_padded (np.ndarray): Padded bitmap
            start (tuple): Starting location (x, y)
            target (tuple): Target location (x, y)

        Returns:
            bool: True if target is reachable from start, False otherwise
        """
        start = (start[1], start[0])  # Convert (x, y) to (row, column)
        target = (target[1], target[0])  # Convert (x, y) to (row, column)
        queue = deque([start])
        visited = set()

        while queue:
            curr = queue.popleft()
            # print(f"Popped: {curr}")
            if curr == target:
                return True
            if curr in visited:
                continue
            visited.add(curr)

            # Define neighbors in (row, column) format
            neighbors = (
                (curr[0]-1, curr[1]), (curr[0]+1, curr[1]),  # Up, Down
                (curr[0], curr[1]-1), (curr[0], curr[1]+1)   # Left, Right
            )
            
            for neighbor in neighbors:
                if (0 <= neighbor[0] < bit_map_padded.shape[0] and
                        0 <= neighbor[1] < bit_map_padded.shape[1] and
                        bit_map_padded[neighbor] == 0 and
                        neighbor not in visited):
                    queue.append(neighbor)
        return False
    
    def set_genotype_info(self):
        """ Extract information from genotype and genotype type. """

        self.measures = MazeMeasures.get_all_measures()

        if self.gt_type == 'BM+SGL':
            # Genotype expressed as flattened bitmap + start/goal locations
            # Length is [(size-2) x (size-2) + (2 x 2)] 
            #           [  bitmap    +    start/goal  ]
            self.genotype_n_extra_spots = 4
        elif self.gt_type == 'BM+SGIL':
            # Genotype expressed as flattened bitmap + start/goal/info locations
            # Length is [(size-2) x (size-2) + (3 x 2)] 
            #           [  bitmap +  start/goal/info  ]
            self.genotype_n_extra_spots = 6
            self.measures.append(
                'info_avg_reachability' # Avg reachability between info and all blank cells
                )
        else:
            raise ValueError('Unknown genotype type: {}'.format(self.gt_type))
        
        self.genotype_size = (self.size - 2)**2 + self.genotype_n_extra_spots
        self.genotype_lower_bounds = np.array(
            [0] * (self.size - 2)**2 + [1] * self.genotype_n_extra_spots)
        self.genotype_upper_bounds = np.array(
            [1] * (self.size - 2)**2 + [self.size - 2] * self.genotype_n_extra_spots)
        self.genotype_bounds = [(l, u) for l, u in
                                zip(list(self.genotype_lower_bounds), 
                                    list(self.genotype_upper_bounds))]

    def pos_to_flat_pos(self, pos):
        """ Convert position tuple to flattened index. """
        return (pos[0]-1) * (self.size - 2) + (pos[1]-1)
    
    def get_measures_info(self, env_name):
        """ Get info about measures. """
        return MazeMeasures.get_measures_info(env_name)

    def compute_measures(self, genotype=None, measures=None):
        """" Compute the measures for the given maze. """
        # Use current maze measures and genotype if none provided
        if measures is None:
            measures = self.measures
        if genotype is None:
            genotype = self.genotype

        # Extract useful properties of genotype
        pg = self.process_genotype_new(genotype)
        bit_map = pg.bit_map
        start_pos = self.pos_to_flat_pos(pg.start_pos)
        goal_pos = self.pos_to_flat_pos(pg.goal_pos)
        if hasattr(pg, 'info_pos'):
            info_pos = self.pos_to_flat_pos(pg.info_pos)
        else: 
            info_pos = None

        # Get graph info
        adj = self._get_adj(bit_map)  # Get adjacency matrix
        distances, _ = csgraph.floyd_warshall(adj, return_predecessors=True)

        # Compute measures
        return MazeMeasures.compute_measures(
            bit_map=bit_map,
            start_pos = pg.start_pos,
            goal_pos = pg.goal_pos,
            flat_start_pos=start_pos,
            flat_goal_pos=goal_pos,
            flat_info_pos=info_pos,
            adj=adj,
            distances=distances,
            measures=measures
        )

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

    def process_genotype_new(self, genotype):
        """Extract information from genotype and genotype type."""
        gisn = genotype is None

        assert gisn or len(genotype) == self.genotype_size, \
            f'Genotype length {len(genotype)} != {self.genotype_size}'
            
        # Extract bitmap and start/goal/info locations
        bit_map = None if gisn else np.array(genotype[:((self.size - 2)**2)])
        bit_map = None if gisn else bit_map.reshape(self.size - 2, self.size - 2)
        bit_map_padded = np.pad(bit_map, pad_width=1, mode='constant', constant_values=1)
        
        if self.gt_type == 'BM+SGL':
            start_pos = None if gisn else np.array(genotype[-4:-2]).astype(int)
            goal_pos = None if gisn else np.array(genotype[-2:]).astype(int)
        elif self.gt_type == 'BM+SGIL':
            start_pos = None if gisn else np.array(genotype[-6:-4]).astype(int)
            goal_pos = None if gisn else np.array(genotype[-4:-2]).astype(int)
            info_pos = None if gisn else np.array(genotype[-2:]).astype(int)

        # Return processed genotype
        processed_genotype = {
            'bit_map': bit_map,
            'bit_map_padded': bit_map_padded,
            'start_pos': start_pos,
            'goal_pos': goal_pos,
            'genotype': genotype,
            'genotype_bounds': self.genotype_bounds,
            'genotype_size': self.genotype_size,
            'genotype_lower_bounds': self.genotype_lower_bounds,
            'genotype_upper_bounds': self.genotype_upper_bounds,
        }

        if self.gt_type == 'MB+SGIL':
            processed_genotype['info_pos'] = info_pos
        
        return Namespace(**processed_genotype)
    
    def set_genotype_from_current_grid(self):
        """Set genotype from current grid.
        
        This method is called after a level is generated by a generator
        (i.e. not a genotype), and we want to set the genotype from the current
        grid information.

        Returns:
            Namespace: Processed genotype.
        """
        if self.gt_type == 'BM+SGL':
            # Genotype expressed as flattened bitmap + start/goal locations
            # Length is [(size-2) x (size-2) + (2 x 2)] 
            #           [  bitmap    +    start/goal  ]
            assert not hasattr(self, 'info_pos') or self.info_pos is None, \
                'Invalid gt_type for current maze: info_pos is set'
            genotype_size = (self.size - 2)**2 + 4
            genotype = np.zeros(genotype_size)
            genotype[:((self.size - 2)**2)] = self.bit_map.flatten()
            genotype[-4:-2] = self.start_pos
            genotype[-2:] = self.goal_pos            
        elif self.gt_type == 'BM+SGIL':
            # Genotype expressed as flattened bitmap + start/goal/info locations
            # Length is [(size-2) x (size-2) + (3 x 2)] 
            #           [  bitmap +  start/goal/info  ]
            assert hasattr(self, 'info_pos') and self.info_pos is not None, \
                'Invalid gt_type for current maze: info_pos is not set'
            genotype_size = (self.size - 2)**2 + 6
            genotype = np.zeros(genotype_size)
            genotype[:((self.size - 2)**2)] = self.bit_map.flatten()
            genotype[-6:-4] = self.start_pos
            genotype[-4:-2] = self.goal_pos
            genotype[-2:] = self.info_pos
        else:
            raise NotImplementedError(
                'Unknown genotype type: {}'.format(self.gt_type))
        
        self.genotype = genotype.astype(int)

        return self.process_genotype_new(genotype)
        
    @staticmethod
    def process_genotype(genotype, size, gt_type='BM+SGL'):
        """Extract information from genotype and genotype type."""
        gisn = genotype is None

        # TODO(costales): Take into consideration the upper and lower bounds 
        # provided by the state and goal regions passed into the QD environment

        if gt_type == 'BM+SGL':
            # Genotype expressed as flattened bitmap + start/goal locations
            # Length is [(size-2) x (size-2) + (2 x 2)] 
            #           [  bitmap    +    start/goal  ]

            # Ensure genotype is correct length
            genotype_size = (size - 2)**2 + 4
            genotype_lower_bounds = np.array(
                [0] * (size - 2)**2 + [1] * 4)
            genotype_upper_bounds = np.array(
                [1] * (size - 2)**2 + [size - 2] * 4)
            genotype_bounds = [(l, u) for l, u in 
                                    zip(list(genotype_lower_bounds), 
                                        list(genotype_upper_bounds))]
            assert gisn or len(genotype) == genotype_size, \
                f'Genotype length {len(genotype)} != {genotype_size} \n' \
                f'Genotype: {genotype}'

            # Extract bitmap and start/goal locations
            bit_map = None if gisn else np.array(genotype[:((size - 2)**2)])
            bit_map = None if gisn else bit_map.reshape(size - 2, size - 2)
            bit_map_padded = np.pad(bit_map, pad_width=1, mode='constant', constant_values=1)
            start_pos = None if gisn else np.array(genotype[-4:-2]).astype(int)
            goal_pos = None if gisn else np.array(genotype[-2:]).astype(int)

            # Return processed genotype
            processed_genotype = {
                'bit_map': bit_map,
                'bit_map_padded': bit_map_padded,
                'start_pos': start_pos,
                'goal_pos': goal_pos,
                'genotype': genotype,
                'genotype_bounds': genotype_bounds,
                'genotype_size': genotype_size,
                'genotype_lower_bounds': genotype_lower_bounds,
                'genotype_upper_bounds': genotype_upper_bounds,
            }
            
        elif gt_type == 'BM+SGIL':
            # Genotype expressed as flattened bitmap + start/goal/info locations
            # Length is [(size-2) x (size-2) + (3 x 2)] 
            #           [  bitmap +  start/goal/info  ]

            # Ensure genotype is correct length
            genotype_size = (size - 2)**2 + 6
            genotype_lower_bounds = np.array(
                [0] * (size - 2)**2 + [1] * 6)
            genotype_upper_bounds = np.array(
                [1] * (size - 2)**2 + [size - 2] * 6)
            genotype_bounds = [(l, u) for l, u in 
                                    zip(list(genotype_lower_bounds), 
                                        list(genotype_upper_bounds))]
            assert gisn or len(genotype) == (size - 2)**2 + 6, \
                f'Genotype length {len(genotype)} != {genotype_size}'
            
            # Extract bitmap and start/goal/info locations
            bit_map = None if gisn else np.array(genotype[:((size - 2)**2)])
            bit_map = None if gisn else bit_map.reshape(size - 2, size - 2)
            bit_map_padded = np.pad(bit_map, pad_width=1, mode='constant', constant_values=1)
            start_pos = None if gisn else np.array(genotype[-6:-4]).astype(int)
            goal_pos = None if gisn else np.array(genotype[-4:-2]).astype(int)
            info_pos = None if gisn else np.array(genotype[-2:]).astype(int)

            # Return processed genotype
            processed_genotype = {
                'bit_map': bit_map,
                'bit_map_padded': bit_map_padded,
                'start_pos': start_pos,
                'goal_pos': goal_pos,
                'info_pos': info_pos,
                'genotype': genotype,
                'genotype_bounds': genotype_bounds,
                'genotype_size': genotype_size,
                'genotype_lower_bounds': genotype_lower_bounds,
                'genotype_upper_bounds': genotype_upper_bounds,
            }
        else:
            raise NotImplementedError(
                'Unknown genotype type: {}'.format(gt_type))
        
        return Namespace(**processed_genotype)
    
    @staticmethod
    def is_valid_genotype(processed_genotype, gt_type='BM+SGL'):
        """Check if genotype is valid"""
        pg = processed_genotype
        # Check if genotype within bounds
        if pg.genotype is None:
            return False, 'none_genotype'
        if np.any(pg.genotype < pg.genotype_lower_bounds):
            return False, 'lower_bound_violation'
        if np.any(pg.genotype > pg.genotype_upper_bounds):
            return False, 'upper_bound_violation'
        # Check that common objects do not overlap with each other
        if pg.bit_map_padded[pg.start_pos[1], pg.start_pos[0]] == 1:
            return False, 'start_wall_overlap'
        if pg.bit_map_padded[pg.goal_pos[1], pg.goal_pos[0]] == 1:
            return False, 'goal_wall_overlap'
        if pg.start_pos[1] == pg.goal_pos[1] and \
            pg.start_pos[0] == pg.goal_pos[0]:
            return False, 'start_goal_overlap'
        # Check that gt_type-specific objects do not overlap with each other
        if gt_type == 'BM+SGL':
            pass
        elif gt_type == 'BM+SGIL':
            if pg.bit_map_padded[pg.info_pos[1], pg.info_pos[0]] == 1:
                return False, 'info_wall_overlap'
            if pg.start_pos[1] == pg.info_pos[1] and \
                pg.start_pos[0] == pg.info_pos[0]:
                return False, 'start_info_overlap'
            if pg.goal_pos[1] == pg.info_pos[1] and \
                pg.goal_pos[0] == pg.info_pos[0]:
                return False, 'goal_info_overlap'
        # Check that agent can reach goal/info from start position
        if not MazeEnv.is_reachable(
                pg.bit_map_padded, pg.start_pos, pg.goal_pos):
            return False, 'start_goal_unreachable'
        if gt_type == 'BM+SGIL':
            if not MazeEnv.is_reachable(
                    pg.bit_map_padded, pg.start_pos, pg.info_pos):
                return False, 'start_info_unreachable'
            # NOTE: info_pos will always be able to reach goal_pos if previous
            # checks pass (triangle inequality)
        return True, None

    def generate_level_from_genotype(self, genotype, gt_type='BM+SGL'):
        """Generate level from genotype, which is a sequence of ints"""
        if genotype is not None:
            self.genotype = np.array(genotype)
        
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
        self.bit_map = pg.bit_map
        self.start_pos = pg.start_pos
        self.goal_pos = pg.goal_pos

        # Set gt_type-specific variables
        if gt_type == 'BM+SGL':
            pass
        elif gt_type == 'BM+SGIL':
            self.info_pos = pg.info_pos

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

        if self.maze_generator is not None:
            self.maze_generator.reset_seed(self.rng_seed)
            self.bit_map = self.maze_generator.generate_bitmap()
            self.bit_map_padded = np.pad(self.bit_map, pad_width=1, mode='constant', constant_values=1)
            self.bit_map_shape = self.bit_map.shape
            self.bit_map_size = self.bit_map.size
        else:
            self.bit_map = self.new_bitmap_fn()

        prevs = []
        # Set start location
        if self._init_start_pos is None:
            # Sample starting location
            self.start_pos, prng = sample_maze_location(
                size=self.size,
                region=self.start_sampler_region,
                sample_mode=self.start_sampler,
                border_size=1,
                bit_map=self.bit_map_padded,
                seed=self.rng_seed,
                prevs=prevs
            )
            self.start_pos = np.array(self.start_pos)
        else:
            self.start_pos = np.array(self.start_pos)
        prevs.append(tuple(self.start_pos))

        # Set goal location
        if self._init_goal_pos is None:
            # Sample goal location
            self.goal_pos, prng = sample_maze_location(
                size=self.size,
                region=self.goal_sampler_region,
                sample_mode=self.goal_sampler,
                border_size=1,
                bit_map=self.bit_map_padded,
                seed=self.rng_seed,
                prng=prng,
                prevs=prevs
            )
            self.goal_pos = np.array(self.goal_pos)
        else:
            self.goal_pos = np.array(self.goal_pos)
        prevs.append(tuple(self.goal_pos))

        # If there is an "info cell" where agent can learn of goal location
        if self.use_info_cell:
            # Set info location
            if self._init_info_pos is None:
                # Sample info location
                self.info_pos, prng = sample_maze_location(
                    size=self.size,
                    region=self.info_sampler_region,
                    sample_mode=self.info_sampler,
                    border_size=1,
                    bit_map=self.bit_map_padded,
                    seed=self.rng_seed,
                    prng=prng,
                    prevs=prevs
                )
                self.info_pos = np.array(self.info_pos)
            else:
                self.info_pos = np.array(self.info_pos)

        return self.goal_pos

    def gen_obs(self):
        """Add goal loc and info loc to observations"""
        obs = super().gen_obs()

        # DEBUG: Test
        obs['goal_known'] = (0,)
        obs['goal_loc'] = (0, 0)
        obs['info_known'] = (0,)
        obs['info_loc'] = (0, 0)

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
                   "Bit map:\n{}\n".format(self.bit_map) + \
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
            self.generate_level_from_genotype(genotype=task)
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
            return self.genotype.copy()
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


if __name__ == '__main__':
    bit_map_padded = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ])

    start_pos = [7, 4]  # (x, y)
    goal_pos = [1, 3]  # (x, y)

    reachable = MazeEnv.is_reachable(bit_map_padded, start_pos, goal_pos)

    # Should return `True`.
    print('Is the target reachable from the start?', reachable)
