"""MazeConfig and MazeModule."""
import logging
import re
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Collection, Optional

import gin
import numpy as np
from scipy.sparse import csgraph
from skimage.segmentation import flood_fill

from .agents.rl_agent import RLAgentResult
from .envs.maze import MazeEnv
from .level import OBJ_TYPES_TO_INT
from .maze_result import MazeResult, MazeMetadata

logger = logging.getLogger(__name__)


@gin.configurable
@dataclass
class MazeConfig:
    """Config for maze envs."""

    # Objectives, Measures.
    objective_name: str = "none"
    """Either 'none', 'dfs', 'accel', or 'paired'. 'accel' and 'paired' can have
    a suffix '_fail' to use fail rate instead of the path length objective."""
    measure_names: Collection[str] = gin.REQUIRED

    # Aug data.
    augment_type: str = "optimal_path"
    """Either 'optimal_path' or 'agent_occupancy' or 'turns'. 'turns' is only
    applicable for 'accel'/'paired' objective."""

    # Results.
    aggregation_type: str = "mean"


class MazeModule:
    """Module for maze envs."""

    MIN_SCORE = 0
    MAX_SCORE = int(1e6)

    def __init__(self, config: MazeConfig):
        self.config = config

    def evaluate(
        self,
        level: np.ndarray,
        n_evals: int,  # pylint: disable = unused-argument
        seed: Optional[int] = None,  # pylint: disable = unused-argument
        rl_agent_func: Optional[callable] = None,
    ):
        """Evaluates the solution.

        Args:
            level: Integer array with shape (lvl_height, lvl_width)
            n_evals: Number of repetitions to aggregate over.
            seed: Seed for the evaluation. Only applies if using stochastic
                settings.
            rl_agent_func: Callable that returns a configured RL Agent to use.
        Returns:
            ObjectiveResult with n_evals solutions.
        """
        np.random.seed(seed)

        # Path length calculation
        adj = MazeEnv._get_adj(level)

        # Find the best distances
        dist, predecessors = csgraph.floyd_warshall(adj,
                                                    return_predecessors=True)
        dist[dist == np.inf] = -np.inf  # For easier argmax to find the diameter

        if dist.max() < 1:  # True even if it is -inf
            objs = [0] * n_evals
            solvability = 0
            optimal_path_length = -1
            n_left_turns = 0
            n_right_turns = 0
            agent_path_length = 0
            frac_explored_cells = 0
            n_repeated_cells = 0

            aug_level = level.copy()
            endpoint_level = level.copy()
        else:
            solvability = 1
            optimal_path_length = dist.max()

            # Label the start and the end point
            endpoints = np.unravel_index(dist.argmax(), dist.shape)
            start_cell, end_cell = zip(
                *np.unravel_index(endpoints, level.shape))
            path_level = level.copy()
            path_level[start_cell] = OBJ_TYPES_TO_INT["S"]
            path_level[end_cell] = OBJ_TYPES_TO_INT["G"]

            endpoint_level = level.copy()
            endpoint_level[start_cell] = OBJ_TYPES_TO_INT["S"]
            endpoint_level[end_cell] = OBJ_TYPES_TO_INT["G"]

            # Label the path in between
            cur_cell_n = endpoints[0]
            end_cell_n = endpoints[1]
            while True:
                cur_cell_n = predecessors[end_cell_n, cur_cell_n]
                if cur_cell_n == end_cell_n:
                    break
                cur_cell = np.unravel_index(cur_cell_n, level.shape)
                path_level[cur_cell] = OBJ_TYPES_TO_INT["P"]

            if self.config.objective_name == "none":
                objs = [1] * n_evals
                n_left_turns = 0
                n_right_turns = 0
                agent_path_length = 0
                frac_explored_cells = 0
                n_repeated_cells = 0

                if self.config.augment_type == "optimal_path":
                    aug_level = path_level
                elif self.config.augment_type == "agent_occupancy":
                    aug_level = np.zeros_like(path_level, dtype=float)
                    aug_level[path_level == OBJ_TYPES_TO_INT["S"]] = 1
                    aug_level[path_level == OBJ_TYPES_TO_INT["G"]] = 1
                    aug_level[path_level == OBJ_TYPES_TO_INT["P"]] = 1
                    aug_level /= n_evals
                elif self.config.augment_type == "turns":
                    warnings.warn(
                        "'turns' is not applicable to 'none' objective.")
                    aug_level = np.zeros_like(path_level, dtype=float)
                    aug_level[path_level == OBJ_TYPES_TO_INT["S"]] = 1
                    aug_level[path_level == OBJ_TYPES_TO_INT["G"]] = 1
                    aug_level[path_level == OBJ_TYPES_TO_INT["P"]] = 1
                    aug_level /= n_evals
                else:
                    raise ValueError(
                        f"Unknown augment type: {self.config.augment_type}")
            elif self.config.objective_name == "dfs":
                # DFS path
                dfo, _ = csgraph.depth_first_order(adj, int(endpoints[0]))
                dfs_dist = np.where(dfo == endpoints[1])[0][0] + 1
                objs = [dfs_dist / optimal_path_length] * n_evals
                n_left_turns = 0
                n_right_turns = 0
                agent_path_length = 0
                frac_explored_cells = 0
                n_repeated_cells = 0

                if self.config.augment_type == "optimal_path":
                    aug_level = path_level
                elif self.config.augment_type == "agent_occupancy":
                    aug_level = np.zeros_like(level, dtype=float)
                    for cell_n in dfo[:dfs_dist]:
                        cell = np.unravel_index(cell_n, level.shape)
                        aug_level[cell] += 1
                    aug_level /= n_evals
                elif self.config.augment_type == "turns":
                    warnings.warn(
                        "'turns' is not applicable to 'dfs' objective.")
                    aug_level = np.zeros_like(level, dtype=float)
                    for cell_n in dfo[:dfs_dist]:
                        cell = np.unravel_index(cell_n, level.shape)
                        aug_level[cell] += 1
                    aug_level /= n_evals
                else:
                    raise ValueError(
                        f"Unknown augment type: {self.config.augment_type}")
            elif re.search("accel|paired", self.config.objective_name):
                # Offset start, goal to account for the added outer walls
                start_pos = (start_cell[1] + 1, start_cell[0] + 1)
                goal_pos = (end_cell[1] + 1, end_cell[0] + 1)
                env_func = partial(MazeEnv,
                                   size=level.shape[0] + 2,
                                   bit_map=level,
                                   start_pos=start_pos,
                                   goal_pos=goal_pos)
                rl_agent = rl_agent_func(env_func, n_evals)

                agent_eval_kwargs = {}
                if self.config.augment_type == "turns":
                    agent_eval_kwargs["aug_type"] = "turns"
                rl_result: RLAgentResult = rl_agent.eval_and_track(
                    level_shape=level.shape, **agent_eval_kwargs)
                aug_level = rl_result.aug_level
                agent_path_length = np.mean(rl_result.path_lengths)
                n_left_turns = np.median(rl_result.n_left_turns)
                n_right_turns = np.median(rl_result.n_right_turns)
                n_repeated_cells = rl_result.n_repeated_cells

                flood_fill_level = flood_fill(level,
                                              start_cell,
                                              -1,
                                              connectivity=1)
                n_reachable_cells = np.sum(flood_fill_level == -1)
                n_explored_cells = np.sum(aug_level > 0)
                frac_explored_cells = n_explored_cells / n_reachable_cells

                if "_fail" in self.config.objective_name:
                    objs = rl_result.failed_list
                elif "_none" in self.config.objective_name:
                    objs = [1] * n_evals
                else:
                    objs = rl_result.path_lengths

                if self.config.augment_type == "optimal_path":
                    aug_level = path_level
                elif self.config.augment_type in ["agent_occupancy", "turns"]:
                    pass
                else:
                    raise ValueError(
                        f"Unknown augment type: {self.config.augment_type}")
            else:
                raise ValueError(
                    f"Unknown objective name: {self.config.objective_name}")

        num_blocks = int(np.sum(level == OBJ_TYPES_TO_INT["X"]))

        maze_metadata = MazeMetadata(
            level=endpoint_level,
            aug_level=aug_level,
            objs=objs,
            solvability=solvability,
            num_blocks=num_blocks,
            optimal_path_length=optimal_path_length,
            agent_path_length=agent_path_length,
            n_left_turns=n_left_turns,
            n_right_turns=n_right_turns,
            n_repeated_cells=n_repeated_cells,
            frac_explored_cells=frac_explored_cells,
        )

        return MazeResult.from_raw(
            maze_metadata=maze_metadata,
            opts={
                "aggregation": self.config.aggregation_type,
                "measure_names": self.config.measure_names,
            },
        )

    def actual_qd_score(self, objs: "array-like"):
        """Calculates QD score of the given objectives.

        Scores are normalized to be non-negative by subtracting a constant min
        score.

        Args:
            objs: List of objective values.
        """
        objs = np.array(objs)
        objs -= self.MIN_SCORE
        if np.any(objs < 0):
            warnings.warn("Some objective values are still negative.")
        return np.sum(objs)