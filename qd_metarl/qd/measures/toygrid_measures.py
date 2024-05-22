import numpy as np
from typing import Any


class MeasureInfo:
    def __init__(self, 
                 full_range: tuple[float, float], 
                 sample_range: tuple[float, float], 
                 normal_params: tuple[float, float],
                 max_cells: int, 
                 sample_dist: Any,
                 fn: callable, 
                 args: tuple[str]):
        self.full_range = full_range
        self.sample_range = sample_range
        self.normal_params = normal_params
        self.max_cells = max_cells
        self.sample_dist = sample_dist
        self.fn = fn
        self.args = args

class ToyGridMeasures:
    
    @staticmethod
    def get_measures_info(env_name):
        del env_name
        measures_info = {
            'x_pos': MeasureInfo(full_range=(0.9, 5.1), sample_range=(0.9, 5.1),
                                 normal_params=None, max_cells=None, sample_dist=None,
                                 fn=None, args=None),                    
            'y_pos': MeasureInfo(full_range=(0.9, 9.1), sample_range=(0.9, 9.1),
                                 normal_params=None, max_cells=None, sample_dist=None,
                                 fn=None, args=None),
        }
        return measures_info

    @staticmethod
    def compute_measures(genotype: np.ndarray,
                         goal_pos: tuple[int, int],
                         measures: list):
        """" Compute the measures for the given maze. """

        if measures is None:
            measures = ToyGridMeasures.get_all_measures()
        
        # Compute only the relevant ones
        measures_dict = dict()
        if 'x_pos' in measures:
            measures_dict['x_pos'] = goal_pos[1]
        if 'y_pos' in measures:
            measures_dict['y_pos'] = goal_pos[0]
            
        # Ensure no NaNs or Infs
        for k, v in measures_dict.items():
            if np.isnan(v) or np.isinf(v):
                measures_dict[k] = -1
            
        return measures_dict
    
    @staticmethod
    def get_all_measures():
        """Return a list of all available measures for the Maze environment."""
        return ['y_pos', 'x_pos']
