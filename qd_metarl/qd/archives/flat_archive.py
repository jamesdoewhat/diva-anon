"""Custom (flat) GridArchive."""
import gin
from numpy_groupies import aggregate_nb as aggregate
import numpy as np
import ribs.archives
from ribs.archives._elite import EliteBatch, Elite
from ribs.archives._archive_stats import ArchiveStats
from ribs._utils import (check_1d_shape, check_batch_shape, check_finite,
                         check_is_1d, readonly, validate_batch_args,
                         validate_single_args)
from typing import Any


@gin.configurable
class FlatArchive(ribs.archives.ArchiveBase):
    """ Flat version of GridArchive. """

    def __init__(self,
                 solution_dim,
                 cells: int,
                 seed: int = None,
                 dtype: Any = np.float64,
                 record_history: bool = False,
                 qd_score_offset: float = 0.0,
                 threshold_min: float = -0.00001):
        super().__init__(solution_dim=solution_dim,
                         cells=cells,
                         measure_dim=1,  # Placeholder value
                         learning_rate=1.0,
                         seed=seed,
                         dtype=dtype,
                         qd_score_offset=qd_score_offset,
                         threshold_min=threshold_min)
        
        self._record_history = False
        # record_history
        # self._history = [] if self._record_history else None
        self._sample_weights = np.zeros(self._cells, dtype=self.dtype)

        self._occupied_indices = np.array(list(range(self._cells))).astype(np.int32)
    
        # Keep track of solution set to ensure no duplicates
        self._solution_set = set()
        self._steps_since_last_updated = dict()
        self._stale_solutions_to_update = None

    @property
    def occupied_indices(self):
        return self._occupied_indices[:self._num_occupied]
    
    @property
    def unoccupied_indices(self):
        # Create an array of all possible indices from 0 to self._cells-1
        all_indices = np.arange(self._cells)
        # Use np.setdiff1d to find indices that are in all_indices but not in self._occupied_indices
        # It doesn't require the input arrays to be sorted but will sort them internally for efficiency
        return np.setdiff1d(all_indices, self._occupied_indices[:self._num_occupied], assume_unique=True)

    def get_matrices(self):
        """ Returns a matrix of the objective values (and occupied) for all cells. """
        indices = self.occupied_indices
        
        # Assert that there are no inf values in indices
        assert not np.any(np.isinf(indices))
        
        matrices = {
            'objective': self._objective_arr[indices],
            'occupied_indices': indices,
            'solutions': self._solution_arr[indices],         
        }
        
        return matrices

    def new_history_gen(self):
        """ Starts a new generation in the history. """
        return
        # if self._record_history:
        #     self._history.append([])

    def history(self):
        """ Gets the current history. """
        return None
        # return self._history

    def best_elite(self):
        """ Returns the best Elite in the archive. """
        if self.empty:
            raise IndexError("No elements in archive.")

        objectives = self._objective_values[self._occupied_indices_cols]
        idx = self._occupied_indices[np.argmax(objectives)]
        return ribs.archives.Elite(
            readonly(self._solutions[idx]),
            self._objective_values[idx],
            readonly(self._behavior_values[idx]),
            idx,
            self._metadata[idx],
        )

    def sample_elites(self, n):
        """ Samples elites from the archive, supporting nonuniform sampling.

        Args:
            n (int): Number of elites to sample.
        Returns:
            EliteBatch: A batch of elites randomly selected from the archive.
        Raises:
            IndexError: The archive is empty.
        """
        if self.empty:
            raise IndexError("No elements in archive.")
        if len(np.nonzero(self._sample_weights)) <= 1:
            # Sample uniformly.
            random_indices = self._rng.integers(self._num_occupied, size=n)
            selected_indices = self._occupied_indices[random_indices]
        else:
            # Use self._sample_weights to sample:
            sample_weights = self._sample_weights / np.sum(self._sample_weights)
            selected_indices = self._rng.choice(
                list(range(len(self._solution_arr))), size=n, p=sample_weights)

        return EliteBatch(
            readonly(self._solution_arr[selected_indices]),
            readonly(self._objective_arr[selected_indices]),
            readonly(self._measures_arr[selected_indices]),
            readonly(selected_indices),
            readonly(self._metadata_arr[selected_indices]),
        )
    
    def update_sample_weights(self, seed_weights, seed2index, weight_sum_to_check):
        """ Updates the sample weights for the archive.

        Args:
            seed_weights (dict): A dictionary mapping seeds to their weights.
            seed2index (dict): A dictionary mapping seeds to their indices in
                the archive.
        """
        # Reset the sample weights.
        self._sample_weights = np.zeros(self._cells, dtype=self.dtype)
        # Update the sample weights.
        for seed, weight in seed_weights.items():
            self._sample_weights[seed2index[seed]] = weight

        if not np.isclose(np.sum(self._sample_weights), weight_sum_to_check):
            print('WARNING: Sample weights sums not close: ', np.sum(self._sample_weights), weight_sum_to_check)

    def retrieve(self, measures_batch):
        del measures_batch
        raise NotImplementedError(
            "FlatArchive does not support retrieval.")

    def retrieve_single(self, measures):
        del measures
        raise NotImplementedError(
            "FlatArchive does not support retrieval.")
    
    def index_of(self, measures_batch):
        del measures_batch
        raise NotImplementedError(
            "FlatArchive does not support indexing by measures.")
    
    def index_of_single(self, measures):
        del measures
        raise NotImplementedError(
            "FlatArchive does not support indexing by measures.")
    
    def add(self,
            solution_batch,
            objective_batch,
            measures_batch,  # User should input dummy values for consistency
            metadata_batch=None):
        
        """ Inserts a batch of solutions into the archive. 
        
        We keep the archive sorted by objective value, so we need to merge the
        new solutions with the old solutions. We do this by creating a temporary
        array to store the merged solutions, and then taking the top `self._cells`
        solutions from the merged list.

        NOTE: Archive sorted in descending order of objective value.
        """
        self._state["add"] += 1

        # Sort each batch in descending order of objective value.
        sort_indices = np.argsort(-objective_batch)
        solution_batch = solution_batch[sort_indices]
        objective_batch = objective_batch[sort_indices]
        if measures_batch is not None:
            measures_batch = measures_batch[sort_indices]
        if metadata_batch is not None:
            metadata_batch = metadata_batch[sort_indices]
        
        ## Step 0: Preprocess input. ##
        solution_batch = np.asarray(solution_batch)
        objective_batch = np.asarray(objective_batch)
        measures_batch = np.asarray(measures_batch)
        batch_size = solution_batch.shape[0]
        metadata_batch = (np.empty(batch_size, dtype=object) if
                          metadata_batch is None else np.asarray(metadata_batch,
                                                                 dtype=object))

        validate_batch_args(
            archive=self,
            solution_batch=solution_batch,
            objective_batch=objective_batch,
            measures_batch=measures_batch,
            metadata_batch=metadata_batch,
        )

        # Create temporary arrays to store merged solutions
        merged_solutions = np.empty((2*self._cells, self._solution_dim), 
                                    dtype=self.dtype)
        merged_objectives = np.empty(2*self._cells, 
                                     dtype=self.dtype)
        merged_measures = np.empty((2*self._cells, self._measure_dim),
                                   dtype=self.dtype)
        merged_metadata = np.empty(2*self._cells, dtype=object)

        # Initialize value_batch and status_batch
        value_batch = np.zeros(batch_size, dtype=self.dtype)
        status_batch = np.zeros(batch_size, dtype=np.int32)
        is_new = np.zeros(batch_size, dtype=bool)
        improve_existing = np.zeros(batch_size, dtype=bool)
        
        i, j, k = 0, 0, 0  # indices for i=old solutions, j=new solutions, and k=merged list, respectively
        while i < self._num_occupied and j < len(solution_batch):
            if self._objective_arr[i] > objective_batch[j]:
                merged_solutions[k] = self._solution_arr[i]
                merged_objectives[k] = self._objective_arr[i]
                merged_measures[k] = self._measures_arr[i]
                merged_metadata[k] = self._metadata_arr[i]
                i += 1
                k += 1
            else:
                if (tuple(solution_batch[j].tolist()) not in self._solution_set
                    and objective_batch[j] > self._threshold_min):
                    merged_solutions[k] = solution_batch[j]
                    merged_objectives[k] = objective_batch[j]
                    if measures_batch is not None:
                        merged_measures[k] = measures_batch[j]
                    if metadata_batch is not None:
                        merged_metadata[k] = metadata_batch[j]
                    if k < self._cells:
                        # Track improvement
                        value_batch[j] = objective_batch[j] - self._objective_arr[i]
                        improve_existing[j] = True
                        self._solution_set.add(tuple(solution_batch[j].tolist()))
                        self._steps_since_last_updated[tuple(solution_batch[j].tolist())] = 0
                    k += 1
                j += 1
            

        # If there are leftover solutions in old solutions
        while i < self._num_occupied:
            merged_solutions[k] = self._solution_arr[i]
            merged_objectives[k] = self._objective_arr[i]
            merged_measures[k] = self._measures_arr[i]
            merged_metadata[k] = self._metadata_arr[i]
            if k >= self._cells:
                self._solution_set.remove(tuple(self._solution_arr[i].tolist()))
                self._steps_since_last_updated.pop(tuple(self._solution_arr[i].tolist()))
            i += 1
            k += 1

        # If there are leftover solutions in new solutions
        while j < len(objective_batch):
            if (tuple(solution_batch[j].tolist()) not in self._solution_set
                and objective_batch[j] > self._threshold_min):
                merged_solutions[k] = solution_batch[j]
                merged_objectives[k] = objective_batch[j]
                if measures_batch is not None:
                    merged_measures[k] = measures_batch[j]
                if metadata_batch is not None:
                    merged_metadata[k] = metadata_batch[j]
                if k < self._cells:
                    is_new[j] = True
                    value_batch[j] = objective_batch[j] - self._threshold_min
                    self._solution_set.add(tuple(solution_batch[j].tolist()))
                    self._steps_since_last_updated[tuple(solution_batch[j].tolist())] = 0
                k += 1
            j += 1


        self._num_occupied = min(self._cells, k)
        
        # Update status_batch
        status_batch[is_new] = 2
        status_batch[improve_existing] = 1

        # Now, we simply take the top `self._cells` solutions from the merged list
        self._solution_arr = merged_solutions[:self._cells]
        self._objective_arr = merged_objectives[:self._cells]
        self._measures_arr = merged_measures[:self._cells]
        self._metadata_arr = merged_metadata[:self._cells]
        # Get num non-empty cells in self._objective_arr
        self._occupied_arr.fill(False)
        self._occupied_arr[:self._num_occupied] = True

        # Return early if we cannot insert any solutions
        can_insert = is_new | improve_existing
        if not np.any(can_insert):
            return status_batch, value_batch

        # Update the thresholds
        pass  # Nothing to do

        ## Update archive stats. ##
        self._objective_sum = np.sum(self._objective_arr[:self._num_occupied])
        new_qd_score = (self._objective_sum - self.dtype(len(self))
                        * self._qd_score_offset)
        # Get first nonzero element of status_batch
        max_idx = 0
        max_obj_insert = objective_batch[0]


        if self._stats.obj_max is None or max_obj_insert > self._stats.obj_max:
            new_obj_max = max_obj_insert
            self._best_elite = Elite(
                readonly(np.copy(solution_batch[max_idx])),
                objective_batch[max_idx],
                readonly(np.copy(measures_batch[max_idx])),
                0,
                metadata_batch[max_idx],
            )
        else:
            new_obj_max = self._stats.obj_max
        
        norm_qd_score = self.dtype(new_qd_score / self.cells),

        self._stats = ArchiveStats(
            num_elites=len(self),
            coverage=self.dtype(len(self) / self.cells),
            qd_score=new_qd_score,
            norm_qd_score=norm_qd_score,
            obj_max=new_obj_max,
            obj_mean=self._objective_sum / self.dtype(len(self)),
        )

        # Swap indices to match the original order
        status_batch[sort_indices] = status_batch
        value_batch[sort_indices] = value_batch

        # if self._record_history and status_batch.all():
        #     # TODO: is status.all() actually what we want?
        #     self._history[-1].append([objective_batch, measures_batch])

        return status_batch, value_batch
    
    def add_single(self, solution, objective, measures, metadata=None):
        """ Inserts a single solution into the archive. """
        solution_batch = np.asarray([solution]).astype(self.dtype)
        objective_batch = np.asarray([objective]).astype(self.dtype)
        measures_batch = np.asarray([measures]).astype(self.dtype)
        if metadata is None:
            metadata_batch = None
        else:
            metadata_batch = np.asarray([metadata])
        
        status_batch, value_batch = self.add(solution_batch, objective_batch,
                                             measures_batch, metadata_batch)
        
        return status_batch[0], value_batch[0]
    
    ### TODO: All of the below methods are laughably inefficient... 
    ###       You'll want to update them at some point...
    def reset_sslu(self):
        """ Resets the steps since last updated for all cells. """
        for k in self._steps_since_last_updated.keys():
            self._steps_since_last_updated[k] = -1
    
    def increment_sslu(self, val=1): 
        """ Increments the steps since last updated for all cells. """
        for k in self._steps_since_last_updated.keys():
            self._steps_since_last_updated[k] += val
    
    def get_max_sslu(self):
        if len(self._steps_since_last_updated) == 0:
            return 0
        return np.max(list(self._steps_since_last_updated.values()))
    
    def get_stale_solutions(self, sslu_threshold=100):
        """ Return solutions with sslu > sslu_threshold. """
        # First, remove all values in self._steps_since_last_updated that are no longer in the archive
        # TODO: Figure out why these are still here in the first place
        for k in list(self._steps_since_last_updated.keys()):
            # Use numpy to check if k is in self._solution_arr
            karr = np.array(k)
            if len(karr.shape) == 1:
                karr = karr.reshape(1, -1)
            if not np.any((self._solution_arr == karr).all(axis=1)):
                self._steps_since_last_updated.pop(k)

        solutions = []
        for k, v in self._steps_since_last_updated.items():
            if v > sslu_threshold or v == -1:
                solutions.append(np.array(k))
        self._stale_solutions_to_update = np.array(solutions).astype(self._solution_arr.dtype)
        return self._stale_solutions_to_update
    
    def update_stale_solutions(self, objective_batch):
        """ Updates the stale solutions with the given objective_batch. """
        objective_batch = objective_batch[:len(self._stale_solutions_to_update)]

        for k in self._steps_since_last_updated.keys():
            self._steps_since_last_updated[k] = 0
        
        # Identify the indices of stale solutions in your archive.
        stale_indices = [np.where((self._solution_arr == k).all(axis=1))[0][0] 
                         for k in self._stale_solutions_to_update]

        # Update the values at these indices with new objectives.
        for idx, new_objective in zip(stale_indices, objective_batch):
            self._objective_arr[idx] = new_objective
        
        # Be sure to preserve order of the archive.
        self.sort_by_objective()
        self._stale_solutions_to_update = None

    def sort_by_objective(self):
        """ Sorts the archive by objective value in descending order. """
        # Get indices that would sort the objective array in descending order
        sorted_indices = np.argsort(self._objective_arr[:self._num_occupied])[::-1]
        
        # Use these indices to rearrange the objective array
        self._objective_arr = np.array(self._objective_arr)[sorted_indices]
        self._solution_arr = np.array(self._solution_arr)[sorted_indices]
        self._measures_arr = np.array(self._measures_arr)[sorted_indices]
        self._metadata_arr = np.array(self._metadata_arr)[sorted_indices]

        # If self._steps_since_last_updated is a dictionary, you would update it like:
        keys = list(self._steps_since_last_updated.keys())
        values = list(self._steps_since_last_updated.values())

        keys_sorted = [keys[i] for i in sorted_indices]
        values_sorted = [values[i] for i in sorted_indices]

        self._steps_since_last_updated = dict(zip(keys_sorted, values_sorted))

    def to_grid_archive(self, grid_archive_fn, compute_measures_fn, measures, measure_selector=None, gt_type=None):
        """ Converts the archive to a GridArchive. """
        grid_archive = grid_archive_fn()
        new_measures = []
        for s in self._solution_arr[:self._num_occupied]:
            meas = compute_measures_fn(s, measures, gt_type=gt_type)
            new_measures.append([meas[k] for k in measures])
        if measure_selector is not None:
            new_measures, _ = measure_selector.transform_data(new_measures)
        # Add solutions
        # grid_archive.new_history_gen()
        grid_archive.add(self._solution_arr[:self._num_occupied],
                         self._objective_arr[:self._num_occupied], 
                         new_measures,
                         self._metadata_arr[:self._num_occupied])
        return grid_archive