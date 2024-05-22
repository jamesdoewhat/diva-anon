"""Custom GridArchive."""
from typing import Sequence, Tuple
import gin
from numpy_groupies import aggregate_nb as aggregate
from scipy.stats import norm
from scipy.stats import multivariate_normal
import numpy as np
import ribs.archives
from ribs.archives._elite import EliteBatch
from ribs._utils import (check_batch_shape, check_finite,
                         validate_batch_args, readonly)
from scipy.spatial import KDTree
from scipy.special import logsumexp
from scipy.ndimage import gaussian_filter



def safe_logsumexp(inputs, axis=None, keepdims=False):
    """Avoid underflow and handle shapes correctly in logsumexp."""
    a_max = np.max(inputs, axis=axis, keepdims=True)
    min_exp = np.log(np.finfo(np.float64).tiny)
    inputs_adj = np.clip(inputs - a_max, a_min=min_exp, a_max=None)
    output = np.log(np.sum(np.exp(inputs_adj), axis=axis, keepdims=True))
    
    if keepdims:
        output += a_max
    else:
        output += a_max.squeeze(axis)
    
    return output


@gin.configurable
class GridArchive(ribs.archives.GridArchive):
    """ Based on pyribs GridArchive.

    This archive records history of its objectives and behavior values if
    record_history is True. Before each generation, call new_history_gen() to
    start recording history for that gen. new_history_gen() must be called
    before calling add() for the first time.
    """

    def __init__(self,
                 solution_dim,
                 dims,
                 ranges,
                 seed=None,
                 dtype=np.float64,
                 record_history=False,
                 qd_score_offset=0.0,
                 threshold_min=-0.00001,
                 use_normal_prior=False,
                 normal_prior_mean=0.5,
                 normal_prior_std_dev=0.3,
                 kd_prior=None,
                 sample_ranges=None,
                 update_sample_mask=False,
                 normal_prior_axes=None,
                 sol_type=np.int8,
                 sparsity_reweighting=False,
                 sparsity_reweighting_sigma=0.1,
                 sample_mask_min_solutions=20,
                 prior_mixing_percentage=1.0,
                 sample_mask_mixing_percentage=1.0,
                 ):
        super().__init__(solution_dim=solution_dim,
                         dims=dims, 
                         ranges=ranges, 
                         seed=seed, 
                         dtype=dtype, 
                         qd_score_offset=qd_score_offset,
                         threshold_min=threshold_min)
        # Change dtype to whatever sol_type is passed in
        # since pyribs by default sets a dtype uniformly across all arrays
        self._solution_arr = self._solution_arr.astype(sol_type)
        
        # Set the archive's tracking variables
        self._record_history = False  # record_history
        # self._history = [] if self._record_history else None
        self._sample_weights = np.zeros(self._cells, dtype=self.dtype)
        self._post_prior_sample_weights = np.zeros(self._cells, dtype=self.dtype)
        self._steps_since_last_updated = np.zeros(self._cells, dtype=int) - 1
        self._stale_indices_to_update = None
        self.warm_start_percentage_complete = 0.0
        
        # Set the prior weights if used
        self._sample_mask_min_solutions = sample_mask_min_solutions
        self._sparsity_reweighting = sparsity_reweighting
        self._sparsity_reweighting_sigma = sparsity_reweighting_sigma
        self._use_normal_prior = use_normal_prior
        self._normal_prior_mean = normal_prior_mean
        self._normal_prior_std_dev = normal_prior_std_dev
        self._kd_prior = kd_prior
        self._use_kd_prior = kd_prior is not None
        self._use_prior = self._use_normal_prior or self._use_kd_prior
        if self._use_normal_prior:
            self.set_normal_prior_weights(normal_prior_axes)
        elif self._use_kd_prior:
            self.set_kd_prior_weights()

        # Initialize the sample mask
        self._init_sample_mask(sample_ranges, update_sample_mask)

        # Must initialize reduced sample weights (used for efficient sampling)
        self._indices_for_sampling = np.array(list(range(len(self._solution_arr))))
        self._indices_for_sampling_reduced = self._indices_for_sampling[self._sample_weights > 0]
        self._sample_weights_reduced = self._sample_weights[self._sample_weights > 0]

        print('GridArchive.__init__ > Archive total:         ', self._cells)
        print('GridArchive.__init__ > Target mask total:     ', np.sum(self._target_sample_mask))
        print('GridArchive.__init__ > Target mask percentage:', np.round(np.mean(self._target_sample_mask), 4)*100, '%')

    def _init_sample_mask(self, sample_ranges, update_sample_mask):
        """ Initialize the sample mask and related variables. """
        self._sample_ranges = sample_ranges
        self._update_sample_mask = update_sample_mask
        # Set sample mask
        self._target_sample_mask_multid = np.ones(self._cells, dtype=int)
        # Reshape to dims 
        self._target_sample_mask_multid = self._target_sample_mask_multid.reshape(self._dims)
        if sample_ranges is not None:
            assert len(sample_ranges) == len(self._dims)
            for i, r in enumerate(sample_ranges):
                # Skip if sample range is not defined
                if r is None: 
                    continue
                # We want to set the mask 0 for all cells outside the sample 
                # range, for this current dimension
                lower_bounds = self._boundaries[i][:-1] # NOTE see: GridArchive.boundaries
                upper_bounds = self._boundaries[i][1:]
                
                # We want to include all cells j where
                # lower_bounds[j] <= r[0] and upper_bounds[j] >= r[1]

                # Find the indices of the cells that are outside the range
                mask = (lower_bounds > r[1]) | (upper_bounds < r[0])

                # Set all _sample_mask values along the i'th axis at mask indices to 0
                self._target_sample_mask_multid = np.moveaxis(self._target_sample_mask_multid, i, 0)
                self._target_sample_mask_multid[mask] = 0
                self._target_sample_mask_multid = np.moveaxis(self._target_sample_mask_multid, 0, i)
        
        self._target_sample_mask = self._target_sample_mask_multid.ravel()
        self.target_size = np.sum(self._target_sample_mask)
        self._sample_weights = (self._target_sample_mask + 1e-9) / np.sum(self._target_sample_mask)
        self._uniform_sampling = True
        # Our sample mask is initially the full archive, and we want to shrink
        # towards the target sample mask
        self._sample_mask_multid = np.ones(self._dims, dtype=int)
        self._sample_mask = self._sample_mask_multid.ravel()
        self._edge_distances = self._calculate_initial_edge_distances()
        self._edge_boundaries = self._calculate_initial_edge_boundaries()
        print('GridArchive._init_sample_mask > edge_distances: ', self._edge_distances)
        print('GridArchive._init_sample_mask > edge_boundaries:', self._edge_boundaries)
        self._target_reached = False
        self.num_sols_in_target = 0
        self.target_percentage_covered = 0.0

    @property
    def occupied_mask(self):
        occupied_mask = np.zeros(self._dims, dtype=bool)
        occupied_mask.flat[self.occupied_indices] = True
        return occupied_mask
    
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

    def _compute_new_thresholds(self, threshold_arr, objective_batch,
                                index_batch, learning_rate):
        """ Update thresholds.

        Args:
            threshold_arr (np.ndarray): The threshold of the cells before
                updating. 1D array.
            objective_batch (np.ndarray): The objective values of the solution
                that is inserted into the archive for each cell. 1D array. We
                assume that the objective values are all higher than the
                thresholds of their respective cells.
            index_batch (np.ndarray): The archive index of the elements in
                objective batch.
        Returns:
            `new_threshold_batch` (A self.dtype array of new
            thresholds) and `threshold_update_indices` (A boolean
            array indicating which entries in `threshold_arr` should
            be updated.
        """
        # NOTE(costales): Duplicated this function since np.nan was not working
        # as a fill value for aggregate; replaced with 0 instead

        # Even though we do this check, it should not be possible to have
        # empty objective_batch or index_batch in the add() method since
        # we check that at least one cell is being updated by seeing if
        # can_insert has any True values.
        if objective_batch.size == 0 or index_batch.size == 0:
            return np.array([], dtype=self.dtype), np.array([], dtype=bool)

        # Compute the number of objectives inserted into each cell.
        objective_sizes = aggregate(index_batch,
                                    objective_batch,
                                    func="len",
                                    fill_value=0,
                                    size=threshold_arr.size)

        # These indices are with respect to the archive, so we can directly pass
        # them to threshold_arr.
        threshold_update_indices = objective_sizes > 0

        # Compute the sum of the objectives inserted into each cell.
        objective_sums = aggregate(index_batch,
                                   objective_batch,
                                   func="sum",
                                   fill_value=0,  # TODO: Not sure this is right
                                   size=threshold_arr.size)

        # Throw away indices that we do not care about.
        objective_sizes = objective_sizes[threshold_update_indices]
        objective_sums = objective_sums[threshold_update_indices]

        # Unlike in add_single, we do not need to worry about
        # old_threshold having -np.inf here as a result of threshold_min
        # being -np.inf. This is because the case with threshold_min =
        # -np.inf is handled separately since we compute the new
        # threshold based on the max objective in each cell in that case.
        old_threshold = np.copy(threshold_arr[threshold_update_indices])

        ratio = self.dtype(1.0 - learning_rate)**objective_sizes
        new_threshold_batch = (ratio * old_threshold +
                               (objective_sums / objective_sizes) * (1 - ratio))

        return new_threshold_batch, threshold_update_indices

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

    def new_history_gen(self):
        """ Starts a new generation in the history. """
        return
        # if self._record_history:
        #     self._history.append([])

    def history(self):
        """ Gets the current history. """
        return None
        # return self._history
    
    def reset_sslu(self):
        """ Resets the steps since last updated for all cells. """
        self._steps_since_last_updated[:] = -1
    
    def increment_sslu(self, val=1):
        """ Increments the steps since last updated for all cells. """
        self._steps_since_last_updated[self.occupied_indices] += val

    def get_max_sslu(self):
        if self._num_occupied == 0:
            return 0
        return np.max(self._steps_since_last_updated[:self._num_occupied])
    
    def get_sslu_matrix(self):
        """ Returns a matrix of the steps since last updated for all cells. """
        matrix = np.zeros(self._dims)
        indices = self.occupied_indices
        grid_indices = self.int_to_grid_index(indices)
        for i in range(len(indices)):
            # convert gind from numpy array to list
            gind = tuple(grid_indices[i].tolist())
            matrix[gind] = self._steps_since_last_updated[indices[i]]
        return matrix
    
    def get_matrices(self):
        """ Returns a matrix of the objective values (and occupied) for all cells. """
        indices = self.occupied_indices
        grid_indices = self.int_to_grid_index(indices)

        obj_matrix = np.zeros(self._dims)
        occupied_matrix = np.zeros(self._dims)
        sample_weight_matrix = np.zeros(self._dims)

        if self._use_prior:
            log_prior_reweighted_matrix = np.zeros(self._dims)
            log_prior_matrix = self._log_prior_weights_multid
            log_prior_reweighted_array = self._get_sparsity_reweighted_log_prior_weights()
            log_prior_reweighted_matrix = log_prior_reweighted_array.reshape(self._dims)

        # Fill in the matrices
        for ind, gind in zip(indices, grid_indices):
            # convert gind from numpy array to tuple for indexing
            gind = tuple(gind.tolist())
            # Index matrices by gind and pull values from flat index
            obj_matrix[gind] = self._objective_arr[ind]
            occupied_matrix[gind] = 1
            sample_weight_matrix[gind] = self._sample_weights[ind]
            
            if self._use_prior:
                log_prior_reweighted_matrix[gind] = log_prior_reweighted_array[ind]

        sample_mask = self._sample_mask_multid

        # Assert that there are no inf values in indices
        assert not np.any(np.isinf(indices))
        
        matrices = {
            'objective_matrix': obj_matrix,
            'occupied_matrix': occupied_matrix,
            # 'sample_mask_matrix': sample_mask,
            'sample_weight_matrix': sample_weight_matrix,
            'log_prior_matrix': log_prior_matrix if self._use_prior else None,
            # NOTE: abandoning for now:
            # 'log_prior_reweighted_matrix': log_prior_reweighted_matrix if self._use_prior else None,
            'occupied_indices': indices,
            'solutions': self._solution_arr[indices],         
        }
        
        return matrices

    def get_stale_solutions(self, sslu_threshold=100):
        """ Return solutions with sslu > sslu_threshold. """
        occupied_indices = self.occupied_indices
        condition = ((self._steps_since_last_updated[occupied_indices] > sslu_threshold) | 
                     ((self._steps_since_last_updated[occupied_indices]) == -1))
        indices = occupied_indices[condition]
        self._stale_indices_to_update = indices
        return self._solution_arr[indices]
    
    def update_stale_solutions(self, objective_batch):
        """ Updates the stale solutions with the given objective_batch. """
        objective_batch = objective_batch[:len(self._stale_indices_to_update)]  # We might have padded
        self._objective_arr[self._stale_indices_to_update] = objective_batch
        self._steps_since_last_updated[self._stale_indices_to_update] = 0
        self._stale_indices_to_update = None

    def get_out_of_bounds_indices(self, measures_batch):
        """ Returns indices of measures_batch that are out of bounds. """
        measures_batch = np.asarray(measures_batch)
        check_batch_shape(measures_batch, "measures_batch", self.measure_dim, "measure_dim")
        check_finite(measures_batch, "measures_batch")

        # Adding epsilon accounts for floating point precision errors from
        # transforming measures. We then cast to int32 to obtain integer indices.
        grid_index_batch = (
            (self._dims *
             (measures_batch - self._lower_bounds) + self._epsilon) /
            self._interval_size).astype(np.int32)

        # Find indices of measures_batch that are out of bounds.
        out_of_bounds_indices = np.any(
            (grid_index_batch < 0) | (grid_index_batch >= self._dims), axis=1)
        
        return out_of_bounds_indices

    def add(self, solution_batch, objective_batch, measures_batch, 
            metadata_batch=None):
        """ Adds a batch of solutions to the archive. 
        
        This method has added history (DSAGE) and sslu (DIVA) functionality.
        """
        # Check inputs and set out of bounds objectives to -inf to prevent
        # them from being added
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
        out_of_bounds_indices = self.get_out_of_bounds_indices(measures_batch)
        # For all values that are -1, set the objective to (effectively) -inf
        objective_batch[out_of_bounds_indices] = -10e9

        # Add solutions
        status, val = super().add(solution_batch, objective_batch, 
                                  measures_batch, metadata_batch)

        # # Only save obj and BCs in the history.
        # if self._record_history and status.all():
        #     # TODO: is status.all() actually what we want?
        #     self._history[-1].append([objective_batch, measures_batch])

        # Update self._steps_since_last_updated
        indices = np.nonzero(status)[0]  # Get indices where status is not zero
        if len(indices) == 0:
            return status, val
        # Get indices of cells that were updated
        archive_indices = self.index_of(measures_batch[indices])
        # For each cell that was updated, reset the steps since last update
        self._steps_since_last_updated[archive_indices] = 0

        occupied_indices = self.occupied_indices
        assert not np.any(np.isinf(occupied_indices))
        
        return status, val

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

        # Count the number of nonzero sample weights
        nonzero_sample_weights = np.nonzero(self._sample_weights)

        # This is just to demonstrate the dimensionality (there's an extra dimension)
        assert len(nonzero_sample_weights) == 1

        if len(nonzero_sample_weights[0]) <= 1:
            # Sample uniformly if sample weights are all zero.
            random_indices = self._rng.integers(self._num_occupied, size=n)
            selected_indices = self._occupied_indices[random_indices]
        else:
            # Use reduced sample weights for sampling
            sample_weights = self._sample_weights_reduced
            
            if self._uniform_sampling:
                selected_indices = self._rng.choice(self._indices_for_sampling_reduced, size=n)
            else:
                selected_indices = self._rng.choice(self._indices_for_sampling_reduced, size=n, p=sample_weights)

            self._post_prior_sample_weights = sample_weights

        # DEBUG - Print statements to check values of solutions
        # print('GridArchive.sample_elites > selected_indices:', selected_indices)
        # print('GridArchive.sample_elites > solutions:', readonly(self._solution_arr[selected_indices]))
        
        # DEBUG - Print warning solution array is filled with the same constant
        ret_sols = readonly(self._solution_arr[selected_indices])
        if np.max(ret_sols) == np.min(ret_sols):
            print('GridArchive.sample_elites > WARNING: Solutions are uniform value:', np.max(ret_sols))

        # Return the selected elites
        # TODO: Need to get rid of EliteBatch use for future versions of pyribs
        return EliteBatch(
            readonly(ret_sols),
            readonly(self._objective_arr[selected_indices]),
            readonly(self._measures_arr[selected_indices]),
            readonly(selected_indices),
            readonly(self._metadata_arr[selected_indices]))
    
    @staticmethod
    def standardize_bounds(lower_bounds, upper_bounds, means, std_devs):
        standardized_lower_bounds = [(l - means[i]) / std_devs[i] for i, l in enumerate(lower_bounds)]
        standardized_upper_bounds = [(u - means[i]) / std_devs[i] for i, u in enumerate(upper_bounds)]
        return standardized_lower_bounds, standardized_upper_bounds

    def set_normal_prior_weights(self, axes=None):
        """ Set the log of prior weights with normal distributions over specified axes 
        and uniform over others, extending to the specified coverage percentile. """
        if hasattr(self, '_log_prior_weights') and self._log_prior_weights is not None:
            raise ValueError('Prior weights already set.')

        num_dims = len(self._dims)
        if axes is None or not axes:  # Adjust to handle empty or None
            # Set a uniform distribution across all dimensions
            # Log of uniform distribution, can be set to zero for simplicity
            uniform_log_prob = -np.log(np.product(self._dims))  # To normalize across all points
            self._log_prior_weights_multid = np.full(self._dims, uniform_log_prob)
            self._log_prior_weights = self._log_prior_weights_multid.ravel()
            return

        # Define means and standard deviations
        z_score_99 = 2.576  # Z-score for the 99th percentile
        means = np.array([(l + u) / 2 for l, u in zip(self._lower_bounds, self._upper_bounds)])
        range_span = np.array([(u - l) / 2 for l, u in zip(self._lower_bounds, self._upper_bounds)])
        std_devs = np.array([span / z_score_99 for span in range_span])  # Standard deviation adjusted for 99% coverage
        relevant_means = means[axes]
        relevant_std_devs = std_devs[axes]
        relevant_lower_bounds = np.array(self._lower_bounds)[axes]
        relevant_upper_bounds = np.array(self._upper_bounds)[axes]
        relevant_dims = np.array(self._dims)[axes]
        print(f'GridArchive.set_normal_prior_weights > mean: {means}')
        print(f'GridArchive.set_normal_prior_weights > std_devs: {std_devs}')
        print(f'GridArchive.set_normal_prior_weights > lower_bounds: {self._lower_bounds}')
        print(f'GridArchive.set_normal_prior_weights > upper_bounds: {self._upper_bounds}')

        # Standardize bounds
        print('Standardizing bounds...')
        standardized_lower_bounds, standardized_upper_bounds = self.standardize_bounds(relevant_lower_bounds, relevant_upper_bounds, relevant_means, relevant_std_devs)
        print(f'GridArchive.set_normal_prior_weights > standardized_lower_bounds: {standardized_lower_bounds}')
        print(f'GridArchive.set_normal_prior_weights > standardized_upper_bounds: {standardized_upper_bounds}')

        # Create grid with standardized bounds
        grid_points = [np.linspace(l, u, d) for l, u, d in zip(standardized_lower_bounds, standardized_upper_bounds, relevant_dims)]
        mesh = np.meshgrid(*grid_points, indexing='ij')
        positions = np.stack(mesh, axis=-1)

        # Use standardized means (0) and standard deviations (1) for the normal distribution
        standardized_means = np.zeros_like(relevant_means)
        standardized_covariance = np.eye(len(relevant_dims))  # Identity matrix for unit variance

        # Calculate normal prior weights using logpdf
        rv = multivariate_normal(mean=standardized_means, cov=standardized_covariance)
        log_np_weights = rv.logpdf(positions.reshape(-1, len(standardized_means))).reshape(positions.shape[:-1])
        
        # Expand to full dimensionality
        full_log_np_weights = np.zeros(self._dims)
        # Assign computed log probabilities to slices, broadcasting along non-specified axes
        indexer = tuple(slice(None) if i in axes else np.newaxis for i in range(num_dims))
        full_log_np_weights[indexer] = log_np_weights

        # Stabilize and normalize log probabilities
        max_log = np.max(full_log_np_weights)
        stabilized_log_weights = full_log_np_weights - max_log  # normalize by max to avoid underflow
        # Clip log probabilities to avoid too low values that cause underflow on exp
        clipped_log_weights = np.clip(stabilized_log_weights, a_min=-700, a_max=None)

        self._log_prior_weights_multid = clipped_log_weights
        self._log_prior_weights = clipped_log_weights.ravel()

    def set_kd_prior_weights(self):
        """ Returns the kd prior weights for each cell. """
        raise NotImplementedError('This method is defunct.')

    def update_sample_mask(self, perc=None):
        # Update sample mask
        if (not self._update_sample_mask or self._target_reached or 
                self._num_occupied <= self._sample_mask_min_solutions):
            print(f'GridArchive.update_sample_mask > NOT UPDATING SAMPLE MASK (SM)')
            return
        print(f'GridArchive.update_sample_mask > UPDATING SAMPLE MASK (SM)')
        self._shrink_edges(perc=perc)
        self._sample_mask = self._sample_mask_multid.ravel()

    def _calculate_initial_edge_distances(self):
        """
        Calculate the distances by which the sample_mask exceeds the target_sample_mask
        along each dimension and at each end, to identify shrinkage priorities.
        """
        target_bounds_by_axis = []
        edge_distances = {}
        for axis in range(len(self._dims)):
            sample_bounds = self._get_bounds(self._sample_mask_multid, axis)
            target_bounds = self._get_bounds(self._target_sample_mask_multid, axis)
            target_bounds_by_axis.append(target_bounds)
            # Distance from the start of the sample_mask to the start of the target_sample_mask
            start_distance = target_bounds[0] - sample_bounds[0]
            # Distance from the end of the target_sample_mask to the end of the sample_mask
            end_distance = sample_bounds[1] - target_bounds[1]
            # Store distances, ensuring they're non-negative (indicating areas where the sample_mask can shrink)
            edge_distances[axis] = {'start': int(max(0, start_distance)), 'end': int(max(0, end_distance))}
        
        self.target_bounds_by_axis = target_bounds_by_axis
        return edge_distances
    
    def _calculate_initial_edge_boundaries(self):
        """
        Calculate the initial edge boundaries of the sample mask, which are the indices
        along each axis where the mask transitions from 0 to 1.
        """
        edge_boundaries = {}
        for axis in range(len(self._dims)):
            # Find the first and last indices along the axis where the mask is True
            start_index, end_index = self._get_bounds(self._sample_mask_multid, axis)
            edge_boundaries[axis] = {'start': int(start_index), 'end': int(end_index)}
        
        return edge_boundaries
    
    def _get_bounds(self, mask, axis):
        """
        Determine the first and last indices along a given axis where the mask is True.
        """
        projection = np.any(mask, axis=tuple(i for i in range(mask.ndim) if i != axis))
        first_index = np.argmax(projection)
        last_index = len(projection) - np.argmax(projection[::-1]) - 1
        
        return first_index, last_index

    def _shrink_edges(self, perc=None):
        """Shrink the furthest edges towards the target, ensuring at least N solutions."""
        if perc is None:
            perc = 1.0

        occupied_mask = self.occupied_mask
        
        changes_made = True
        # Until no possible changes can be made...
        while changes_made:
            changes_made = False
            # Get a list of axes and edges sorted by distance, furthest first
            edge_list = [(axis, edge, self._edge_distances[axis][edge]) 
                         for axis in range(len(self._dims)) 
                         for edge in ['start', 'end']]
            sorted_edges = sorted(edge_list, key=lambda x: x[2], reverse=True)

            for axis, edge, distance in sorted_edges:
                if distance > 0:  # Potential to shrink
                    # Calculate the minimum edge boundary based on percentage
                    target_bounds = self.target_bounds_by_axis[axis]
                    full_dim = self._dims[axis]
                    if edge == 'start':
                        extra_space_start = target_bounds[0] - 0  # Distance from the start of the full dimension to the start of the target
                        min_edge_limit = int(extra_space_start * perc)
                    else:  # 'end'
                        extra_space_end = full_dim - target_bounds[1]  # Distance from the end of the target to the end of the full dimension
                        min_edge_limit = full_dim - int(extra_space_end * perc)

                    new_boundary = self._edge_boundaries[axis][edge] + (1 if edge == 'start' else -1)
                    if (edge == 'start' and new_boundary <= min_edge_limit) or (edge == 'end' and new_boundary >= min_edge_limit):
                        if self._attempt_shrink_edge(axis, edge, occupied_mask):
                            print('GridArchive._shrink_edges > shrank:', axis, edge)
                            changes_made = True
                            self._edge_boundaries[axis][edge] = new_boundary
                            self._edge_distances[axis][edge] -= 1
                            break  # Exit the loop to start over
                        print('GridArchive._shrink_edges > no shrink b.c. min population constraint:', axis, edge)
                    else:
                        print('GridArchive._shrink_edges > no shrink b.c. percentage constraint:', axis, edge, perc)
                
                    # if self._attempt_shrink_edge(axis, edge, occupied_mask):
                    #     print('GridArchive._shrink_edges > shrank:', axis, edge)
                    #     changes_made = True
                    #     if edge == 'start':
                    #         self._edge_boundaries[axis][edge] += 1
                    #     else:  # 'end'
                    #         self._edge_boundaries[axis][edge] -= 1
                    #     self._edge_distances[axis][edge] -= 1
                    #     break  # Exit the loop to start over with re-evaluated and re-sorted edges

            # Check for match with the target mask after each round of attempts
            if np.array_equal(self._sample_mask_multid, self._target_sample_mask_multid):
                self.target_reached = True
                break

    def _attempt_shrink_edge(self, axis, edge, occupied_mask):
        """Attempt to shrink the specified edge, return True if successful."""
        boundary_idx = self._edge_boundaries[axis][edge]

        slice_obj = [slice(None)] * len(self._dims)
        slice_obj[axis] = slice(boundary_idx, boundary_idx + 1)

        test_mask = self._sample_mask_multid.copy()
        test_mask[tuple(slice_obj)] = False

        # Check 
        new_num_sols = np.sum(test_mask & occupied_mask)
        enough_sols_remain = new_num_sols >= self._sample_mask_min_solutions

        # NOTE: We used to take the difference into account when we shrunk.
        # difference = new_num_sols - np.sum(self._sample_mask & occupied_mask)
        # if difference != 0 and enough_sols_remain:
        
        if enough_sols_remain:
            self._sample_mask_multid = test_mask
            return True
        return False

    def get_mask_bounds(self):
        """
        Calculate the start and end indices for each axis of the current sample mask,
        indicating the bounds of the 'occupied' region.

        Returns:
            A dictionary with axis numbers as keys and tuples (start, end) as values,
            representing the start and end indices along each axis.
        """
        bounds = {}
        # Use len(self._dims) to iterate over each dimension of the array
        for axis in range(len(self._dims)):
            # Project the mask down to this axis to find where it's 'True'
            projection = np.any(self._sample_mask_multid, axis=tuple(i for i in range(len(self._dims)) if i != axis))

            # Find the first and last True value along the projection
            start_index = np.argmax(projection)
            end_index = len(projection) - np.argmax(projection[::-1]) - 1

            # Check for the case where the mask might be all False along this projection
            if not projection.any():  # If there's no True value in the projection
                start_index = end_index = None  # Indicate an empty or non-occupied mask along this axis

            bounds[axis] = (start_index, end_index)

        return bounds

    def _get_sparsity_reweighted_log_prior_weights(self):
        """Reweights the sample weights based on sparsity."""
        print('Performing sparsity reweighting...')

        # NOTE: We're abandoning this functionality for now
        return self._log_prior_weights

        log_prior_weights = self._log_prior_weights.copy()
        occupied_indices = self.occupied_indices
        unoccupied_indices = self.unoccupied_indices
        
        # Create a KDTree for quick nearest neighbor lookup
        tree = KDTree(self.int_to_grid_index(occupied_indices))
        
        # For each unoccupied cell, find the 50 nearest occupied neighbors
        _, nearest_indices = tree.query(self.int_to_grid_index(unoccupied_indices), k=50)
        
        # Convert nearest_indices from tree index space back to the flat grid space 
        actual_nearest_indices = occupied_indices[nearest_indices]

        # Convert log weights of unoccupied indices to linear space and divide evenly
        weights_to_distribute = np.exp(self._log_prior_weights[unoccupied_indices]) / 50

        # Initialize an array to store additional weights for occupied cells
        additional_weights = np.zeros_like(log_prior_weights, dtype=float)

        # Use numpy's advanced indexing to add weights to all neighbors at once
        np.add.at(additional_weights, actual_nearest_indices, weights_to_distribute[:, None])

        # Convert additional_weights back to log space and add to original log_prior_weights
        # Using log1p for numerical stability when converting small probabilities
        additional_log_weights = np.log1p(additional_weights)
        log_prior_weights += additional_log_weights

        # No normalization here - assuming it is handled elsewhere
        return log_prior_weights

    def update_sample_weights(self, seed_weights, seed2index, weight_sum_to_check):
        """ Updates the sample weights for the archive.

        Args:
            seed_weights (dict): A dictionary mapping seeds to their weights.
            seed2index (dict): A dictionary mapping seeds to their indices in
                the archive.
        """
        pprefix = 'GridArchive.update_sample_weights >'
        occupied_indices = self.occupied_indices
        unoccupied_indices = self.unoccupied_indices
        print(f'{pprefix} Updating sample weights (SW)...')

        ## 1) Set the new sample weights
        # Reset and update the sample weights with new values.
        if seed_weights is None:
            self._sample_weights = np.ones(self._cells, dtype=self.dtype)
        else:
            self._sample_weights = np.zeros(self._cells, dtype=self.dtype)
            for seed, weight in seed_weights.items():
                self._sample_weights[seed2index[seed]] = weight
            # Check that the sum of the sample weights is as expected
            if not np.isclose(np.sum(self._sample_weights), weight_sum_to_check):
                print(f'{pprefix} WARNING: Sample weights sums not close:')
                print('\t> sum of sample weights:', np.sum(self._sample_weights)) 
                print('\t> weight_sum_to_check:  ', weight_sum_to_check)
        # If min is the same as max, then we just do uniform sampling for efficiency
        max_sample_weights = np.max(self._sample_weights[occupied_indices])
        min_sample_weights = np.min(self._sample_weights[occupied_indices])
        if max_sample_weights == min_sample_weights:
            print(f'{pprefix} Uniform weights from PLR')
            self._uniform_sampling = True
        else:
            print(f'{pprefix} Non-uniform weights from PLR')
            self._uniform_sampling = False
        # Convert sample_weights to log scale for numerical stability
        # in the following computations (ignoring division )
        old_settings = np.seterr()
        np.seterr(divide='ignore')
        log_sample_weights = np.log(self._sample_weights)
        np.seterr(**old_settings)

        ## 2) Update sample mask
        # Check how many solutions in target mask region
        occupied_mask = self.occupied_mask
        self.num_sols_in_target = np.sum(self._target_sample_mask_multid & occupied_mask)
        self.target_percentage_covered = self.num_sols_in_target / self.target_size
        print(f'{pprefix} num_sols_in_target:', self.num_sols_in_target)

        ## 3) Apply prior, sparsity reweighting, and sample mask
        if self._use_prior:
            # Sparsity reweighting
            if self._sparsity_reweighting:
                log_prior_weights = self._get_sparsity_reweighted_log_prior_weights()
            else:
                log_prior_weights = self._log_prior_weights
            # Apply prior
            self._uniform_sampling = False
            log_sample_weights += log_prior_weights
        # Convert back to linear scale (-infs should be zeros)
        max_log_weight = np.max(log_sample_weights)
        normalized_log_weights = log_sample_weights - max_log_weight
        log_prob_threshold = -100  # Corresponding to a very small probability
        thresholded_log_weights = np.clip(normalized_log_weights, a_min=log_prob_threshold, a_max=None)
        self._sample_weights = np.exp(thresholded_log_weights)
        # Apply the sample mask and normalize
        self._sample_weights = self._sample_weights * self._sample_mask
        print(f'{pprefix} Sum SM:', np.sum(self._sample_mask))

        # Before setting unoccupied values to zero, check what each sums are respectively
        sum_occupied = np.sum(self._sample_weights[occupied_indices])
        sum_unoccupied = np.sum(self._sample_weights[unoccupied_indices])
        print(f'{pprefix} Sum SW occupied:', sum_occupied)
        print(f'{pprefix} Sum SW unoccupied (before zeroing):', sum_unoccupied)
        # Set unoccupied values to zero
        self._sample_weights[unoccupied_indices] = 0  # CRUCIAL

        # Re-normalize the sample weights
        sws = np.sum(self._sample_weights)
        if sws > 0:
            self._sample_weights = self._sample_weights / sws

        print(f'{pprefix} Sum SW:', np.sum(self._sample_weights))
        print(f'{pprefix} Sum SW unoccupied after zeroing:', np.sum(self._sample_weights[unoccupied_indices]))
        assert np.isclose(np.sum(self._sample_weights), 1.0)
        assert np.sum(self._sample_weights[unoccupied_indices]) == 0
        
        ## 4) Update the reduced sampling arrays for efficiency
        # Reduce sampling array size to make sampling faster
        self._indices_for_sampling_reduced = self._indices_for_sampling[self._sample_weights > 0]
        self._sample_weights_reduced = self._sample_weights[self._sample_weights > 0]