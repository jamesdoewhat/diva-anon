import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import KDTree

from qd_metarl.qd.archives import FlatArchive
from qd_metarl.qd.emitters import MapElitesCustomEmitter
from qd_metarl.qd.emitters import EvolutionStrategyCustomEmitter
from ribs.schedulers import Scheduler


def create_neg_env_scheduler(
    genotype_size=1,
    seed=123,
    cells=10_000,
    emitter_type='es',
    initial_population_num=10,
    initial_params_list=None,
    batch_size=5,
    genotype_bounds=None,
    mutation_percentage=0.05):
    """ Create a QD scheduler for negative environment sampling. """

    # Define archive
    archive = FlatArchive(
        solution_dim=genotype_size,
        cells=cells,
        threshold_min=-0.00001,
        seed=seed,
    )
    
    # Create emitters
    if emitter_type == 'es':
        emitters = [
            EvolutionStrategyCustomEmitter(
                archive,
                initial_params_list[i],
                bounds = genotype_bounds,
                seed=seed+i,
                batch_size=batch_size,
                initial_population=initial_population_num
            ) for i in range(5)  # Create 5 separate emitters.
        ]
    elif emitter_type == 'me':
        emitters = [
            MapElitesCustomEmitter(
                archive,
                initial_params_list[i].flatten(),
                bounds=genotype_bounds,
                seed=seed+i,
                batch_size=batch_size,
                initial_population=initial_population_num,
                mutation_k=int(genotype_size * mutation_percentage)
            ) for i in range(5)  # Create 5 separate emitters.
        ]
    else:
        raise ValueError(f'Invalid emitter type: {emitter_type}')

    # Create scheduler
    scheduler = Scheduler(archive, emitters)

    return scheduler


class NoveltySearch:
    def __init__(self, 
                 k_nearest, 
                 feature_extractor,
                 scheduler,
                 process_genotype_fn,
                 is_valid_genotype_fn,
                 standardize_genotype_fn):
        self.k_nearest = k_nearest
        self.feature_extractor = feature_extractor  # Function to extract features from a solution
        self.scheduler = scheduler
        self.sol2features = dict()
        self.process_genotype_fn = process_genotype_fn
        self.is_valid_genotype_fn = is_valid_genotype_fn
        self.standardize_genotype_fn = standardize_genotype_fn

    def calculate_novelty(self, solution, archive_features):
        solution = self.standardize_genotype_fn(solution)
        if solution in self.sol2features:
            features = self.sol2features[solution]
        else:
            features = self.feature_extractor(solution)
            self.sol2features[solution] = features
        if len(self.sol2features) == 1:
            return 0.0, features
        print('archive_features', archive_features)
        print('features', features)
        distances = cdist([features], archive_features, metric='euclidean').flatten()
        k_nearest_distances = np.partition(distances, self.k_nearest)[:self.k_nearest]
        novelty_score = np.mean(k_nearest_distances)
        return novelty_score, features

    def evaluate(self, solutions):
        # Create current features array
        archive_sols = self.scheduler.archive._solution_arr[
                                        :self.scheduler.archive._num_occupied]
        archive_features = np.array([self.sol2features[sol] 
                                     for sol in archive_sols])
        scores = []
        all_features = []
        for solution in solutions:
            score, features = self.calculate_novelty(solution, archive_features)
            scores.append(score)
            all_features.append(features)
        return scores, all_features
    
    def run(self, iterations):
        for _ in range(iterations):
            sols = self.scheduler.ask()
            validities=[]
            for sol in sols:
                pg = self.process_genotype_fn(sol)
                valid, _ = self.is_valid_genotype_fn(pg)
                validities.append(valid)

            scores, features = self.evaluate(sols)

            # Create combined scores with invalid scores set to -10,000
            combined_scores = []
            for sol, score, valid in zip(sols, scores, validities):
                if valid:
                    combined_scores.append(score)
                else:
                    combined_scores.append(-10_000)
            
            # Tell scheduler the scores
            self.scheduler.tell(combined_scores, features)

        solutions = self.scheduler.archive._solution_arr[
                                        :self.scheduler.archive._num_occupied]
        return solutions