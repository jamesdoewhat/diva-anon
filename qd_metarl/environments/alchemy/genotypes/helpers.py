import itertools

from dm_alchemy.types.graphs import possible_constraints
from qd_metarl.environments.alchemy.alchemy_gym import graph_list
from dm_alchemy.types.graphs import no_bottleneck_constraints
from dm_alchemy.types.graphs import bottleneck1_constraints
from dm_alchemy.types.graphs import bottleneck2_constraints
from dm_alchemy.types.graphs import bottleneck3_constraints
from dm_alchemy.types.graphs import possible_constraints


_PERMUTATIONS = list(itertools.permutations([0, 1, 2]))
IDX_TO_PERMUTATION = {i: p for i, p in enumerate(_PERMUTATIONS)}
PERMUTATION_TO_IDX = {p: i for i, p in IDX_TO_PERMUTATION.items()}

IDX_TO_GRAPH_CONSTRAINTS = {
    0: no_bottleneck_constraints(),
    1: bottleneck1_constraints(),
    2: bottleneck2_constraints(),
    3: bottleneck3_constraints(),
}

GRAPH_LIST = graph_list(possible_constraints())