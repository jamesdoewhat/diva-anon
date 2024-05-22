# Modified from https://github.com/john-science/mazelib/blob/main/mazelib/generate/BacktrackingGenerator.py
import numpy as np
import matplotlib.pyplot as plt
from mazelib.generate.MazeGenAlgo import MazeGenAlgo


def showPNG(grid):
    """Generate a simple image of the maze."""
    plt.figure(figsize=(10, 5))
    plt.imshow(grid, cmap=plt.cm.binary, interpolation='nearest')
    plt.xticks([]), plt.yticks([])
    plt.show()


class CustomMazeGenerator(MazeGenAlgo):
    """
    Same as BacktrackingGenerator from mazelib, but uses a local prng instead
    of random and np.random global rng states.

    NOTE: Produces (w*2+1, w*2+1) shaped bitmap for grid. Thus, for desired
    size of e.g. 15, one should set w=h=7.
    """

    def __init__(self, w, h, seed=None):
        super(CustomMazeGenerator, self).__init__(w, h)
        self.prng = np.random.RandomState(seed)
        self.seed = seed
        # print('## ENVIRONMENT INIT WITH SEED: {}'.format(self.seed))

    def reset_seed(self, seed):
        self.prng = np.random.RandomState(seed)
        self.seed = seed
        # print('#### RESETTING STATE; seed: {}; random state: {}'.format(
        #     self.seed, self.prng.get_state()[1][-1]))

    def generate(self):
        """Same as original function, but uses prng.randint instead of 
        randrange.
        """
        # create empty grid, with walls
        # print('## GENERATING WITH PRNG: {}'.format(self.prng.get_state()[1][-1]))
        grid = np.empty((self.H, self.W), dtype=np.int8)
        grid.fill(1)

        # crow = randrange(1, self.H, 2)
        crow = 1 + self.prng.randint(0, self.H//2) * 2
        # ccol = randrange(1, self.W, 2)
        ccol = 1 + self.prng.randint(0, self.W//2) * 2
        track = [(crow, ccol)]
        grid[crow][ccol] = 0

        while track:
            (crow, ccol) = track[-1]
            neighbors = self._find_neighbors(crow, ccol, grid, True)

            if len(neighbors) == 0:
                track = track[:-1]
            else:
                nrow, ncol = neighbors[0]
                grid[nrow][ncol] = 0
                grid[(nrow + crow) // 2][(ncol + ccol) // 2] = 0

                track += [(nrow, ncol)]

        return grid
    
    def generate_bitmap(self):
        """
        Generates bitmap in style expected from MazeEnv. Specifically,
        we do not include outer walls in bitmap.
        """
        grid = self.generate()
        return grid[1:-1, 1:-1]

    def _find_neighbors(self, r, c, grid, is_wall=False):
        """Same functionality as original but using local prng for shuffle."""
        ns = []

        if r > 1 and grid[r - 2][c] == is_wall:
            ns.append((r - 2, c))
        if r < self.H - 2 and grid[r + 2][c] == is_wall:
            ns.append((r + 2, c))
        if c > 1 and grid[r][c - 2] == is_wall:
            ns.append((r, c - 2))
        if c < self.W - 2 and grid[r][c + 2] == is_wall:
            ns.append((r, c + 2))

        self.prng.shuffle(ns)
        
        return ns
    

if __name__ == '__main__':
    
    # Testing functionality
    maze = CustomMazeGenerator(7, 7, seed=123)
    grid = maze.generate()
    bitmap = maze.generate_bitmap()

    # Print generated grid
    print('\nGrid:\n\n', grid)

    # Print generated bitmap
    print('\nBitmap:\n\n', bitmap)

    showPNG(grid)


    