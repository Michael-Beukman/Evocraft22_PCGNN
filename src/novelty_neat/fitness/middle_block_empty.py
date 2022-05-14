from math import ceil
from typing import Dict, List

import numpy as np
import scipy
from games.level import Level
from games.maze.maze_game import MazeGame
from games.maze.maze_level import MazeLevel
from novelty_neat.fitness.fitness import NeatFitnessFunction
from novelty_neat.generation import NeatLevelGenerator
from novelty_neat.maze.neat_maze_level_generation import GenerateMazeLevelsUsingTiling
from novelty_neat.types import LevelNeuralNet
import scipy.stats as stats
from scipy.spatial import distance
import skimage.morphology as morph

class MiddleBlockEmpty(NeatFitnessFunction):
    def __init__(self, number_of_levels_to_generate: int, level_gen: NeatLevelGenerator):
        """Good if middle block is open
        """
        super().__init__(
            number_of_levels_to_generate=number_of_levels_to_generate, level_gen=level_gen)

    def calc_fitness(self, nets: List[LevelNeuralNet], levels: List[List[Level]]) -> List[float]:

        final_answer = []
        # If we already generated enough levels, then we can just use those.
        # Otherwise we should use others.
        if len(levels) and len(levels[0]) == self.number_of_levels:
            mylevels = levels
        else:
            mylevels = []
            for net in nets:
                temp = []
                for i in range(self.number_of_levels):
                    temp.append(self.level_gen(net))
                mylevels.append(temp)
        
        def calc_dist(level: Level):
            ans = level.map
            w, h, d = ans.shape
            return np.isclose(ans[w//2, h//2, d//2], 0)
        
        for group in mylevels:
            d = 0
            K = 0
            for l in group:
                is_empty = calc_dist(l)
                K += is_empty
            final_answer.append(K / len(group))
            assert 0 <= final_answer[-1] <= 1
        
        return final_answer

    def __repr__(self) -> str:
        return f"MiddleBlockEmpty(number_of_levels_to_generate={self.number_of_levels}, level_gen={self.level_gen})"