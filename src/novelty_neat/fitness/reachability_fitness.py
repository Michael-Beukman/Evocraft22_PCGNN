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

class ReachabilityFitness(NeatFitnessFunction):
    def __init__(self, number_of_levels_to_generate: int, level_gen: NeatLevelGenerator):
        """Reward a specific number of connected regions
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
            labelled = morph.label(1 - level.map, connectivity=1)
            alls = (set(np.unique(labelled)) - {0})
            return len(alls)
        
        for group in mylevels:
            d = 0
            K = 0
            for l in group:
                current = calc_dist(l)
                # we want current == 1
                dist = (2 * (current - 1)) ** 2
                dist = 1/dist if dist != 0 else 1
                K += (dist)
            final_answer.append(K / len(group))
            assert 0 <= final_answer[-1] <= 1
        
        return final_answer

    def __repr__(self) -> str:
        return f"ReachabilityFitness(number_of_levels_to_generate={self.number_of_levels}, level_gen={self.level_gen})"