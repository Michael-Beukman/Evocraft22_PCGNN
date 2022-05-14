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

class SimpleDoorFitness(NeatFitnessFunction):
    def __init__(self, 
                 number_of_levels_to_generate: int, 
                 level_gen: NeatLevelGenerator):
        super().__init__(number_of_levels_to_generate=number_of_levels_to_generate, level_gen=level_gen)

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
                

        def calc_fitness(level: Level):
            def has_door(single_face: np.ndarray):
                if single_face.size > np.sum(single_face) > single_face.size * 0.95:
                    # How many connected components?
                    A = morph.label(1 - single_face, connectivity=1)
                    num_doors = len(np.unique(A)) - 1
                    if 1 <= num_doors <= 1: return 1
                    return 0.5 if num_doors <= 3 else 0.25
                return 0
            
            def dothings(A, reverse=False):
                return (has_door(A) + (has_door(A[:, -4:]) if reverse else has_door(A[-4:, :]))) / 2
            
            M = level.map
            ans = 0
            for k in [0, -1]:
                ans += dothings(M[:, :, k])
                ans += dothings(M[k, :, :], True)
            ans /= 4
            return ans
                    
        
        for group in mylevels:
            d = 0
            for l in group:
                d += calc_fitness(l)
            final_answer.append(d / len(group))
            assert 0 <= final_answer[-1] <= 1
        
        return final_answer

    def __repr__(self) -> str:
        return f"SimpleDoorFitness(number_of_levels_to_generate={self.number_of_levels}, level_gen={self.level_gen})"
