from math import ceil
from typing import Dict, List

import numpy as np
from games.level import Level
from games.maze.maze_game import MazeGame
from games.maze.maze_level import MazeLevel
from novelty_neat.fitness.fitness import NeatFitnessFunction
from novelty_neat.fitness.minecraft.mcraftutils import get_counts_of_array
from novelty_neat.generation import NeatLevelGenerator
from novelty_neat.maze.neat_maze_level_generation import GenerateMazeLevelsUsingTiling
from novelty_neat.types import LevelNeuralNet
import scipy.stats as stats
from scipy.spatial import distance
import skimage.morphology as morph
import math

class NoSingleBlocksMultipleTilesFitness(NeatFitnessFunction):
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
                A = morph.label(single_face == 0, connectivity=1)
                counts = get_counts_of_array(A)
                if len(counts) == 0: return 0
                C = min(counts)
                if C <= 1: return 0.1
                elif 2 <= C <= 4: return 1
                else: return 1 / C

            
            M = level.map
            ans = 0
            for k in [0, -1]:
                ans += has_door(M[:, :, k])
                ans += has_door(M[k, :, :].T)

            ans /= 4
            return np.clip(ans, 0, 1)
                    
        
        for group in mylevels:
            d = 0
            for l in group:
                d += calc_fitness(l)
            final_answer.append(d / len(group))
            assert 0 <= final_answer[-1] <= 1
        
        return final_answer

    def __repr__(self) -> str:
        return f"NoSingleBlocksMultipleTilesFitness(number_of_levels_to_generate={self.number_of_levels}, level_gen={self.level_gen})"
