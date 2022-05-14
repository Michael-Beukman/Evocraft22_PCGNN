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

class DoorFitness(NeatFitnessFunction):
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

            def has_good_door(single_face: np.ndarray):
                how_many_tiles = (1 - single_face).sum()
                
                # reduce the free space to two
                if how_many_tiles == 2: 
                    how_many_good = 1
                elif how_many_tiles == 1:
                    how_many_good = 0.7
                elif how_many_tiles == 0:
                    how_many_good = 0.33
                else:
                    how_many_good = 1 / how_many_tiles

                return how_many_good
                
            
            M = level.map
            ans = []
            for k in [0, -1]:
                ans.append(has_good_door(M[:, :, k]))
                ans.append(has_good_door(M[k, :, :]))

            return np.clip(np.mean(ans), 0, 1)
                    
        
        for group in mylevels:
            d = 0
            for l in group:
                d += calc_fitness(l)
            final_answer.append(d / len(group))
            assert 0 <= final_answer[-1] <= 1
        
        return final_answer

    def __repr__(self) -> str:
        return f"AnotherDoorFitness(number_of_levels_to_generate={self.number_of_levels}, level_gen={self.level_gen})"
