from math import ceil
from typing import Dict, List

import numpy as np
import scipy
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

class DecorationFitness(NeatFitnessFunction):
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
            M = level.map
            how_much_air = (M == 0).mean()
            
            how_much_air = np.clip(how_much_air / 0.9, 0, 1)
            how_many_torches = 1 - (M[1:-1, :, 1:-1] == 1).mean()
            
            L = (np.logical_and(M == 2, M == 3))
            L[:, -1, :] = True
            labelled = morph.label(L, connectivity=1)
            counts = get_counts_of_array(labelled)
            good_counts = 0
            if len(counts) == 0: good_counts = 0.25
            elif len(counts) == 1: good_counts = 1
            else: good_counts = 1 / len(counts)
            
            
            return (how_much_air + how_many_torches + good_counts) / 3
            
                    
        
        for group in mylevels:
            d = 0
            for l in group:
                d += calc_fitness(l)
            final_answer.append(d / len(group))
            assert 0 <= final_answer[-1] <= 1
        
        return final_answer

    def __repr__(self) -> str:
        return f"DecorationFitness(number_of_levels_to_generate={self.number_of_levels}, level_gen={self.level_gen})"
