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

class DecorationFitnessV5(NeatFitnessFunction):
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
                

        def penalise_min_max(a, mins, maxs):
            if mins <= a <= maxs: return 1
            if a < mins: return a / max(mins, 1) / 2
            if a > maxs: return np.clip(0.5 / (a - maxs), 0, 1)
            return 0
            
        
        def calc_fitness(level: Level):
            M = level.map
            how_much_air = (M == 0).mean()
            
            how_much_air = np.clip(how_much_air / 0.95, 0, 1)
            how_many_torches = 1 - (M[1:-1, :, 1:-1] == 1).mean()
            
            L = M > 1
            L[:, -1, :] = True
            labelled = morph.label(L, connectivity=1)
            counts = len(get_counts_of_array(labelled))
            
            L = M > 0
            L[:, -1, :] = True
            labelled = morph.label(L, connectivity=1)
            counts2 = len(get_counts_of_array(labelled))

            S1 = counts if counts <= 1 else (0.5 / counts) ** 2
            S2 = counts2 if counts2 <= 1 else (0.5 / counts2) ** 2

            
            A = penalise_min_max((M == 1).sum(), 3, 6)
            B = penalise_min_max((M == 2).sum(), 0, 6)
            C = penalise_min_max((M == 3).sum(), 0, 2)
            D = penalise_min_max((M == 4).sum(), 0, 2)
            
            return (how_much_air + how_many_torches+ A + B + C + D + S1 + S2) / 8
            
                    
        
        for group in mylevels:
            d = 0
            for l in group:
                d += calc_fitness(l)
            final_answer.append(d / len(group))
            assert 0 <= final_answer[-1] <= 1
        
        return final_answer

    def __repr__(self) -> str:
        return f"DecorationFitnessV5(number_of_levels_to_generate={self.number_of_levels}, level_gen={self.level_gen})"
