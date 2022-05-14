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

class TownLarger_ReachableMore_FitnessCoherence(NeatFitnessFunction):
    def __init__(self, 
                 number_of_levels_to_generate: int, 
                 level_gen: NeatLevelGenerator):
        """
            Town_Reachable_Fitness:
                Are all houses (1) connected by roads (3)
        """
        super().__init__(number_of_levels_to_generate=number_of_levels_to_generate, level_gen=level_gen)

    def calc_fitness(self, nets: List[LevelNeuralNet], levels: List[List[Level]]) -> List[float]:

        final_answer = []
        # If we already generated enough levels, then we can just use those.
        # Otherwise we should use others.
        assert len(levels) and len(levels[0]) == self.number_of_levels
        mylevels = levels

        
        # TEST = desired_walls.sum()
        
        def calc_fitness(level: Level):
            M = level.map
            assert M.shape[1] == 1, f"BAD: {M.shape}"
            M = M[:, 0, :]

            HOUSE1 = level.tile_types_reversed['house']
            HOUSE3 = level.tile_types_reversed['house_3']
            
            Ls = {
                h: len(get_counts_of_array(morph.label(M == h))) for h in range(HOUSE1, HOUSE3 + 1)
            }
            
            goods = 0
            mycounts = 0
            
            for h in range(HOUSE1, HOUSE3 + 1):
                for h2 in range(h + 1, HOUSE3 + 1):
                    mycounts += 1
                    temp = get_counts_of_array(morph.label(np.logical_and(M == h, M == h2), connectivity=1))
                    
                    best = Ls[h] + Ls[h2]
                    current = len(temp)
                    if best != 0: 
                        goods += current / best
            goods /= mycounts
                    
                    
    

            return goods
        
        for group in mylevels:
            d = 0
            for l in group:
                d += calc_fitness(l)
            final_answer.append(d / len(group))
            assert 0 <= final_answer[-1] <= 1
        
        return final_answer

    def __repr__(self) -> str:
        return f"TownLarger_ReachableMore_FitnessCoherence(number_of_levels_to_generate={self.number_of_levels}, level_gen={self.level_gen})"
