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

class Town_Reachable_FitnessV2(NeatFitnessFunction):
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
            # See if all 
            HOUSE = level.tile_types_reversed['house']
            ROAD  = level.tile_types_reversed['road']
            # Now, 
            where_is_house = M == HOUSE
            new = np.copy(M)
            new[where_is_house] = ROAD
            where_is_road = new == ROAD
            new[~where_is_road] = 0
            new[where_is_road] = 1
            
            labelled = morph.label(new, connectivity=1)
            # Now, we must reward having labelled just being one joined thing. I.e. penalise multiple different things.
            counts = get_counts_of_array(labelled) 
            if len(counts) == 0: return 0
            if len(counts) == 1: return 1
            if len(counts) >= 8: return 1 / len(counts)        

            # Now, this is between 2 and 7.
            return 1.5 * np.max(counts) / np.sum(counts) / len(counts)
        
        for group in mylevels:
            d = 0
            for l in group:
                d += calc_fitness(l)
            final_answer.append(d / len(group))
            assert 0 <= final_answer[-1] <= 1
        
        return final_answer

    def __repr__(self) -> str:
        return f"Town_Reachable_FitnessV2(number_of_levels_to_generate={self.number_of_levels}, level_gen={self.level_gen})"
