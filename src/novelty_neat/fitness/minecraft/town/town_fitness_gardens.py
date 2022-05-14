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

class Town_Gardens_Fitness(NeatFitnessFunction):
    def __init__(self, 
                 number_of_levels_to_generate: int, 
                 level_gen: NeatLevelGenerator):
        """
            Town_Gardens_Fitness:
                Maximise garden tiles
                Up to some limit
                More than 2 connected ones,
                
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
            GARDEN = level.tile_types_reversed['garden']
            where_is_gd = M == GARDEN
            new = np.copy(M)
            new[where_is_gd] = 1
            new[~where_is_gd] = 0
            
            labelled = morph.label(new, connectivity=2)

            counts = get_counts_of_array(labelled) 

            if len(counts) == 0:
                return 0
            if sum(counts) > M.size / 6: 
                return 1 - (sum(counts) - M.size / 6) / (M.size - M.size / 6)
            
            if 2 <= len(counts) <= 4: 
                return 1 if np.min(counts) >= 2 else 0.5
            return np.clip(len(counts) - 4 / 20, 0, 0.5)
        
        for group in mylevels:
            d = 0
            for l in group:
                d += calc_fitness(l)
            final_answer.append(d / len(group))
            assert 0 <= final_answer[-1] <= 1, final_answer[-1]
        
        return final_answer

    def __repr__(self) -> str:
        return f"Town_Gardens_Fitness(number_of_levels_to_generate={self.number_of_levels}, level_gen={self.level_gen})"
