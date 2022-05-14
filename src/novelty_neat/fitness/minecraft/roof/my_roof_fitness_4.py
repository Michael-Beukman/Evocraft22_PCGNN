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

class MyRoofFitnessV4(NeatFitnessFunction):
    def __init__(self, 
                 number_of_levels_to_generate: int, 
                 level_gen: NeatLevelGenerator):
        """
            RoofFitness
                Is triangle?
                Does cover entire ground?
                1-Connected
        """
        super().__init__(number_of_levels_to_generate=number_of_levels_to_generate, level_gen=level_gen)

    def calc_fitness(self, nets: List[LevelNeuralNet], levels: List[List[Level]]) -> List[float]:

        final_answer = []
        # If we already generated enough levels, then we can just use those.
        # Otherwise we should use others.
        assert len(levels) and len(levels[0]) == self.number_of_levels
        mylevels = levels

        
        def calc_fitness(level: Level):
            M = level.map
            
            how_much_cover = (M.sum(axis=1) > 0).mean()
            # bads = (M.sum(axis=1) > 2).mean()
            how_much_connected = morph.label(M, connectivity=1)
            list_of_things = get_counts_of_array(how_much_connected)
            L = len(list_of_things)
            if L <= 1:
                how_much_connected = L
            else:
                how_much_connected = 1 / L ** 2

            how_many_tiles = M.mean()
            
            return (2 * how_much_cover + how_much_connected + 1*(1 - how_many_tiles)) / 4
            
        
        for group in mylevels:
            d = 0
            for l in group:
                d += calc_fitness(l)
            final_answer.append(d / len(group))
            assert 0 <= final_answer[-1] <= 1
        
        return final_answer

    def __repr__(self) -> str:
        return f"MyRoofFitnessV4(number_of_levels_to_generate={self.number_of_levels}, level_gen={self.level_gen})"
