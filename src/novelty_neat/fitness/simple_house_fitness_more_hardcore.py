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

class SimpleIsReasonableHouseHardcoreFitness(NeatFitnessFunction):
    def __init__(self, 
                 number_of_levels_to_generate: int, 
                 level_gen: NeatLevelGenerator, square=False):
        """
            Is this a reasonable house?
            Specifically, 
                - Does it have walls?
                - Does it have a door
                - Does it have an airgap in the middle of it?
                - Are all airgaps connected to each other.
                - Are there no freestanding walls?
        """
        super().__init__(number_of_levels_to_generate=number_of_levels_to_generate, level_gen=level_gen)
        self.square = square

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
                
        desired_walls = np.zeros_like(levels[0][0].map)
        
        for k in [0, -1]:
            desired_walls[k, :, :] = 1
            desired_walls[:, :, k] = 1
            desired_walls[:, k, :] = 1
        
        # TEST = desired_walls.sum()
        
        def calc_fitness(level: Level):
            # Walls
            A = (desired_walls == level.map).mean()
            if self.square: return A ** 2
            else: return A

        
        for group in mylevels:
            d = 0
            for l in group:
                d += calc_fitness(l)
            final_answer.append(d / len(group))
            assert 0 <= final_answer[-1] <= 1
        
        return final_answer

    def __repr__(self) -> str:
        return f"SimpleIsReasonableHouseHardcoreFitness{'_Square' if self.square else ''}(number_of_levels_to_generate={self.number_of_levels}, level_gen={self.level_gen})"
