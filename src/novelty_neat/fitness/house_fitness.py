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

class IsReasonableHouseFitness(NeatFitnessFunction):
    def __init__(self, number_of_levels_to_generate: int, level_gen: NeatLevelGenerator):
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
            # Walls
            desired_walls = np.zeros_like(level.map)
            for k in [0, -1]:
                desired_walls[k, :, :] = 1
                desired_walls[:, :, k] = 1
            desired_walls[:, 0, :] = 1
            
            how_many_good = morph.label(level.map, connectivity=1)
            
            uniques = np.unique(how_many_good)
            alls_ = (set(uniques) - {0})
            list_of_counts = []
            for j in alls_:
                list_of_counts.append((how_many_good == j).sum())
            if len(list_of_counts) != 0:
                how_many_things = max(list_of_counts) / sum(list_of_counts)
                how_many_things = (how_many_things) / len(list_of_counts)
            else:
                how_many_things = 0.5

            # How much does this overlap with 'correct' walls
            how_much_walls = (desired_walls == level.map)[desired_walls == 1].mean()
            
            # Air Gap -- This part must be largely air
            how_much_air_inside = (1 - level.map[1:-1, 1:-1, 1:-1]).mean()
            if 0.7 <= how_much_air_inside <= 0.9:
                how_much_air_inside = 1
            else:
                if how_much_air_inside <= 0.7:
                    how_much_air_inside = how_much_air_inside / 0.7
                else:
                    how_much_air_inside = 1 - (how_much_air_inside - 0.7) / 0.3
            
            # Air Gap, is the air indeed connected to each other (and the outside world)
            temp_level = np.zeros((level.width + 2, level.height + 2, level.depth + 2))
            temp_level[1:-1, 1:-1, 1:-1] = level.map
            ans = morph.label(1 - temp_level, connectivity=1)
            # if this is >1, then it is reachable
            is_reachable_outside = ans[0, 0, 0] == ans[level.width // 2, level.height // 2, level.depth // 2]
            uniques = np.unique(ans)
            alls_ = (set(uniques) - {0})
            list_of_counts = []
            for j in alls_:
                list_of_counts.append((ans == j).sum())
            
            if len(list_of_counts) == 0:
                how_much_reachable = 0
            else:
                how_much_reachable = max(list_of_counts) / sum(list_of_counts) / len(list_of_counts)
            
            
            return how_much_walls, how_many_things, how_much_air_inside, is_reachable_outside, how_much_reachable

        
        for group in mylevels:
            d = 0
            for l in group:
                walls, how_many_things, air, reach, how_much_reach = calc_fitness(l)
                
                d += (walls ** 2 + how_many_things ** 2 + 2 * air ** 2 + (reach / 2) ** 2 + how_much_reach ** 2) / 5.25
            final_answer.append(d / len(group))
            assert 0 <= final_answer[-1] <= 1
        
        return final_answer

    def __repr__(self) -> str:
        return f"IsReasonableHouseFitness(number_of_levels_to_generate={self.number_of_levels}, level_gen={self.level_gen})"
