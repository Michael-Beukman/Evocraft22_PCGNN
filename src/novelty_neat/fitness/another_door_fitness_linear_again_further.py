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

class AnotherDoorFitnessLinearFurther(NeatFitnessFunction):
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
            def is_good_roof_or_ground(single_face: np.ndarray):
                how_many_air_tiles = (1 - single_face).sum()
                if how_many_air_tiles == 0:
                    return 1
                else:
                    return 1 / how_many_air_tiles**1

            def has_door(single_face: np.ndarray):
                how_many_tiles = (1 - single_face).sum()
                A = morph.label(1 - single_face, connectivity=1)
                counts = get_counts_of_array(A)
                
                # reduce the free space to two
                if how_many_tiles == 2: 
                    how_many_good = 1
                elif how_many_tiles < 2:
                    how_many_good = 0
                else:
                    how_many_good = 1 / how_many_tiles**1
                

                # allow only two free blocks per set
                ONLY_TWO = 0

                for cnt in counts:
                    if cnt >= 2: ONLY_TWO += 1

                if len(counts) > 0:
                    ONLY_TWO = ONLY_TWO / len(counts)**1
                

                # reduce the number of door sets
                if len(counts) == 1:
                    how_many_things = 1
                elif len(counts) < 1:
                    how_many_things = 0
                else:
                    how_many_things = 1 / len(counts)**1

                # all doors must be vertical
                vertical_fitness = []

                # all doors must be grounded???
                grounded_fitness = []

                for label in np.unique(A):
                    if label ==  0:
                        continue

                    # 1. VERTICAL FITNESS CALCULATION

                    possible_door_masked_out = np.where(A == label, 1, 0)
                    temp = np.sum(possible_door_masked_out, axis=1)
                    temp = np.clip(temp, 0, 1)

                    if sum(temp) == 1:
                        vertical_fitness.append(1)
                    else:
                        vertical_fitness.append(((len(temp) - sum(temp))/len(temp)))

                    # 2. GROUNDED FITNESS CALCULATION

                    # get minimum y-coordinate of a set of air blocks
                    # we are using max function because indexing is the inverse of height
                    y_cord = max([y for y in range(len(A)) if sum(A[y]) > 0])

                    # coord_of_air_block(row index of the closest air block to ground) / ground_coord(max row index)
                    grounded_fitness.append(y_cord/len(A))


                if len(vertical_fitness) == 0:
                    vertical_fitness = 0
                else:
                    vertical_fitness = np.mean(vertical_fitness)

                if len(grounded_fitness) == 0:
                    grounded_fitness = 0
                else:
                    grounded_fitness = np.mean(grounded_fitness) 

                    
                
                
                return (grounded_fitness + vertical_fitness + how_many_things + how_many_good + ONLY_TWO) / 5
                

                
            
            def dothings(A, reverse=False):
                return (has_door(A))
            
            def dothings_bad(A):
                return (1 - A).mean() # how many air blocks -- more air is bad.
            
            M = level.map
            ans = 0
            for k in [0, -1]:
                ans += dothings(M[:, -3:, k])
                ans += dothings(M[k, -3:, :].T)
                
                ans += is_good_roof_or_ground(M[:, k, :]) # fitness for good rough and ground. NO DOORS, all closed
                ans -= (1 - M[k, :, :].mean())
            ans /= 6
            return np.clip(ans, 0, 1)
                    
        
        for group in mylevels:
            d = 0
            for l in group:
                d += calc_fitness(l)
            final_answer.append(d / len(group))
            assert 0 <= final_answer[-1] <= 1
        
        return final_answer

    def __repr__(self) -> str:
        return f"AnotherDoorFitnessLinearFurther(number_of_levels_to_generate={self.number_of_levels}, level_gen={self.level_gen})"
