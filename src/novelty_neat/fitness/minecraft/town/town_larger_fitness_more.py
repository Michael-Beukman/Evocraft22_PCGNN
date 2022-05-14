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

class TownLarger_ReachableMore_Fitness(NeatFitnessFunction):
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
            ROAD  = level.tile_types_reversed['road']
            # Now, 
            HOUSE1 = level.tile_types_reversed['house']
            HOUSE3 = level.tile_types_reversed['house_3']
            where_is_house = np.logical_and(M >= HOUSE1, M <= HOUSE3)
            new = np.copy(M)
            new[where_is_house] = ROAD
            where_is_road = new == ROAD
            new[~where_is_road] = 0
            new[where_is_road] = 1
            
            labelled2 = morph.label((M == ROAD), connectivity=1)
            
            labelled = morph.label(new, connectivity=1)
            # Now, we must reward having labelled just being one joined thing. I.e. penalise multiple different things.
            counts2 = get_counts_of_array(labelled2) 
            
            
            counts = get_counts_of_array(labelled) 
            if len(counts) == 0: A = 0
            elif len(counts) == 1: A = 1
            elif len(counts) >= 8: A = 1 / len(counts)        
            else: A = 1.5 * np.max(counts) / np.sum(counts) / len(counts)
            
            if len(counts2) == 0: B = 0
            elif len(counts2) == 1: B = 1
            elif len(counts2) >= 8: B = 1 / len(counts2)        
            else: B = 1.5 * np.max(counts2) / np.sum(counts2) / len(counts2)

            # coherence
            
            Ls = {
                h: len(get_counts_of_array(morph.label(M == h))) for h in range(HOUSE1, HOUSE3 + 1)
            }
            goods = 0
            mycounts = 0
            for h in range(HOUSE1, HOUSE3 + 1):
                for h2 in range(h + 1, HOUSE3 + 1):
                    mycounts += 1
                    temp = get_counts_of_array(morph.label(np.logical_and(M == h, M == h2)))
                    
                    best = Ls[h] + Ls[h2]
                    current = len(temp)
                    if best != 0: 
                        goods += current / best
            goods /= mycounts
                    
                    
            
            # garden
            temp = np.argwhere(M == level.tile_types_reversed['garden']) # shape of (num_torch, 3)
            if len(temp) == 0:
                DIST = 0
            elif len(temp) == 1: DIST = 0.1
            else:
                mycount = 0
                my_dist = 0
                for i, loc in enumerate(temp[:-1]):
                    for j, loc2 in enumerate(temp[i+1:]):
                        my_dist += np.linalg.norm(loc - loc2)
                        mycount += 1
                # norm
                my_dist /= mycount
                my_dist /= np.sqrt(M.shape[0] ** 2 + M.shape[1] ** 2)
                my_dist = np.clip(my_dist, 0, 1)
                DIST = my_dist

            return (np.clip((A + B) / 2, 0, 1) + goods + DIST) / 3
        
        for group in mylevels:
            d = 0
            for l in group:
                d += calc_fitness(l)
            final_answer.append(d / len(group))
            assert 0 <= final_answer[-1] <= 1
        
        return final_answer

    def __repr__(self) -> str:
        return f"TownLarger_ReachableMore_Fitness(number_of_levels_to_generate={self.number_of_levels}, level_gen={self.level_gen})"
