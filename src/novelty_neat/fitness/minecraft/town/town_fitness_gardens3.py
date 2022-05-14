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

class Gardens_FitnessV3(NeatFitnessFunction):
    def __init__(self, 
                 number_of_levels_to_generate: int, 
                 level_gen: NeatLevelGenerator):
        """
            Gardens_Fitness:
                Maximise garden tiles
                Up to some limit
                More than 2 connected ones,
                
        """
        super().__init__(number_of_levels_to_generate=number_of_levels_to_generate, level_gen=level_gen)

    # function to have find all tree blocks in the grid
    def get_tree_tiles(self, level : Level, tree_ids : List, min_dist : int= 5,):
        # flat two_d 
        tree_locations = np.argwhere(np.logical_and(level.map >= 1, level.map <= 3))
        # print(len(tree_locations))
        # print(tree_locations)
        bad_trees = 0
        for i, tree in enumerate(tree_locations):
            for j, other_tree in enumerate(tree_locations):
                if i != j:
                    if np.linalg.norm(tree - other_tree) <  min_dist:
                        bad_trees += 1
                        break
                    
        return bad_trees


    def calc_fitness(self, nets: List[LevelNeuralNet], levels: List[List[Level]]) -> List[float]:

        final_answer = []
        # If we already generated enough levels, then we can just use those.
        # Otherwise we should use others.
        assert len(levels) and len(levels[0]) == self.number_of_levels
        mylevels = levels

        
        # TEST = desired_walls.sum()
        

        def calc_fitness(level: Level):
            min_prop_empty = 0.1 # how much free space is allowed at the minimum
            max_prop_full = 0.7 # how much of the space can be populated at maximum
            water_dampness = 0.05
            dirt, water = 0, 6
            
            sapling_types = [1, 2, 3] # set of ids to represent tree sapling types
            flower_types = [4, 5] # sset of idss to represent flower types
            num_trees = np.logical_and(level.map >= 1, level.map <= 3).sum() 
            num_flowers = np.logical_and(level.map >= 4, level.map <= 5).sum()
            # pretend dirt is blank space
            density_score = 0
            num_dirt = np.mean(level.map == dirt)
            if num_dirt > min_prop_empty and num_dirt < max_prop_full:
                density_score = 1
            num_bad_trees = self.get_tree_tiles(level, sapling_types, min_dist=6)
            tree_score = 0.2
            if num_trees != 0:
                tree_score = 1 - (num_bad_trees/num_trees)
            # then have some max proportion that is allowed to be populated e.g not dirt
            water_prop = np.mean(level.map == water)
            water_score = 0
            if water_prop > water_dampness:
                water_score = -1
            else:
                water_score = 1
            free_point = 1 # to prevent negative assert
            has_tree = 0
            has_flowers = 0
            if num_flowers > 0:
                has_flowers = 1
            if num_trees > 0:
                has_tree = 1

            return (density_score + tree_score + water_score + has_tree + has_flowers + free_point) / 6
            
        for group in mylevels:
            d = 0
            for l in group:
                d += calc_fitness(l)
            final_answer.append(d / len(group))
            assert 0 <= final_answer[-1] <= 1, final_answer[-1] 
        return final_answer



    def __repr__(self) -> str:
        return f"Gardens_FitnessV3(number_of_levels_to_generate={self.number_of_levels}, level_gen={self.level_gen})"
