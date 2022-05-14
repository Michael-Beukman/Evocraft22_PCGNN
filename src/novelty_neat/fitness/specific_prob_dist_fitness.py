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

class SpecificProbabilityDistributionFitness(NeatFitnessFunction):
    def __init__(self, number_of_levels_to_generate: int, level_gen: NeatLevelGenerator, desired_dist: Dict[str, float] = None):
        """This is the constructor for the entropy fitness function.
        """
        super().__init__(
            number_of_levels_to_generate=number_of_levels_to_generate, level_gen=level_gen)
        self.desired_dist = desired_dist
        assert desired_dist is not None

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
        keys = list(self.desired_dist.keys())
        index_dic = {key: i for i, key in enumerate(keys)}
        desired_dist = np.array([self.desired_dist[k] for k in keys])
        def calc_dist(level: Level):
            this_dist = [0 for _ in keys]
            for tile in np.unique(level.map):
                this_name = keys[int(tile)] # level.tile_types[int(tile)]
                this_count = (level.map == tile).sum()
                this_dist[index_dic[this_name]] = this_count
            
            return np.array(this_dist)
        
        for group in mylevels:
            d = 0
            for l in group:
                d += calc_dist(l)
            d = d / d.sum()
            final_answer.append(
                1 - distance.jensenshannon(d, desired_dist)
            )
            assert 0 <= final_answer[-1] <= 1
        
        return final_answer

    def __repr__(self) -> str:
        return f"SpecificProbabilityDistributionFitness(number_of_levels_to_generate={self.number_of_levels}, level_gen={self.level_gen}, desired_dist={self.desired_dist})"


if __name__ == '__main__':
    func = EntropyFitness(5, GenerateMazeLevelsUsingTiling(game=MazeGame(MazeLevel())))
    print(func.params())
