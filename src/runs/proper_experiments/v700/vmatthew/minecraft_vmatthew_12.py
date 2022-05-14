from cmath import exp
from functools import partial
import os
from pprint import pprint
import sys
import threading
from typing import List
import wandb
from baselines.ga.genetic_algorithm_pcg import GeneticAlgorithmIndividual, GeneticAlgorithmIndividualMaze, GeneticAlgorithmPCG
from common.types import Verbosity
from common.utils import get_date
from experiments.experiment import Experiment
from experiments.config import Config
from games.minecraft.gardens.minecraft_game_garden import MinecraftGameGarden
from games.minecraft.gardens.minecraft_level_garden import MinecraftGardenLevel
from games.minecraft.minecraft_game import MinecraftGame
from games.minecraft.minecraft_level import MinecraftLevel
from games.minecraft.towns.minecraft_game_town import MinecraftGameTown
from games.minecraft.towns.minecraft_level_town import MinecraftTown
from metrics.a_star.a_star_metrics import AStarDifficultyMetric, AStarDiversityAndDifficultyMetric, AStarDiversityMetric, AStarEditDistanceDiversityMetric, AStarSolvabilityMetric
from metrics.average_pairwise_distance import AveragePairWiseDistanceMetric, AveragePairWiseDistanceMetricOnlyPlayable
from metrics.combination_metrics import RLAndSolvabilityMetric
from metrics.horn.compression_distance import CompressionDistanceMetric
from metrics.horn.leniency import LeniencyMetric
from metrics.horn.linearity import LinearityMetric
from metrics.rl.tabular.rl_agent_metric import RLAgentMetric
from metrics.rl.tabular.rl_difficulty_metric import RLDifficultyMetric
from metrics.solvability import SolvabilityMetric
import os
from novelty_neat.fitness.door_fitnesses import SimpleDoorFitness
from novelty_neat.fitness.entropy import EntropyFitness
from novelty_neat.fitness.fitness import CombinedFitness, NeatFitnessFunction
from novelty_neat.fitness.house_fitness import IsReasonableHouseFitness
from novelty_neat.fitness.middle_block_empty import MiddleBlockEmpty
from novelty_neat.fitness.minecraft.garden.town_fitness_gardens import Gardens_Fitness
from novelty_neat.fitness.minecraft.town.town_fitness import Town_Reachable_Fitness
from novelty_neat.fitness.minecraft.town.town_fitness_gardens import Town_Gardens_Fitness
from novelty_neat.fitness.minecraft.town.town_fitness_separate_houses import Town_SeparateHouses_Fitness
from novelty_neat.fitness.reachability_fitness import ReachabilityFitness
from novelty_neat.fitness.regions import BetterRegionsFitness, RegionsFitness
from novelty_neat.fitness.simple_house_fitness import SimpleIsReasonableHouseFitness
from novelty_neat.fitness.simple_house_fitness_more_hardcore import SimpleIsReasonableHouseHardcoreFitness
from novelty_neat.fitness.specific_prob_dist_fitness import SpecificProbabilityDistributionFitness
from novelty_neat.general.neat_generate_general_level import GenerateGeneralLevelUsingTiling3D
from novelty_neat.maze.neat_maze_fitness import PartialSolvabilityFitness, SolvabilityFitness
from novelty_neat.maze.neat_maze_level_generation import GenerateMazeLevelsUsingMoreContext, GenerateMazeLevelsUsingCPPNCoordinates, GenerateMazeLevelsUsingTilingVariableTileSize

from novelty_neat.novelty_neat import NoveltyNeatPCG
from novelty_neat.novelty.novelty_metric import NoveltyArchive, NoveltyIntraGenerator, NoveltyMetric
from novelty_neat.novelty.distance_functions.distance import dist_jensen_shannon_compare_probabilities, euclidean_distance, image_hash_distance_average, image_hash_distance_perceptual, image_hash_distance_perceptual_3d, image_hash_distance_perceptual_simple, image_hash_distance_wavelet_3d, jensen_shannon_compare_trajectories_distance, jensen_shannon_distance, minecraft_js_divergence_tile_distros, minecraft_js_divergence_tile_distros_matthew, threed_level_jensen_shannon, threed_level_kl_divergence, visual_diversity, visual_diversity_only_reachable, image_hash_distance_wavelet, dist_compare_shortest_paths
import neat
import cProfile
import pstats
import ray

os.environ['WANDB_SILENT'] = 'True'
GAME_CLASS = MinecraftGameGarden
LEVEL_CLASS = MinecraftGardenLevel

# DOES TOWNS
def experiment_712_matthew(name='a', 
                   gens=200, 
                   pop_size=20,
                   distance_func=visual_diversity, 
                   max_dist=10**3,
                   use_prob_dist = False,
                   town_fitness = False,
                   all_weights: List[float] = None,
                   do_one_hot=True,
                   town_separate_houses = False,
                   do_garden_fitness = False
                   ):
    """
    Tries to generate useful shapes, with 10^3 blocks and only one 

    Returns:
        _type_: _description_
    """
    ray.init()
    name = f'experiment_712_matthew{name}' 
    game = 'Minecraft'
    method = 'NEAT'
    date = get_date()
    generations = gens

    if do_one_hot:
        config_file = f'runs/proper_experiments/v700/config/tiling_generate_106_1_balanced_pop{pop_size}'
    else:
        config_file = f'runs/proper_experiments/v700/config/tiling_generate_28_5_balanced_pop{pop_size}'
    print(f"Doing now -- {name} -- {gens} -- {distance_func} -- {max_dist}")
    results_directory = f'../results/experiments/{name}/{game}/{method}/{date}/{pop_size}/{generations}'
    maze_game = GAME_CLASS(LEVEL_CLASS.random())
    level_generator = GenerateGeneralLevelUsingTiling3D(
        game=maze_game, 
        number_of_random_variables=2, 
        do_padding_randomly=False,
        random_perturb_size=0.1565, 
        use_one_hot_encoding=do_one_hot)
    
    def get_overall_fitness() -> NeatFitnessFunction:
        num_levels = 10
        num_levels_other = num_levels
        K = 5
        # distance_func = visual_diversity; max_dist = maze_game.level.width * maze_game.level.height * maze_game.level.depth
        funcs = [
            NoveltyMetric(level_generator, distance_func, max_dist=max_dist, number_of_levels=num_levels, 
    number_of_neighbours=K, lambd=0, archive_mode=NoveltyArchive.RANDOM,
    should_use_all_pairs=False),
NoveltyIntraGenerator(num_levels, level_generator, distance_func, max_dist=max_dist, 
number_of_neighbours=min(5, num_levels - 1))
        ]
        weights = [1, 1]
        if use_prob_dist != False:
            funcs.append(
            SpecificProbabilityDistributionFitness(number_of_levels_to_generate=num_levels, level_gen=level_generator, desired_dist={
                                            'air': 0.6,
                                            'dirt': 0.4,
                                        } if use_prob_dist == True else use_prob_dist),
            )
            weights.append(1)
        
        # if town_fitness:
        #     funcs.append(Town_Reachable_Fitness(number_of_levels_to_generate=num_levels, level_gen=level_generator))
        #     weights.append(1)
            
        # if town_separate_houses:
        #     funcs.append(Town_SeparateHouses_Fitness(number_of_levels_to_generate=num_levels, level_gen=level_generator))
        #     weights.append(1)
        
        if do_garden_fitness:
            funcs.append(Gardens_Fitness(number_of_levels_to_generate=num_levels, level_gen=level_generator))
            weights.append(1)
        
        if all_weights is not None:
            assert len(weights) == len(all_weights)
            weights = all_weights
        
        return CombinedFitness(funcs, weights,
                                  number_of_levels_to_generate=num_levels_other, level_gen=level_generator, mode='add')
    args = {
        'population_size': pop_size,
        'number_of_generations': generations,
        'fitness': get_overall_fitness().params(),
        'level_gen': 'tiling',
        'config_filename': config_file
    }
    print(args)
    def get_neat_config():
        return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           config_file)
    
    def get_pop(seed):
        level = LEVEL_CLASS.random()
        game = GAME_CLASS(level)
        fitness = get_overall_fitness()
        return NoveltyNeatPCG(game, level, level_generator=level_generator, fitness_calculator=fitness, neat_config=get_neat_config(),
                              num_generations=generations, results_dir_to_save=f"{results_directory}/{seed}/saves")

    @ray.remote
    def single_func(seed):
        config = Config(
            name=name,
            game=game,
            method=method,
            seed=seed,
            results_directory=os.path.join(results_directory, str(seed)),
            method_parameters=args,
            date=date
        )
        print("Date = ", config.date, config.results_directory, config.hash(seed=False))
        experiment = Experiment(config, lambda: get_pop(seed), [], log_to_wandb=True, verbose=Verbosity.PROGRESS if seed == 0 else Verbosity.NONE)

        experiment.do_all()
        wandb.finish()

        try:
            print('num novel', len(experiment.method.fitness_calculator.fitnesses[0].previously_novel_individuals))
        except Exception as e:
            print("Length failed with msg", e)
        return f"Completed with seed = {seed}"
    
    futures = [single_func.remote(i) for i in range(1)]
    print(ray.get(futures))

if __name__ == '__main__':
    arg = sys.argv[-1]
    {
        'a': lambda: experiment_712_matthew(arg, 100, 20, visual_diversity, 10**3, do_garden_fitness=True, do_one_hot=False,       all_weights=[1, 1, 4]), # first one that you liked
        'b': lambda: experiment_712_matthew(arg, 1000, 20, minecraft_js_divergence_tile_distros, 1, do_garden_fitness=True, do_one_hot=False,       all_weights=[1, 1, 4]), 
        'c': lambda: experiment_712_matthew(arg, 500, 20, minecraft_js_divergence_tile_distros, 1, do_garden_fitness=True, do_one_hot=False,       all_weights=[2, 10, 10]),
        'd': lambda: experiment_712_matthew(arg, 300, 20, minecraft_js_divergence_tile_distros_matthew, 1, do_garden_fitness=True, do_one_hot=False,       all_weights=[1, 1, 4]),
        'e': lambda: experiment_712_matthew(arg, 1500, 20, minecraft_js_divergence_tile_distros_matthew, 1, do_garden_fitness=True, do_one_hot=False,       all_weights=[1, 2, 10]), # 
    }[arg]()
    