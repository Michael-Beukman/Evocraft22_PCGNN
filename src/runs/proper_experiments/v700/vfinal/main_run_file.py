from cmath import exp
from functools import partial
import os
from pprint import pprint
import sys
import threading
from typing import Callable, List
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
from novelty_neat.novelty.distance_functions.distance import dist_jensen_shannon_compare_probabilities, euclidean_distance, image_hash_distance_average, image_hash_distance_perceptual_3d, image_hash_distance_perceptual_simple, image_hash_distance_wavelet_3d, jensen_shannon_compare_trajectories_distance, jensen_shannon_distance, minecraft_js_divergence_tile_distros, minecraft_js_divergence_tile_distros_matthew, threed_level_jensen_shannon, threed_level_kl_divergence, visual_diversity, visual_diversity_only_reachable, image_hash_distance_wavelet, dist_compare_shortest_paths
import neat
import cProfile
import pstats
import ray

def save_config_file(in_dim, out_dim, pop_size):
    fname = f'runs/proper_experiments/v700/config/tiling_generate_{in_dim}_{out_dim}_balanced_pop{pop_size}'
    s = f"""
[NEAT]
fitness_criterion     = max
fitness_threshold     = 10000000
pop_size              = {pop_size}
reset_on_extinction   = False

[DefaultGenome]
# node activation options
activation_default      = random
activation_mutate_rate  = 0.5
activation_options      = sigmoid sin gauss

# node aggregation options
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

# node bias options
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

# genome compatibility options
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

# connection add/remove rates
conn_add_prob           = 0.6
conn_delete_prob        = 0.3

# connection enable options
enabled_default         = True
enabled_mutate_rate     = 0.1

feed_forward            = True
initial_connection      = full

# node add/remove rates
node_add_prob           = 0.6
node_delete_prob        = 0.3

# network parameters
num_hidden              = 0
num_inputs              = {in_dim}
num_outputs             = {out_dim}

# node response options
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

# connection weight options
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
"""
    
    if not os.path.exists(fname):
        with open(fname, 'w+') as f:
            f.write(s)
    return fname
    
os.environ['WANDB_SILENT'] = 'True'


def run_main_experiment(
                   experiment_name: str,
                   name='a', 
                   gens=200, 
                   pop_size=20,
                   distance_func=visual_diversity, 
                   max_dist=10**3,
                   GAME_CLASS = MinecraftGameGarden,
                   LEVEL_CLASS = MinecraftGardenLevel,
                   all_weights: List[float] = None,
                   additional_fitnesses: List[Callable[[], NeatFitnessFunction]] = [],
                   NUM_SEEDS=5,
                   
                   predict_size: int = 1,
                   context_size: int = 1,
                   random_perturb = 0.1565
                   ):
    """
    Tries to generate useful shapes, with 10^3 blocks and only one 

    Returns:
        _type_: _description_
    """
    ray.init()
    name = f'experiment_{experiment_name}_{name}' 
    game = 'Minecraft'
    method = 'PCGNN'
    date = get_date()
    generations = gens

    game_to_use = GAME_CLASS(LEVEL_CLASS.random())
    
    
    t_s = context_size * 2 + 1
    p_s = predict_size ** 3
    in_dim = t_s ** 3 - 1 + 2
    if predict_size > 1:
        in_dim = (t_s + predict_size - 1) ** 3 + 2
    config_file = save_config_file(in_dim=in_dim, out_dim = p_s * len(game_to_use.level.tile_types), pop_size=pop_size)
    
    print(f"Doing now -- {name} -- {gens} -- {distance_func} -- {max_dist}")
    results_directory = f'../results/experiments/{name}/{game}/{method}/{date}/{pop_size}/{generations}'
    level_generator = GenerateGeneralLevelUsingTiling3D(
        game=game_to_use,
        number_of_random_variables=2, 
        do_padding_randomly=False,
        predict_size=predict_size,
        context_size=context_size,
        random_perturb_size=random_perturb, 
        use_one_hot_encoding=False)
    
    def get_overall_fitness() -> NeatFitnessFunction:
        num_levels = 10
        num_levels_other = num_levels
        K = 5
        funcs = [
            NoveltyMetric(level_generator, distance_func, max_dist=max_dist, number_of_levels=num_levels, 
                            number_of_neighbours=K, lambd=0, archive_mode=NoveltyArchive.RANDOM,
                            should_use_all_pairs=False),
            NoveltyIntraGenerator(num_levels, level_generator, distance_func, max_dist=max_dist, 
                                    number_of_neighbours=min(5, num_levels - 1))
        ]
        weights = [1, 1]

        if len(additional_fitnesses):
            funcs += [a(number_of_levels_to_generate=num_levels, level_gen=level_generator) for a in additional_fitnesses]
            weights += [1] * len(additional_fitnesses)
        
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


    futures = [single_func.remote(i) for i in range(NUM_SEEDS)]
    print(ray.get(futures))
