import sys
from games.minecraft.gardens.minecraft_game_garden import MinecraftGameGarden
from games.minecraft.gardens.minecraft_level_garden import MinecraftGardenLevel
from games.minecraft.minecraft_game import MinecraftGame
from games.minecraft.minecraft_level import MinecraftLevel
from games.minecraft.towns.minecraft_game_town import MinecraftGameTown
from games.minecraft.towns.minecraft_game_town_larger import MinecraftGameTownLarger
from games.minecraft.towns.minecraft_level_town import MinecraftTown
from games.minecraft.towns.minecraft_level_town_larger import MinecraftTownLarger
from novelty_neat.fitness.another_door_fitness import AnotherDoorFitness
from novelty_neat.fitness.another_door_fitness_linear import AnotherDoorFitnessLinear
from novelty_neat.fitness.another_door_fitness_linear_more import AnotherDoorFitnessLinearMore
from novelty_neat.fitness.door_fitnesses import SimpleDoorFitness
from novelty_neat.fitness.minecraft.garden.town_fitness_gardens import Gardens_Fitness
from novelty_neat.fitness.minecraft.garden.town_fitness_gardens_v2 import Gardens_FitnessV2
from novelty_neat.fitness.minecraft.town.town_fitness import Town_Reachable_Fitness
from novelty_neat.fitness.minecraft.town.town_fitness_more import Town_ReachableMore_Fitness
from novelty_neat.fitness.minecraft.town.town_larger_fitness_more import TownLarger_ReachableMore_Fitness
from novelty_neat.fitness.minecraft.town.town_larger_fitness_more2 import TownLarger_ReachableMore_Fitness2
from novelty_neat.fitness.minecraft.town.town_larger_fitness_more3 import TownLarger_ReachableMore_Fitness3
from novelty_neat.fitness.minecraft.town.town_larger_fitness_more_coherence import TownLarger_ReachableMore_FitnessCoherence
from novelty_neat.fitness.simple_house_fitness import SimpleIsReasonableHouseFitness
from novelty_neat.fitness.specific_prob_dist_fitness import SpecificProbabilityDistributionFitness
from novelty_neat.novelty.distance_functions.distance import good_distance, minecraft_js_divergence_tile_distros, minecraft_js_divergence_tile_distros_good, visual_diversity, visual_diversity_normalised

from runs.proper_experiments.v700.vfinal.main_run_file import run_main_experiment

def temp(number_of_levels_to_generate, level_gen):
    return SpecificProbabilityDistributionFitness(number_of_levels_to_generate, level_gen, {'empty': 0.1, 'house': 0.35, 'garden': 0.25, 'road': 0.3})

def temp2(number_of_levels_to_generate, level_gen):
    return SpecificProbabilityDistributionFitness(number_of_levels_to_generate, level_gen, {'empty': 0.05, 'house': 0.3, 'garden': 0.35, 'road': 0.3})


def temp3(number_of_levels_to_generate, level_gen):
    return SpecificProbabilityDistributionFitness(number_of_levels_to_generate, level_gen, {
        'empty' :  0.05, 
        'house' :  0.1, 
        'house_1': 0.1, 
        'house_3': 0.1, 
        'garden': 0.35, 
        'road': 0.3})



if __name__ == '__main__':
    arg = sys.argv[-1]
    {
        'a': lambda: run_main_experiment('755_towns', arg, 500, 50, visual_diversity_normalised, 1, 
                                 GAME_CLASS=MinecraftGameTown, LEVEL_CLASS=MinecraftTown,
                                 additional_fitnesses=[Town_Reachable_Fitness, temp], 
                                 all_weights=[1, 1, 4, 2], NUM_SEEDS=5),
        
        
        'b': lambda: run_main_experiment('755_towns', arg, 500, 50, visual_diversity_normalised, 1, 
                                 GAME_CLASS=MinecraftGameTown, LEVEL_CLASS=MinecraftTown,
                                 additional_fitnesses=[Town_ReachableMore_Fitness, temp2], 
                                 all_weights=[1, 1, 4, 2], NUM_SEEDS=5),
        
        
        'c': lambda: run_main_experiment('755_towns', arg, 350, 30, visual_diversity_normalised, 1, 
                                 GAME_CLASS=MinecraftGameTownLarger, LEVEL_CLASS=MinecraftTownLarger,
                                 additional_fitnesses=[TownLarger_ReachableMore_Fitness, temp3], 
                                 all_weights=[1, 1, 4, 3], NUM_SEEDS=5),
        
        'd': lambda: run_main_experiment('755_towns', arg, 350, 30, visual_diversity_normalised, 1, 
                                 GAME_CLASS=MinecraftGameTownLarger, LEVEL_CLASS=MinecraftTownLarger,
                                 additional_fitnesses=[TownLarger_ReachableMore_Fitness2, temp3], 
                                 all_weights=[1, 1, 4, 1.5], NUM_SEEDS=5),
        
        'e': lambda: run_main_experiment('755_towns', arg, 1000, 50, visual_diversity_normalised, 1, 
                                 GAME_CLASS=MinecraftGameTownLarger, LEVEL_CLASS=MinecraftTownLarger,
                                 additional_fitnesses=[TownLarger_ReachableMore_Fitness3, TownLarger_ReachableMore_FitnessCoherence, temp3], 
                                 all_weights=[1, 1, 4, 4, 1.5], NUM_SEEDS=5),
        
        'f': lambda: run_main_experiment('755_towns', arg, 1000, 50, good_distance, 1, 
                                 GAME_CLASS=MinecraftGameTownLarger, LEVEL_CLASS=MinecraftTownLarger,
                                 additional_fitnesses=[TownLarger_ReachableMore_Fitness3, TownLarger_ReachableMore_FitnessCoherence, temp3], 
                                 all_weights=[1, 1, 4, 4, 1.5], NUM_SEEDS=5),
        
        'g': lambda: run_main_experiment('755_towns', arg, 1000, 50, good_distance, 1, 
                                 GAME_CLASS=MinecraftGameTownLarger, LEVEL_CLASS=MinecraftTownLarger,
                                 additional_fitnesses=[TownLarger_ReachableMore_Fitness3, TownLarger_ReachableMore_FitnessCoherence, temp3], 
                                 all_weights=[1, 1, 2, 4, 1.5], NUM_SEEDS=5),
        
        'h': lambda: run_main_experiment('755_towns', arg, 1000, 50, good_distance, 1, 
                                 GAME_CLASS=MinecraftGameTownLarger, LEVEL_CLASS=MinecraftTownLarger,
                                 additional_fitnesses=[TownLarger_ReachableMore_Fitness3, TownLarger_ReachableMore_FitnessCoherence, temp3], 
                                 all_weights=[1, 1, 2, 2, 4], NUM_SEEDS=5),
        
        'i': lambda: run_main_experiment('755_towns', arg, 1000, 50, visual_diversity_normalised, 1, 
                                 GAME_CLASS=MinecraftGameTownLarger, LEVEL_CLASS=MinecraftTownLarger,
                                 additional_fitnesses=[TownLarger_ReachableMore_Fitness3, TownLarger_ReachableMore_FitnessCoherence, temp3], 
                                 all_weights=[1, 1, 2, 4, 1.5], NUM_SEEDS=5),
        
        'j': lambda: run_main_experiment('755_towns', arg, 1000, 50, visual_diversity_normalised, 1, 
                                 GAME_CLASS=MinecraftGameTownLarger, LEVEL_CLASS=MinecraftTownLarger,
                                 additional_fitnesses=[TownLarger_ReachableMore_Fitness3, TownLarger_ReachableMore_FitnessCoherence, temp3], 
                                 all_weights=[1, 1, 2, 2, 4], NUM_SEEDS=5),
    }[arg]()
    