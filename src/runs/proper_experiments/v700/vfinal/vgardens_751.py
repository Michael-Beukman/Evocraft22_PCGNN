import sys
from games.minecraft.gardens.minecraft_game_garden import MinecraftGameGarden
from games.minecraft.gardens.minecraft_level_garden import MinecraftGardenLevel
from novelty_neat.fitness.minecraft.garden.town_fitness_gardens import Gardens_Fitness
from novelty_neat.fitness.minecraft.garden.town_fitness_gardens_v2 import Gardens_FitnessV2
from novelty_neat.fitness.minecraft.town.town_fitness_gardens3 import Gardens_FitnessV3
from novelty_neat.novelty.distance_functions.distance import minecraft_js_divergence_tile_distros, visual_diversity, visual_diversity_normalised

from runs.proper_experiments.v700.vfinal.main_run_file import run_main_experiment


if __name__ == '__main__':
    arg = sys.argv[-1]
    {
        'a': lambda: run_main_experiment('751_gardens', arg, 2000, 50, visual_diversity_normalised, 1, 
                                 GAME_CLASS=MinecraftGameGarden, LEVEL_CLASS=MinecraftGardenLevel,
                                 additional_fitnesses=[Gardens_Fitness], all_weights=[1, 1, 4], NUM_SEEDS=5),
        
        'b': lambda: run_main_experiment('751_gardens', arg, 2000, 50, minecraft_js_divergence_tile_distros, 1, 
                                 GAME_CLASS=MinecraftGameGarden, LEVEL_CLASS=MinecraftGardenLevel,
                                 additional_fitnesses=[Gardens_Fitness], all_weights=[1, 1, 4], NUM_SEEDS=5),   
        
        'c': lambda: run_main_experiment('751_gardens', arg, 2000, 50, minecraft_js_divergence_tile_distros, 1, 
                                 GAME_CLASS=MinecraftGameGarden, LEVEL_CLASS=MinecraftGardenLevel,
                                 additional_fitnesses=[Gardens_Fitness], all_weights=[1, 2, 10], NUM_SEEDS=5),
        
        
        'd': lambda: run_main_experiment('751_gardens', arg, 2000, 50, visual_diversity_normalised, 1, 
                                 GAME_CLASS=MinecraftGameGarden, LEVEL_CLASS=MinecraftGardenLevel,
                                 additional_fitnesses=[Gardens_FitnessV2], all_weights=[1, 1, 4], NUM_SEEDS=5),
        
        'e': lambda: run_main_experiment('751_gardens', arg, 2000, 50, minecraft_js_divergence_tile_distros, 1, 
                                 GAME_CLASS=MinecraftGameGarden, LEVEL_CLASS=MinecraftGardenLevel,
                                 additional_fitnesses=[Gardens_FitnessV2], all_weights=[1, 1, 4], NUM_SEEDS=5),   
        
        'f': lambda: run_main_experiment('751_gardens', arg, 2000, 50, minecraft_js_divergence_tile_distros, 1, 
                                 GAME_CLASS=MinecraftGameGarden, LEVEL_CLASS=MinecraftGardenLevel,
                                 additional_fitnesses=[Gardens_FitnessV2], all_weights=[1, 2, 10], NUM_SEEDS=5),
        
        
        'g': lambda: run_main_experiment('751_gardens', arg, 1000, 20, minecraft_js_divergence_tile_distros, 1, 
                                 GAME_CLASS=MinecraftGameGarden, LEVEL_CLASS=MinecraftGardenLevel,
                                 additional_fitnesses=[Gardens_FitnessV3], all_weights=[1, 1, 4], NUM_SEEDS=5),
    }[arg]()
    