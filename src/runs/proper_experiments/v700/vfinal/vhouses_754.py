import sys
from games.minecraft.gardens.minecraft_game_garden import MinecraftGameGarden
from games.minecraft.gardens.minecraft_level_garden import MinecraftGardenLevel
from games.minecraft.houses.minecraft_house_game import MinecraftHouseGame
from games.minecraft.houses.minecraft_house_level import MinecraftHouseLevel
from games.minecraft.minecraft_game import MinecraftGame
from games.minecraft.minecraft_level import MinecraftLevel
from novelty_neat.fitness.another_door_fitness import AnotherDoorFitness
from novelty_neat.fitness.another_door_fitness_linear import AnotherDoorFitnessLinear
from novelty_neat.fitness.another_door_fitness_linear_again_further import AnotherDoorFitnessLinearFurther
from novelty_neat.fitness.another_door_fitness_linear_more import AnotherDoorFitnessLinearMore
from novelty_neat.fitness.door_fitnesses import SimpleDoorFitness
from novelty_neat.fitness.minecraft.garden.town_fitness_gardens import Gardens_Fitness
from novelty_neat.fitness.minecraft.garden.town_fitness_gardens_v2 import Gardens_FitnessV2
from novelty_neat.fitness.minecraft.house.door_fitness_multiple_tiles import DoorFitnessMultipleTiles
from novelty_neat.fitness.minecraft.house.no_single_blocks_multiple_tiles import NoSingleBlocksMultipleTilesFitness
from novelty_neat.fitness.minecraft.house.simple_house_fitness_multiple_tiles import SimpleHouseFitnessMultipleTiles
from novelty_neat.fitness.minecraft.house.tile_coherence_fitness import TileCoherenceFitness
from novelty_neat.fitness.no_single_blocks_fitness import NoSingleBlocksFitness
from novelty_neat.fitness.simple_house_fitness import SimpleIsReasonableHouseFitness
from novelty_neat.novelty.distance_functions.distance import minecraft_js_divergence_tile_distros, minecraft_js_divergence_tile_distros_matthew, visual_diversity, visual_diversity_normalised

from runs.proper_experiments.v700.vfinal.main_run_file import run_main_experiment


if __name__ == '__main__':
    arg = sys.argv[-1]
    {
        'a': lambda: run_main_experiment('754_houses', arg, 1000, 50, visual_diversity_normalised, 1, 
                                 GAME_CLASS=MinecraftGame, LEVEL_CLASS=MinecraftLevel,
                                 additional_fitnesses=[SimpleIsReasonableHouseFitness, AnotherDoorFitness], 
                                 all_weights=[1, 1, 4, 1], NUM_SEEDS=5),
        'b': lambda: run_main_experiment('754_houses', arg, 1000, 50, visual_diversity_normalised, 1, 
                                 GAME_CLASS=MinecraftGame, LEVEL_CLASS=MinecraftLevel,
                                 additional_fitnesses=[SimpleIsReasonableHouseFitness, AnotherDoorFitness], 
                                 all_weights=[1, 1, 4, 4], NUM_SEEDS=5),
        
    
        'c': lambda: run_main_experiment('754_houses', arg, 1000, 50, visual_diversity_normalised, 1, 
                                GAME_CLASS=MinecraftGame, LEVEL_CLASS=MinecraftLevel,
                                additional_fitnesses=[SimpleIsReasonableHouseFitness, AnotherDoorFitness], 
                                all_weights=[1, 1, 4, 2], NUM_SEEDS=5),
        
        'd': lambda: run_main_experiment('754_houses', arg, 1000, 50, visual_diversity_normalised, 1, 
                                GAME_CLASS=MinecraftGame, LEVEL_CLASS=MinecraftLevel,
                                additional_fitnesses=[SimpleIsReasonableHouseFitness, AnotherDoorFitness], 
                                all_weights=[1, 1, 2, 4], NUM_SEEDS=5),
        
        
        
        'e': lambda: run_main_experiment('754_houses', arg, 1000, 50, visual_diversity_normalised, 1, 
                                GAME_CLASS=MinecraftGame, LEVEL_CLASS=MinecraftLevel,
                                additional_fitnesses=[SimpleIsReasonableHouseFitness, AnotherDoorFitness], 
                                all_weights=[1, 1, 10, 2], NUM_SEEDS=5),
        
        'f': lambda: run_main_experiment('754_houses', arg, 1000, 50, visual_diversity_normalised, 1, 
                                GAME_CLASS=MinecraftGame, LEVEL_CLASS=MinecraftLevel,
                                additional_fitnesses=[SimpleIsReasonableHouseFitness, AnotherDoorFitness, SimpleDoorFitness], 
                                all_weights=[1, 1, 4, 2, 4], NUM_SEEDS=5),
        
        
        'g': lambda: run_main_experiment('754_houses', arg, 1000, 50, visual_diversity_normalised, 1, 
                                 GAME_CLASS=MinecraftGame, LEVEL_CLASS=MinecraftLevel,
                                 additional_fitnesses=[SimpleIsReasonableHouseFitness, AnotherDoorFitnessLinear], 
                                 all_weights=[1, 1, 4, 1], NUM_SEEDS=5),
        'h': lambda: run_main_experiment('754_houses', arg, 1000, 50, visual_diversity_normalised, 1, 
                                 GAME_CLASS=MinecraftGame, LEVEL_CLASS=MinecraftLevel,
                                 additional_fitnesses=[SimpleIsReasonableHouseFitness, AnotherDoorFitnessLinear], 
                                 all_weights=[1, 1, 4, 4], NUM_SEEDS=5),
        
        
        'i': lambda: run_main_experiment('754_houses', arg, 1000, 50, visual_diversity_normalised, 1, 
                                 GAME_CLASS=MinecraftGame, LEVEL_CLASS=MinecraftLevel,
                                 additional_fitnesses=[SimpleIsReasonableHouseFitness, AnotherDoorFitnessLinearMore], 
                                 all_weights=[1, 1, 4, 1], NUM_SEEDS=5),
        'j': lambda: run_main_experiment('754_houses', arg, 1000, 50, visual_diversity_normalised, 1, 
                                 GAME_CLASS=MinecraftGame, LEVEL_CLASS=MinecraftLevel,
                                 additional_fitnesses=[SimpleIsReasonableHouseFitness, AnotherDoorFitnessLinearMore], 
                                 all_weights=[1, 1, 4, 4], NUM_SEEDS=5),
        
        
        'k': lambda: run_main_experiment('754_houses', arg, 400, 50, visual_diversity_normalised, 1, 
                                 GAME_CLASS=MinecraftGame, LEVEL_CLASS=MinecraftLevel,
                                 additional_fitnesses=[SimpleIsReasonableHouseFitness, AnotherDoorFitnessLinearMore, NoSingleBlocksFitness], 
                                 all_weights=[1, 1, 4, 1, 4], NUM_SEEDS=5),
        'l': lambda: run_main_experiment('754_houses', arg, 400, 50, visual_diversity_normalised, 1, 
                                 GAME_CLASS=MinecraftGame, LEVEL_CLASS=MinecraftLevel,
                                 additional_fitnesses=[SimpleIsReasonableHouseFitness, AnotherDoorFitnessLinearMore, NoSingleBlocksFitness], 
                                 all_weights=[1, 1, 4, 4, 4], NUM_SEEDS=5),
        
        'm': lambda: run_main_experiment('754_houses', arg, 400, 50, minecraft_js_divergence_tile_distros_matthew, 1, 
                                 GAME_CLASS=MinecraftHouseGame, LEVEL_CLASS=MinecraftHouseLevel,
                                 additional_fitnesses=[SimpleHouseFitnessMultipleTiles, DoorFitnessMultipleTiles, NoSingleBlocksMultipleTilesFitness, TileCoherenceFitness], 
                                 all_weights=[2, 1, 4, 4, 4, 8], NUM_SEEDS=5),
        
        
        'n': lambda: run_main_experiment('754_houses', arg, 400, 50, minecraft_js_divergence_tile_distros_matthew, 1, 
                                 GAME_CLASS=MinecraftHouseGame, LEVEL_CLASS=MinecraftHouseLevel,
                                 additional_fitnesses=[SimpleHouseFitnessMultipleTiles, DoorFitnessMultipleTiles, NoSingleBlocksMultipleTilesFitness, TileCoherenceFitness], 
                                 all_weights=[2, 1, 4, 4, 4, 8], NUM_SEEDS=1, context_size=2),
        
        
        'o': lambda: run_main_experiment('754_houses', arg, 300, 50, visual_diversity_normalised, 1, 
                                 GAME_CLASS=MinecraftGame, LEVEL_CLASS=MinecraftLevel,
                                 additional_fitnesses=[SimpleIsReasonableHouseFitness, AnotherDoorFitnessLinear], 
                                 all_weights=[1, 1, 4, 1], NUM_SEEDS=5, context_size=2),
    }[arg]()
    