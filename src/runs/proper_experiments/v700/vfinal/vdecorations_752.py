import sys
from games.minecraft.decorations.minecraft_decoration_game import MinecraftDecorationGame
from games.minecraft.decorations.minecraft_decoration_level import MinecraftDecorationLevel
from games.minecraft.gardens.minecraft_game_garden import MinecraftGameGarden
from games.minecraft.gardens.minecraft_level_garden import MinecraftGardenLevel
from novelty_neat.fitness.minecraft.decoration.decoration_fitness import DecorationFitness
from novelty_neat.fitness.minecraft.decoration.decoration_fitness_2 import DecorationFitnessV2
from novelty_neat.fitness.minecraft.decoration.decoration_fitness_3 import DecorationFitnessV3
from novelty_neat.fitness.minecraft.decoration.decoration_fitness_4 import DecorationFitnessV4
from novelty_neat.fitness.minecraft.decoration.decoration_fitness_5 import DecorationFitnessV5
from novelty_neat.fitness.minecraft.decoration.decoration_fitness_6 import DecorationFitnessV6
from novelty_neat.fitness.minecraft.decoration.decoration_fitness_7 import DecorationFitnessV7
from novelty_neat.fitness.minecraft.decoration.decoration_fitness_8 import DecorationFitnessV8
from novelty_neat.fitness.minecraft.decoration.decoration_fitness_9 import DecorationFitnessV9
from novelty_neat.fitness.minecraft.garden.town_fitness_gardens import Gardens_Fitness
from novelty_neat.novelty.distance_functions.distance import minecraft_js_divergence_tile_distros, minecraft_js_divergence_tile_distros_matthew, visual_diversity, visual_diversity_normalised

from runs.proper_experiments.v700.vfinal.main_run_file import run_main_experiment


if __name__ == '__main__':
    arg = sys.argv[-1]
    {
        'a': lambda: run_main_experiment('752_decorations', arg, 100, 20, visual_diversity_normalised, 1, 
                                 GAME_CLASS=MinecraftDecorationGame, LEVEL_CLASS=MinecraftDecorationLevel,
                                 additional_fitnesses=[DecorationFitness], all_weights=[1, 1, 4], NUM_SEEDS=1),
        
        
        'b': lambda: run_main_experiment('752_decorations', arg, 50, 20, minecraft_js_divergence_tile_distros, 1, 
                                 GAME_CLASS=MinecraftDecorationGame, LEVEL_CLASS=MinecraftDecorationLevel,
                                 additional_fitnesses=[DecorationFitnessV2], all_weights=[1, 1, 4], NUM_SEEDS=1),
        
        'c': lambda: run_main_experiment('752_decorations', arg, 50, 20, minecraft_js_divergence_tile_distros, 1, 
                                 GAME_CLASS=MinecraftDecorationGame, LEVEL_CLASS=MinecraftDecorationLevel,
                                 additional_fitnesses=[DecorationFitnessV3], all_weights=[1, 1, 4], NUM_SEEDS=1),
        
        'd': lambda: run_main_experiment('752_decorations', arg, 50, 20, minecraft_js_divergence_tile_distros, 1, 
                                 GAME_CLASS=MinecraftDecorationGame, LEVEL_CLASS=MinecraftDecorationLevel,
                                 additional_fitnesses=[DecorationFitnessV4], all_weights=[1, 1, 4], NUM_SEEDS=1),
        
        'e': lambda: run_main_experiment('752_decorations', arg, 50, 20, minecraft_js_divergence_tile_distros, 1, 
                                 GAME_CLASS=MinecraftDecorationGame, LEVEL_CLASS=MinecraftDecorationLevel,
                                 additional_fitnesses=[DecorationFitnessV5], all_weights=[1, 1, 4], NUM_SEEDS=1),
        
        'f': lambda: run_main_experiment('752_decorations', arg, 200, 20, minecraft_js_divergence_tile_distros, 1, 
                                GAME_CLASS=MinecraftDecorationGame, LEVEL_CLASS=MinecraftDecorationLevel,
                                additional_fitnesses=[DecorationFitnessV6], all_weights=[1, 1, 4], NUM_SEEDS=5),
        
        'g': lambda: run_main_experiment('752_decorations', arg, 200, 20, minecraft_js_divergence_tile_distros_matthew, 1, 
                                GAME_CLASS=MinecraftDecorationGame, LEVEL_CLASS=MinecraftDecorationLevel,
                                additional_fitnesses=[DecorationFitnessV6], all_weights=[1, 1, 4], NUM_SEEDS=5),
        
        'h': lambda: run_main_experiment('752_decorations', arg, 100, 20, minecraft_js_divergence_tile_distros_matthew, 1, 
                                GAME_CLASS=MinecraftDecorationGame, LEVEL_CLASS=MinecraftDecorationLevel,
                                additional_fitnesses=[DecorationFitnessV7], all_weights=[1, 1, 4], NUM_SEEDS=5),
        
        'i': lambda: run_main_experiment('752_decorations', arg, 1000, 20, minecraft_js_divergence_tile_distros_matthew, 1, 
                                GAME_CLASS=MinecraftDecorationGame, LEVEL_CLASS=MinecraftDecorationLevel,
                                additional_fitnesses=[DecorationFitnessV7], all_weights=[1, 1, 4], NUM_SEEDS=5),
        
        'j': lambda: run_main_experiment('752_decorations', arg, 1000, 20, minecraft_js_divergence_tile_distros_matthew, 1, 
                                GAME_CLASS=MinecraftDecorationGame, LEVEL_CLASS=MinecraftDecorationLevel,
                                additional_fitnesses=[DecorationFitnessV8], all_weights=[1, 1, 4], NUM_SEEDS=5),
        
        'k': lambda: run_main_experiment('752_decorations', arg, 500, 50, minecraft_js_divergence_tile_distros_matthew, 1, 
                                GAME_CLASS=MinecraftDecorationGame, LEVEL_CLASS=MinecraftDecorationLevel,
                                additional_fitnesses=[DecorationFitnessV9], all_weights=[1, 1, 4], NUM_SEEDS=5),
    }[arg]()
    