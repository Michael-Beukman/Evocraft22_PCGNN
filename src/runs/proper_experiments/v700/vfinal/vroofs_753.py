import sys
from games.minecraft.decorations.minecraft_decoration_game import MinecraftDecorationGame
from games.minecraft.decorations.minecraft_decoration_level import MinecraftDecorationLevel
from games.minecraft.gardens.minecraft_game_garden import MinecraftGameGarden
from games.minecraft.gardens.minecraft_level_garden import MinecraftGardenLevel
from games.minecraft.roofs.minecraft_roof_game import MinecraftRoofGame
from games.minecraft.roofs.minecraft_roof_level import MinecraftRoofLevel
from novelty_neat.fitness.decoration.decoration_fitness import DecorationFitness
from novelty_neat.fitness.decoration.decoration_fitness_2 import DecorationFitnessV2
from novelty_neat.fitness.decoration.decoration_fitness_3 import DecorationFitnessV3
from novelty_neat.fitness.decoration.decoration_fitness_4 import DecorationFitnessV4
from novelty_neat.fitness.minecraft.garden.town_fitness_gardens import Gardens_Fitness
from novelty_neat.fitness.minecraft.roof.my_roof_fitness import MyRoofFitness
from novelty_neat.fitness.minecraft.roof.my_roof_fitness_2 import MyRoofFitnessV2
from novelty_neat.fitness.minecraft.roof.my_roof_fitness_3 import MyRoofFitnessV3
from novelty_neat.fitness.minecraft.roof.my_roof_fitness_4 import MyRoofFitnessV4
from novelty_neat.fitness.minecraft.roof.my_roof_fitness_5 import MyRoofFitnessV5
from novelty_neat.novelty.distance_functions.distance import minecraft_js_divergence_tile_distros, visual_diversity, visual_diversity_normalised

from runs.proper_experiments.v700.vfinal.main_run_file import run_main_experiment


if __name__ == '__main__':
    arg = sys.argv[-1]
    {
        'a': lambda: run_main_experiment('753_roofs', arg, 100, 20, visual_diversity_normalised, 1, 
                                 GAME_CLASS=MinecraftRoofGame, LEVEL_CLASS=MinecraftRoofLevel,
                                 additional_fitnesses=[MyRoofFitness], all_weights=[1, 1, 4], NUM_SEEDS=1),
        
        'b': lambda: run_main_experiment('753_roofs', arg, 100, 20, visual_diversity_normalised, 1, 
                                GAME_CLASS=MinecraftRoofGame, LEVEL_CLASS=MinecraftRoofLevel,
                                additional_fitnesses=[MyRoofFitnessV2], all_weights=[1, 1, 4], NUM_SEEDS=1),
        
        'c': lambda: run_main_experiment('753_roofs', arg, 100, 20, visual_diversity_normalised, 1, 
                                GAME_CLASS=MinecraftRoofGame, LEVEL_CLASS=MinecraftRoofLevel,
                                additional_fitnesses=[MyRoofFitnessV3], all_weights=[1, 1, 10], NUM_SEEDS=1),
        
            
        'd': lambda: run_main_experiment('753_roofs', arg, 100, 20, visual_diversity_normalised, 1, 
                                GAME_CLASS=MinecraftRoofGame, LEVEL_CLASS=MinecraftRoofLevel,
                                additional_fitnesses=[MyRoofFitnessV4], all_weights=[1, 1, 10], NUM_SEEDS=1),
        
        
        'e': lambda: run_main_experiment('753_roofs', arg, 1000, 50, visual_diversity_normalised, 1, 
                                GAME_CLASS=MinecraftRoofGame, LEVEL_CLASS=MinecraftRoofLevel,
                                additional_fitnesses=[MyRoofFitnessV3], all_weights=[1, 1, 10], NUM_SEEDS=1),
        
        
        'f': lambda: run_main_experiment('753_roofs', arg, 500, 50, visual_diversity_normalised, 1, 
                                GAME_CLASS=MinecraftRoofGame, LEVEL_CLASS=MinecraftRoofLevel,
                                additional_fitnesses=[MyRoofFitnessV5], all_weights=[1, 1, 10], NUM_SEEDS=5),
    

    }[arg]()
    