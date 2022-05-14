import glob
import os

from natsort import natsorted
import numpy as np
from common.utils import load_compressed_pickle
from games.game import Game
from games.minecraft.decorations.minecraft_decoration_level import MinecraftDecorationLevel
from games.minecraft.gardens.minecraft_level_garden import MinecraftGardenLevel, OldMinecraftGardenLevel
from games.minecraft.blocks import *
from games.minecraft.minecraft_game import MinecraftGame
from games.minecraft.minecraft_level import MinecraftLevel
from games.minecraft.roofs.minecraft_roof_level import MinecraftRoofLevel
from games.minecraft.towns.minecraft_level_town import MinecraftTown
from games.minecraft.towns.minecraft_level_town_larger import MinecraftTownLarger
from novelty_neat.general.neat_generate_general_level import GenerateGeneralLevelUsingTiling3D
def get_level_with_same_class_different_shape(game: Game, width_x, width_y, width_z):
    N = game.level.__class__.__name__
    return globals()[N].random(width_x, width_y, width_z)

def get_generator(game: Game):
    return GenerateGeneralLevelUsingTiling3D(
        game=game, 
        number_of_random_variables=2, 
        do_padding_randomly=False,
        random_perturb_size=0.1565, use_one_hot_encoding=False)

def get_tiles(version, level, config):
    def F(i):
        if 'decoration' in version: return [AIR, TORCH, BOOKSHELF, CRAFTING_TABLE, JUKEBOX]
        
        if 'garden' in version: return [AIR, SAPLING, RED_FLOWER, YELLOW_FLOWER, WATER]
        if '712_matthew' in version: return [AIR, SAPLING, TALLGRASS, DEADBUSH, RED_FLOWER, YELLOW_FLOWER, WATER]
        
        
        # Roofs
        if 'roof' in version: 
            if config['USE_DIFFERENT_ROOF_MATERIALS']:
                return [AIR, [PLANKS, MOSSY_COBBLESTONE, REDSTONE_LAMP][int(i) - 1]]
            return [AIR, PLANKS]
        
        if 'house' in version: 
            if config['USE_DIFFERENT_HOUSE_MATERIALS']:
                return [AIR, [BRICK_BLOCK, LOG, STONE][int(i) - 1]]
            return [AIR, BRICK_BLOCK]
    return F


def get_game_network(version, width_x=10, width_z=10, width_y=1, how_many_nets = 1, config = {}, seed=0):
    SA = f'../results/experiments/experiment_{version}/Minecraft/PCGNN/*/*/*/{seed}/saves/self/gen_*.pbz2'
    SB = f'../results/experiments/experiment_{version}/Minecraft/NEAT/*/*/*/{seed}/saves/self/gen_*.pbz2'
    A = glob.glob(SA)
    B = glob.glob(SB)
    if len(A) == 0: 
        SA = SB
        A = B
    
    if seed >= 1 and len(A) == 0:
        return None
    
    assert len(A) != 0, f"{SA}, {SB}"
    self = load_compressed_pickle(natsorted(A)[-1])
    self = self['self']
    if '712_matthew' in version :
        self.game = MinecraftGame(OldMinecraftGardenLevel.random(width_x, width_y, width_z))
    best_network = load_compressed_pickle(natsorted(glob.glob(SA.replace("/self/", "/max/")))[-1])['net']
    all_networks = load_compressed_pickle(natsorted(glob.glob(SA.replace("/self/", "/alls/")))[-1])['nets']
    game = MinecraftGame(
        get_level_with_same_class_different_shape(self.game, width_x, width_y, width_z)
    )
    
    # get all networks:
    best_dir = max(glob.glob('/'.join(SA.replace("/self/", "/max/").split("/")[:-1])))
    all_nets = natsorted(glob.glob(os.path.join(best_dir, 'gen_*.pbz2')))
    all_nets = [load_compressed_pickle(p)['net'] for p in all_nets]
    
    return {
        'game': game,
        'best_network': best_network,
        'all_networks': all_networks,
        'networks_to_use': [best_network] + all_networks[:how_many_nets - 1],
        'all_generations': all_nets,
        'generator': get_generator(game),
        'version': version,
        'tiles': get_tiles(version, game.level, config),
        'config': config
    }

def get_level(generators, name, i, **kwargs):
    config = generators[name]['config']
    gen, net = generators[name]['generator'], np.random.choice(generators[name]['networks_to_use'])
    if 'net_to_override' in kwargs: 
        net = kwargs['net_to_override']
        del kwargs['net_to_override']
    if name == 'house' and config['USE_DIFFERENT_HOUSE_HEIGHTS']:
        game = generators[name]['game']
        _X, _Y, _Z = game.level.map.shape
        i = int(i)
        gen = get_generator(MinecraftGame(get_level_with_same_class_different_shape(game, _X, _Y * i, _Z)))
    if name == 'decoration' and 'start_output' in kwargs:
        game = generators[name]['game']
        _X, _Y, _Z = kwargs['start_output'].shape
        gen = get_generator(MinecraftGame(get_level_with_same_class_different_shape(game, _X, _Y, _Z)))
    
    if 'gen_to_override' in kwargs: 
        gen = kwargs['gen_to_override']
        del kwargs['gen_to_override']
    
    return gen(net, **kwargs)



def make_x(name: str, i: int, start=None, generators=None, **kwargs):
    assert generators is not None
    H = get_level(generators, name, i, start_output=start, **kwargs).map[:, ::-1, :]
    TILES = generators[name]['tiles'](i)
    if name == 'garden':
        T = np.concatenate([H, H], axis=1)
        T[:, -1, :] = GRASS
        new = np.zeros_like(T) + AIR
        new[:, -1, :] = T[:, -1, :]
        for i, j in enumerate(TILES):
            idx = T.astype(np.int32) == i
            if j == WATER:
                idx = idx[:, ::-1, :]
                idx[:, 0, :] = False
                new[idx] = j
            else:
                new[idx] = j
        new = new[:, ::-1, :]
    else:
        new = np.zeros_like(H)
        for i, j in enumerate(TILES):
            idx = H.astype(np.int32) == i
            new[idx] = j
    return new
    
