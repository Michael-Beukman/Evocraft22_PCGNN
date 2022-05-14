from minecraft_pb2 import AIR, BRICK_BLOCK, DIAMOND_BLOCK, GRASS, STONE
from games.minecraft.minecraft_game import MinecraftGame
from games.minecraft.comp.compositional_config import HOUSE_SIZE, X, Y, generators, mapping
from games.minecraft.comp.compositional_shared import get_generator, get_level, get_level_with_same_class_different_shape, make_x
from games.minecraft.comp.compositional_main import Compositional, is_house
from games.minecraft.blocks import COAL_BLOCK, OBSIDIAN, WATER
import random
import numpy as np
import tqdm
np.random.seed(42)
random.seed(42)

TOWN_TILE = 60
TOWN_SIZE = 5
USE_WATER = False

assert TOWN_TILE == X, "X Must be the same as TOWN_TILE"
gen = get_generator(MinecraftGame(get_level_with_same_class_different_shape(
    generators['town']['game'], TOWN_SIZE, 1, TOWN_SIZE)))
level = get_level(generators, 'town', 0, gen_to_override=gen)
print("L", level.map.shape, np.unique(level.map))
print((level.map == 5)*1)
total = 0
new_arr = np.zeros((TOWN_SIZE * TOWN_TILE, Y, TOWN_SIZE *
                   TOWN_TILE), dtype=np.uint8) + AIR
pbar = tqdm.tqdm(total=TOWN_SIZE ** 2)
for i in range(level.map.shape[0]):
    for j in range(level.map.shape[-1]):
        pbar.update()
        tile = int(level.map[i, 0, j])
        if is_house(tile):
            elem = Compositional().get_new_map(
                get_level(generators, 'town', 0).map, mapping, verbose=False)
        elif tile in [0, 2, 4]:
            # garden
            elem = make_x('garden', tile, None,
                          generators=generators,
                          gen_to_override=get_generator(MinecraftGame(get_level_with_same_class_different_shape(generators['garden']['game'], TOWN_TILE, 1, TOWN_TILE))))
        elif tile in [3, 5]:  # road
            TTT = 3
            elem = np.ones((TOWN_TILE, 2 + USE_WATER, TOWN_TILE)) * AIR
            mid = TOWN_TILE // 2
            elem[mid-TTT: mid + TTT, 0, :] = WATER * \
                USE_WATER + STONE * (1 - USE_WATER)
            elem[:, 0, mid-TTT: mid + TTT] = WATER * \
                USE_WATER + STONE * (1 - USE_WATER)

            elem[mid-1: mid + 1, 0, :] = WATER
            elem[:, 0, mid-1: mid + 1] = WATER

            size_of_g = mid-TTT
            for p in range(2):
                for q in range(2):
                    G = make_x('garden', tile, None,
                               generators=generators,
                               gen_to_override=get_generator(MinecraftGame(get_level_with_same_class_different_shape(generators['garden']['game'], size_of_g, 1, size_of_g))))
                    if USE_WATER:
                        abc = np.ones((size_of_g, 3, size_of_g)) * GRASS
                        abc[:, 1:, :] = G
                        G = abc
                    elem[size_of_g*p + TTT * 2 * p:size_of_g*(
                        p+1) + TTT * 2 * p, :, size_of_g*q + TTT * 2 * q:size_of_g*(q+1) + TTT * 2 * q] = G
        else:
            print("BAD", tile)
            assert 0 == 1

        total += np.sum(elem[:, 0, :] == AIR)
        assert elem.shape[0] == (TOWN_TILE)
        assert elem.shape[-1] == (TOWN_TILE)
        new_arr[i * elem.shape[0]: (i+1) * elem.shape[0],
                :elem.shape[1],
                j * elem.shape[2]: (j+1) * elem.shape[2]] = elem
pbar.close()
np.save('test_mine.npy', new_arr)
