from time import sleep
from games.minecraft.comp.compositional_shared import get_generator, get_level, get_level_with_same_class_different_shape, make_x
from games.minecraft.comp.compositional_config import HOUSE_SIZE, ROOF_HEIGHT, RR, generators, all_generators
from games.minecraft.minecraft_game import MinecraftGame
import minecraft_pb2_grpc
from minecraft_pb2 import *
import numpy as np
import grpc

channel = grpc.insecure_channel('localhost:5001')
client = minecraft_pb2_grpc.MinecraftServiceStub(channel)

SSSS = 250
L = HOUSE_SIZE

client.fillCube(FillCubeRequest(
    cube=Cube(
        min=Point(x=-50, y=0, z=-50),
        max=Point(x=SSSS, y=4, z=SSSS)
    ),
    type=GRASS
))

client.fillCube(FillCubeRequest(
    cube=Cube(
        min=Point(x=-50, y=4, z=-50),
        max=Point(x=SSSS, y=30, z=SSSS)
        # min=Point(x=-50, y=5, z=-50),
        # max=Point(x=150, y=40, z=50)
    ),
    type=AIR
))
sleep(3)
def make_level(name_of_thing_to_make):
    level = get_level(generators, name_of_thing_to_make, 1)
    print(level.map.shape)
    return level

def show_level(level,
                X_START=0, Y_START=0, Z_START = 0
               ):
    # client.fillCube(FillCubeRequest(
    #     cube=Cube(
    #         min=Point(x=-L + X_START, y=5, z=-L + Z_START),
    #         max=Point(x=L + X_START, y=100, z=L + Z_START)
    #     ),
    #     type=AIR
    # ))
    arr = level
    print(arr.shape)
    blocks_list = []
    for i in range(arr.shape[0]):
        for k in range(arr.shape[2]):
            for j in range(arr.shape[1]):
            # if j >= 5: continue
                blocks_list.append(
                    Block(position=Point(x=X_START + i - 5, y=4 + j + Y_START, z=Z_START + k - 5), type=int(AIR), orientation=NORTH),
                )
                blocks_list.append(
                    Block(position=Point(x=X_START + i - 5, y=4 + j + Y_START, z=Z_START + k - 5), type=int(arr[i, j, k]), orientation=NORTH),
                )
                # print(list(blocks_list))
                if 0 and len(blocks_list) >= arr.size // 20:
                    client.spawnBlocks(Blocks(blocks=blocks_list))
                    blocks_list = []
    if 0 and len(blocks_list) != 0:
        client.spawnBlocks(Blocks(blocks=blocks_list))
    return blocks_list


def main(name):
    # show_level(make_level(name).map)
    old = generators[name]['tiles']
    print("LL", len(generators[name]['all_generations']))
    for i in range(10):
        LIST = []
        for seed in range(5):
            for y_val in range(1):  
                if all_generators[seed][name] is None: continue
                # NET = all_generators[seed][name]['all_generations'][i]
                NET = all_generators[seed][name]['all_generations'][i]
                # NET = all_generators[seed][name]['all_networks'][i]
                def new(i):
                    if name != 'house': return old(i)
                    return old(i)[:1] + [[COBBLESTONE, BRICK_BLOCK, STONE, PLANKS, LOG][seed]] + old(i)[1:]
                all_generators[seed][name]['tiles'] = new
                # NET = generators[name]['all_generations'][i]
                # game = generators[name]['game']
                # _X, _Y, _Z = game.level.map.shape
                # gen = get_generator(MinecraftGame(get_level_with_same_class_different_shape(game, _X, i * 2 + 4, _Z)))
                for l in range(5 * (1 + (name == 'garden'))):
                    if name == 'garden':
                        game = generators[name]['game']
                        gen = get_generator(MinecraftGame(get_level_with_same_class_different_shape(game, HOUSE_SIZE, 1, HOUSE_SIZE)))
                        LIST.extend(show_level(make_x(name, 1, generators=all_generators[seed], net_to_override=NET, gen_to_override=gen), X_START=15 * i, Z_START=15*l, Y_START=1))
                    else:
                        LIST.extend(show_level(make_x(name, 1, generators=all_generators[seed], net_to_override=NET), X_START=25 * i, Z_START=25*seed, Y_START=l * 8))
            LLL = len(LIST) // 5
            KKKK = 2
            P = LLL // KKKK
            for j in range(KKKK):
                temp = []
                for ppp in range(5):
                    temp.extend(LIST[LLL * ppp: LLL * (ppp+1)][j * P: (j+1) * P])
                    if len(temp) > 100 and 0:
                        client.spawnBlocks(Blocks(blocks=temp))
                        temp = []
                # temp = LIST[P * j:(P * (j + 1)) * 10:LLL]
                # print(P * j, (P * (j + 1)), LLL)
                client.spawnBlocks(Blocks(blocks=temp))
                sleep(0.1)

def diff_sizes(name, roofs=False, decorations=False):
    sizes = [
        (10, 10, 10),
        (6, 8, 6),
        (20, 6, 10),
        (40, 6, 6),
        (10, 30, 6),
    ]
    mats = [
        BRICK_BLOCK,
        STONE,
        PLANKS,
        LOG,
        IRON_BLOCK
    ]
    T = 0
    for ppp, size in enumerate(sizes):
        LIST = []
        old = generators[name]['tiles']
        print("LL", len(generators[name]['all_generations']))
        seed = 0
        i = -1
        NET = all_generators[seed][name]['all_generations'][i]
        def new(i):
            # return old(i)[:1] + [[COBBLESTONE, BRICK_BLOCK, STONE, PLANKS, LOG][seed]] + old(i)[1:]
            return old(i)[:1] + [mats[ppp]] + old(i)[1:]
        all_generators[seed][name]['tiles'] = new
        NET = generators[name]['all_generations'][i]
        game = generators[name]['game']
        gen = get_generator(MinecraftGame(get_level_with_same_class_different_shape(game, *size)))
        HOUSE = make_x(name, 1, generators=all_generators[seed], net_to_override=NET, gen_to_override=gen)
        LIST.extend(show_level(HOUSE, X_START=T, Z_START=25*seed, Y_START=0))
        if roofs:
            LIST.extend(show_level(make_x('roof', 1, generators=all_generators[seed], net_to_override=all_generators[seed]['roof']['all_generations'][i], gen_to_override=get_generator(MinecraftGame(get_level_with_same_class_different_shape(generators['roof']['game'], size[0] + RR *2, ROOF_HEIGHT, size[-1]+ RR *2)))), X_START=T - RR, Z_START=25*seed - RR, Y_START=0 + size[1]))
        if decorations:
            t_start = HOUSE[1:-1, 1:-1, 1:-1]
            t_start[t_start == AIR] = 0
            t_start[t_start != AIR] = 1
            LIST.extend(show_level(make_x('decoration', 1, generators=all_generators[seed], net_to_override=all_generators[seed]['decoration']['all_generations'][i], gen_to_override=get_generator(MinecraftGame(get_level_with_same_class_different_shape(generators['decoration']['game'], size[0] -2, size[1] - 2, size[-1]-2))), start=t_start), X_START=T + 1, Z_START=25*seed + 1, Y_START=1))
        T += size[0] + 10
        client.spawnBlocks(Blocks(blocks=LIST))
        sleep(0.4)

if __name__ == '__main__':
    # main('house')
    main('garden')
    # diff_sizes('house', roofs=True, decorations=True)
    # diff_sizes_roofs('house')