      
from typing import Callable, Dict
import numpy as np
import tqdm
np.random.seed(1)
from games.minecraft.blocks import *

from games.minecraft.comp.compositional_shared import get_level, make_x
from games.minecraft.comp.compositional_config import RR, TILE_X, TILE_Z, generators, mapping, X, Y, Z

def is_house(tile_type: int):
    if 1 <= tile_type <= 3 and len(mapping) > 4: return True
    elif tile_type == 1: return True
    return False


arr = get_level(generators, 'town', 0).map
class Compositional:
    def __init__(self) -> None:
        pass

    def get_new_map(self, old_map: np.ndarray, mapping: Dict[int, Callable[[int], np.ndarray]], verbose=True):
        main = np.zeros((X, Y ,Z), dtype=np.uint8) + AIR
        lengths = np.zeros_like(old_map)
        assert old_map.shape[1] == 1
        pbar = tqdm.tqdm(total=old_map.shape[0] * old_map.shape[-1]) if verbose else None
        for i in range(old_map.shape[0]):
            for j in range(old_map.shape[-1]):
                if verbose: pbar.update(1)
                tile_type = old_map[i, 0, j]
                new_thing = mapping[tile_type](tile_type)
                if is_house(tile_type): new_thing[new_thing == AIR] = REDSTONE_BLOCK
                main[
                        i * TILE_X: (i+1) * TILE_X,
                        :new_thing.shape[1],
                        j * TILE_Z: (j+1) * TILE_Z,
                    ] = new_thing
                lengths[i, 0, j] = new_thing.shape[1]

                if is_house(tile_type):
                    temp = main[
                        i * TILE_X + 1: (i+1) * TILE_X - 1,
                        1:new_thing.shape[1] -1 ,
                        1 + j * TILE_Z: (j+1) * TILE_Z - 1,
                    ]
                    decorations = make_x('decoration', 6, start=temp, generators=generators)
                    decorations[decorations == AIR] = REDSTONE_BLOCK
                    main[
                        i * TILE_X + 1: (i+1) * TILE_X - 1,
                        1:new_thing.shape[1] -1 ,
                        1 + j * TILE_Z: (j+1) * TILE_Z - 1,
                    ] = decorations
        
        if verbose: pbar.close()
        for i in range(old_map.shape[0]):
            for j in range(old_map.shape[-1]):
                tile_type = old_map[i, 0, j]
                S = int(lengths[i, 0, j])
                if is_house(tile_type):
                    roof = roof = make_x('roof', tile_type, generators=generators)
                    if i == 0: roof = roof[RR:, :, :]
                    if i == old_map.shape[0] - 1: roof = roof[:-RR, :, :]
                        
                    if j == old_map.shape[-1] - 1: roof = roof[:, :, :-RR]
                    if j == 0: roof = roof[:, :, RR:]
                    MAIN_TEMP = main[
                            max(i * TILE_X - RR, 0): (i+1) * TILE_X + RR,
                            S:S + roof.shape[1],
                            max(j * TILE_Z - RR, 0): (j+1) * TILE_Z + RR
                        ]
                    idx = np.logical_and(np.logical_and(MAIN_TEMP != BRICK_BLOCK, MAIN_TEMP != COBBLESTONE), np.logical_and(MAIN_TEMP != LOG, MAIN_TEMP != REDSTONE_BLOCK))
                    main[
                            max(i * TILE_X - RR, 0): (i+1) * TILE_X + RR,
                            S:S + roof.shape[1],
                            max(j * TILE_Z - RR, 0): (j+1) * TILE_Z + RR
                        ][idx] = roof[idx]
                # Add in roofs
        # Vertical Walls
        
        main[main == REDSTONE_BLOCK] = AIR
        # return main
        old_map = old_map[:, 0, :]
        for i in range(1, X // TILE_X):
            p1 = i * TILE_X
            p2 = p1 - 1
            blocks = [BRICK_BLOCK, COBBLESTONE, LOG]
            for z in range(X):
                for y in range(1, generators['house']['game'].level.map.shape[1]):
                    if is_house(old_map[i, z // TILE_Z]) and is_house(old_map[i - 1, z // 10]) and ((main[p1, y, z] in blocks and main[p2, y, z] == AIR) or (main[p2, y, z] in blocks and main[p1, y, z] == AIR)):
                        main[p2:p1+1, y, z] = AIR
                        
                    if is_house(old_map[z // TILE_Z, i]) and is_house(old_map[z // TILE_Z, i - 1]) and \
                        ((main[z, y, p1] in blocks and main[z, y, p2] == AIR) or (main[z, y, p2] in blocks and main[z, y, p1] == AIR)):
                        main[z, y, p2:p1+1] = AIR        
        return main
if __name__ == '__main__':
    print("Generating")
    new = Compositional().get_new_map(arr, mapping)
    print("Done generating, now placing blocks in minecraft")
    np.save('test_mine.npy', new)