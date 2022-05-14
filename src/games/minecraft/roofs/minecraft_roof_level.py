from typing import Tuple, Union
from matplotlib import pyplot as plt
import numpy as np
from games.level import Level
from games.minecraft.blocks import BLOCKS_TO_USE_REV


class MinecraftRoofLevel(Level):
    def __init__(self, width=14, height=6, depth=14):

        super().__init__(width, height, tile_types={
            0: 'empty',
            1: 'material',
        })
        self.depth = depth
        self.map = np.zeros((width, height, depth))
    
    @staticmethod
    def random(width=14, height=6, depth=14) -> "MinecraftRoofLevel":
        level = MinecraftRoofLevel(width, height, depth)
        level.map = (np.random.randint(0, len(level.tile_types), size=level.map.shape)).astype(np.int32)
        return level

    
    @staticmethod
    def from_map(map: np.ndarray, **kwargs) -> "MinecraftRoofLevel":
        """Creates a level from an array representing the map.

        Returns:
            MazeLevel: 
        """
        level = MinecraftRoofLevel(map.shape[0], map.shape[1], map.shape[2], **kwargs)
        level.map = map
        return level

    
    def to_file(self, filename: str):
        with open(filename, 'w+') as f:
            f.write(self.str())
    
    def str(self) -> str:
        ans = ""
        for row in self.map:
            for c in row:
                ans += '#' if c == 1 else '.'
            ans += '\n'
        return ans