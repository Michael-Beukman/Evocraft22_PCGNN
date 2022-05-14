from typing import Tuple, Union
from matplotlib import pyplot as plt
import numpy as np
from games.level import Level
from games.minecraft.blocks import BLOCKS_TO_USE_REV


class MinecraftHouseLevel(Level):
    def __init__(self, width=10, height=10, depth=10):
        super().__init__(width, height, tile_types={
            0: 'empty',
            1: 'brick',
            2: 'wood',
            3: 'stone',
        })
        self.depth = depth
        self.map = np.zeros((width, height, depth))
    
    @staticmethod
    def random(width=10, height=10, depth=10) -> "MinecraftHouseLevel":
        level = MinecraftHouseLevel(width, height, depth)
        level.map = (np.random.randint(0, len(level.tile_types), size=level.map.shape)).astype(np.int32)
        return level

    
    @staticmethod
    def from_map(map: np.ndarray, **kwargs) -> "MinecraftHouseLevel":
        """Creates a level from an array representing the map.

        Returns:
            MazeLevel: 
        """
        level = MinecraftHouseLevel(map.shape[0], map.shape[1], map.shape[2], **kwargs)
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