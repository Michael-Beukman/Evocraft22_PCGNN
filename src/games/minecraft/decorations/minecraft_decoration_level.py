from typing import Tuple, Union
from matplotlib import pyplot as plt
import numpy as np
from games.level import Level
from games.minecraft.blocks import BLOCKS_TO_USE_REV

class MinecraftDecorationLevel(Level):
    def __init__(self, width=8, height=8, depth=8):
        super().__init__(width, height, tile_types={
            0: 'air',
            1: 'torch',
            2: 'bookcase',
            3: 'crafting_table',
            4: 'jukebox'
        })
        self.depth = depth
        self.map = np.zeros((width, height, depth))
    
    @staticmethod
    def random(width=8, height=8, depth=8) -> "MinecraftDecorationLevel":
        level = MinecraftDecorationLevel(width, height, depth)
        level.map = (np.random.randint(0, len(level.tile_types), size=level.map.shape)).astype(np.int32)
        return level

    
    @staticmethod
    def from_map(map: np.ndarray, **kwargs) -> "MinecraftDecorationLevel":
        """Creates a level from an array representing the map.

        Returns:
            MazeLevel: 
        """
        level = MinecraftDecorationLevel(map.shape[0], map.shape[1], map.shape[2], **kwargs)
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