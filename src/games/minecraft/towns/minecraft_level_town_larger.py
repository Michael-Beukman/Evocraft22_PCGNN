from typing import Tuple, Union
from matplotlib import pyplot as plt
import numpy as np
from games.level import Level
from games.minecraft.blocks import BLOCKS_TO_USE_REV

class MinecraftTownLarger(Level):
    """
        MinecraftTown
    """
    def __init__(self, width=10, height=1, depth=10):
        """Creates this level

        Args:
            width (int, optional): [description]. Defaults to 14.
            height (int, optional): [description]. Defaults to 14.
            start (Union[Tuple[int, int], None], optional): The start location of the agent. If None, then defaults to (0, 0). Defaults to None.
            end (Union[Tuple[int, int], None], optional): Goal location. If None, defaults to (width-1, height-1). Defaults to None.
        """
        
        super().__init__(width, height, tile_types={
            0: 'empty',
            1: 'house',
            2: 'house_2',
            3: 'house_3',
            4: 'garden',
            5: 'road'
        })
        self.depth = depth
        self.map = np.zeros((width, height, depth))
    
    @staticmethod
    def random(width=10, height=1, depth=10) -> "MinecraftTownLarger":
        level = MinecraftTownLarger(width, height, depth)
        level.map = (np.random.randint(0, len(level.tile_types), size=level.map.shape)).astype(np.int32)
        return level

    
    @staticmethod
    def from_map(map: np.ndarray, **kwargs) -> "MinecraftTownLarger":
        """Creates a level from an array representing the map.

        Returns:
            MazeLevel: 
        """
        level = MinecraftTownLarger(map.shape[0], map.shape[1], map.shape[2], **kwargs)
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