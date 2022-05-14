from typing import Tuple, Union
from matplotlib import pyplot as plt
import numpy as np
from games.level import Level
from games.minecraft.blocks import BLOCKS_TO_USE_REV


class MinecraftLevel(Level):
    """
        Minecraft
    """
    def __init__(self, width=10, height=10, depth=10,
                        start: Union[Tuple[int, int], None] = None,
                        end: Union[Tuple[int, int], None]  = None):
        """Creates this level

        Args:
            width (int, optional): [description]. Defaults to 14.
            height (int, optional): [description]. Defaults to 14.
            start (Union[Tuple[int, int], None], optional): The start location of the agent. If None, then defaults to (0, 0). Defaults to None.
            end (Union[Tuple[int, int], None], optional): Goal location. If None, defaults to (width-1, height-1). Defaults to None.
        """
        
        super().__init__(width, height, tile_types=BLOCKS_TO_USE_REV)
        self.depth = depth
        self.start = start if start is not None else (0, 0)
        self.end = end if end is not None else  (width - 1, height - 1)
        self.map = np.zeros((width, height, depth))
    
    @staticmethod
    def random(width=10, height=10, depth=10) -> "MinecraftLevel":
        level = MinecraftLevel(width, height, depth)
        level.map = (np.random.randint(0, len(BLOCKS_TO_USE_REV), size=level.map.shape)).astype(np.int32)
        return level

    
    @staticmethod
    def from_map(map: np.ndarray, **kwargs) -> "MinecraftLevel":
        """Creates a level from an array representing the map.

        Returns:
            MazeLevel: 
        """
        level = MinecraftLevel(map.shape[0], map.shape[1], map.shape[2], **kwargs)
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