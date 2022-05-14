from typing import Tuple, Union
from matplotlib import pyplot as plt
import numpy as np
from games.level import Level
from games.minecraft.blocks import BLOCKS_TO_USE_REV

class MinecraftGardenLevel(Level):
    def __init__(self, width=10, height=1, depth=10):
       
        super().__init__(width, height, tile_types={
            0: 'air', # change dirt block to grass (pull one block more done)
            1: 'oak',
            2: 'red flower',
            3: 'yellow flower',
            4: 'water', # place block below it to be water (pull one block more down)
        })
        self.depth = depth
        self.map = np.zeros((width, height, depth))
    
    @staticmethod
    def random(width=10, height=1, depth=10) -> "MinecraftGardenLevel":
        level = MinecraftGardenLevel(width, height, depth)
        level.map = (np.random.randint(0, len(level.tile_types), size=level.map.shape)).astype(np.int32)
        return level

    
    @staticmethod
    def from_map(map: np.ndarray, **kwargs) -> "MinecraftGardenLevel":
        """Creates a level from an array representing the map.

        Returns:
            MazeLevel: 
        """
        level = MinecraftGardenLevel(map.shape[0], map.shape[1], map.shape[2], **kwargs)
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
  
  
class OldMinecraftGardenLevel(Level):
    """
        MinecraftGarden
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
            0: 'air', # change dirt block to grass (pull one block more done)
            1: 'oak',
            2: 'oak2',
            3: 'oak3',
            4: 'red flower',
            5: 'yellow flower',
            6: 'water', # place block below it to be water (pull one block more down)
        })
        self.depth = depth
        self.map = np.zeros((width, height, depth))
    
    @staticmethod
    def random(width=10, height=1, depth=10) -> "OldMinecraftGardenLevel":
        level = OldMinecraftGardenLevel(width, height, depth)
        level.map = (np.random.randint(0, len(level.tile_types), size=level.map.shape)).astype(np.int32)
        return level

    
    @staticmethod
    def from_map(map: np.ndarray, **kwargs) -> "OldMinecraftGardenLevel":
        """Creates a level from an array representing the map.

        Returns:
            MazeLevel: 
        """
        level = MinecraftGardenLevel(map.shape[0], map.shape[1], map.shape[2], **kwargs)
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
    
MinecraftGarden = MinecraftGardenLevel
