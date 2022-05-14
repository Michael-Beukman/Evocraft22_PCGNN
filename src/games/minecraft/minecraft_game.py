from typing import Tuple

import numpy as np
from games.game import Game
from games.level import Level


class MinecraftGame(Game):
    """
    Minecraft
    """
    
    def reset(self, level: Level):
        """Resets this env given the level. The player now starts at the original spot again.

        Args:
            level (Level): The level to use.
        """
        self.level = level
        self.current_pos = level.start
        
    def __init__(self, level: Level):
        super().__init__(level)