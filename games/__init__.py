from dataclasses import dataclass, field
from typing import Any, Dict

from common.config_tools import config
from .game import Game
from .dave.dave import Dave
from .maze.maze import Maze
from .maze2.maze2 import Maze2
from .sokoban.sokoban import Sokoban
from .vampy.vampy import Vampy
from .zelda.zelda import Zelda
from . import utils # Don't remove, this left here to be used by other files

GAMES = {
    "dave": Dave,
    "maze": Maze,
    "maze2": Maze2,
    "sokoban": Sokoban,
    "vampy": Vampy,
    "zelda": Zelda,
}

@config
@dataclass
class GameConfig:
    """A config class to define a game and its options.
    The name will specify the game's class as defined the global dict 'GAMES'.
    The options will define the arguments to send to the game's __init__ function.
    """
    name: str
    options: Dict[str, Any] = field(default_factory=lambda:{})

    def create(self) -> Game:
        """Create a game from this config.

        Returns
        -------
        Game
            The created game.
        """
        return GAMES[self.name.lower()](**self.options)
