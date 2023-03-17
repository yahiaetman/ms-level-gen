from dataclasses import dataclass, field
from typing import Any, Dict

from common.config_tools import config
from .game import Game
from . import utils # Don't remove, this left here to be used by other files

def __game_getter(mod: str, cls: str):
    """To avoid importing all the games, this function returns
    a function that imports the game class on demand.

    Parameters
    ----------
    mod : str
        The relative module path that contains the game class
    cls : str
        The class name

    Returns
    -------
    Callable[Type[Game]]
        A function that returns the game class
    """
    return lambda: getattr(__import__(mod, globals(), locals(), level=1, fromlist=[cls]), cls)

GAMES = {
    "dave": __game_getter("dave.dave", "Dave"),
    "maze": __game_getter("maze.maze", "Maze"),
    "maze2": __game_getter("maze2.maze2", "Maze2"),
    "maze3": __game_getter("maze3.maze3", "Maze3"),
    "sokoban": __game_getter("sokoban.sokoban", "Sokoban"),
    "vampy": __game_getter("vampy.vampy", "Vampy"),
    "zelda": __game_getter("zelda.zelda", "Zelda"),
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
        cls = GAMES[self.name.lower()]()
        return cls(**self.options)
