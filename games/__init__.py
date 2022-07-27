from dataclasses import dataclass, field
from typing import Any, Dict
from .game import Game
from .dave.dave import Dave
from .maze.maze import Maze
from .maze2.maze2 import Maze2
from .sokoban.sokoban import Sokoban
from .vampy.vampy import Vampy
from .zelda.zelda import Zelda
from . import utils

GAMES = {
    "dave": Dave,
    "maze": Maze,
    "maze2": Maze2,
    "sokoban": Sokoban,
    "vampy": Vampy,
    "zelda": Zelda,
}

def get_game_class_by_name(name: str):
    return GAMES[name.lower()]

@dataclass
class GameConfig:
    name: str
    options: Dict[str, Any] = field(default_factory=lambda:{})

def create_game(config: GameConfig) -> Game:
    return GAMES[config.name.lower()](**config.options)
