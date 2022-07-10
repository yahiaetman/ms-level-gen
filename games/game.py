from abc import ABC, abstractmethod, abstractproperty
import os, random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from .img_utils import load_sprite_atlas

Level = List[List[int]]
DatasetPostProcessor = Callable[[List[Level]], List[Level]]

class ConditionUtility(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.min = lambda x, y: (x if y is None else (y if x is None else min(x, y)))
        self.max = lambda x, y: (x if y is None else (y if x is None else min(x, y)))
        self.clamp = lambda x, xmin, xmax: self.max(self.min(x, xmax), xmin)
        self.round = round
        self.mul = lambda x, y: x*y
        self.const = lambda x: x
    
    @abstractmethod
    def get_snapping_function(self, prop_name: str) -> Callable[[float, Tuple[int, int]], float]:
        pass

    def get_tolerence(self, prop_name: str, size: Tuple[int, int]) -> float:
        return 1e-8
    
    def get_range_estimates(self, prop_name: str, size: Tuple[int, int]) -> Tuple[float, float]:
        return 0.0, 1.0

class Game(ABC):
    def __init__(self, name, tiles: str, sprite_atlas_path: str, dataset_postprocessors: Optional[Dict[str, DatasetPostProcessor]] = None) -> None:
        super().__init__()
        self.name = name
        self.__tiles = tiles
        self.__tile_to_index = {char.lower(): index for index, char in enumerate(tiles)}
        self.__sprite_atlas_path = sprite_atlas_path
        self.__sprite_atlas: np.ndarray = None
        self.__dataset_postprocessors = dataset_postprocessors or {}

    @property
    def tiles(self) -> str:
        return self.__tiles

    def parse_level(self, level: Union[str, List[str]]) -> Optional[Level]:
        if level is str:
            level = level.splitlines()
        if len(level) == 0: return None
        parsed = [[self.__tile_to_index.get(cell.lower()) for cell in row] for row in level]
        if any(cell is None for row in parsed for cell in row): return None
        if any(len(row) != len(parsed[0]) for row in parsed[1:]): return None
        return parsed
    
    def format_level(self, level: Level) -> str:
        return '\n'.join(''.join(self.tiles[cell] for cell in row) for row in level)
    
    def load_dataset(self, path: str) -> Tuple[List[Level], List[str]]:
        levels = []
        names = []
        files = [os.path.join(path, fname) for fname in os.listdir(path)] if os.path.isdir(path) else [path]
        for fname in files:
            with open(fname, 'r') as f:
                string = f.read()
            lines = string.splitlines(keepends=False)
            level_lines = []
            for line in lines:
                line = line.strip()
                if len(line) == 0: continue
                if line.startswith(';'):
                    if level_lines:
                        level = self.parse_level(level_lines)
                        if level is not None:
                            levels.append(level)
                            names.append(line[1:].strip())
                        level_lines.clear()
                else:
                    level_lines.append(line)
        return levels, names
    
    def save_dataset(self, path: str, levels: List[Level], names: Optional[List[str]] = None):
        names = names or [i for i in range(len(levels))]
        file_content = '\n\n'.join(f'{self.format_level(level)}\n; {name}' for name, level in zip(names, levels))
        with open(path, 'w') as f:
            f.write(file_content)

    @property
    def dataset_postprocessors(self) -> Dict[str, DatasetPostProcessor]:
        return self.__dataset_postprocessors

    @abstractmethod
    def analyze(self, levels: List[Level], **kwargs) -> List[Dict[str, Any]]:
        return [{"level": level, "compilable": False, "playable": False} for level in levels]

    @property
    def possible_augmentation_count(self) -> int:
        return 1
    
    def augment_level(self, level: Level, augnmentation_bits: Optional[int] = None) -> Level:
        return level
    
    def generate_random(self, level_count: int, size: Tuple[int, int], **kwargs) -> List[Level]:
        h, w = size
        return [[[random.randint(0, len(self.__tiles)-1) for _ in range(w)] for _ in range(h)] for _ in range(level_count)]

    def render(self, levels: np.ndarray, padding = 4) -> np.ndarray:
        if self.__sprite_atlas is None: self.__sprite_atlas = load_sprite_atlas(self.__sprite_atlas_path)
        sprite_atlas = self.__sprite_atlas
        assert levels.ndim >= 2, f"Expected levels to have 2 or more dimensions, get {levels.ndim} dimensions"
        if levels.ndim == 2: return self.render(levels[None,:,:])[0]
        *n, h, w = levels.shape
        images = np.empty((*n, 3, h*16, w*16), dtype=sprite_atlas.dtype)
        for index in np.ndindex(*n):
            level = levels[index]
            image = images[index]
            for i, row in enumerate(level):
                image_row = image[:, i*16:(i+1)*16]
                for j, tile in enumerate(row):
                    image_row[:, :, j*16:(j+1)*16] = sprite_atlas[tile]
        if padding is None:
            return images
        return np.pad(images, pad_width=((0,0),)*len(n) + ((0,0),(padding,padding),(padding,padding)), constant_values=255)

    @abstractproperty
    def condition_utility(self) -> ConditionUtility:
        pass