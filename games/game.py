from abc import ABC, abstractmethod, abstractproperty
import os, random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from .img_utils import load_sprite_atlas

# Common Type Definitions
Level = List[List[int]] # A Level is a 2D Array of integers (a list of lists)
DatasetPostProcessor = Callable[[List[Level]], List[Level]] 

class ConditionUtility(ABC):
    """This abstract class defines some functions needed to deal with condition (control) related tasks.

    The main functions it provides are:
    1-  Create a snapping function for a given property and optionally, at a given size.
        A snapping function attempts to snap any value to the nearest valid value for the given property (and size).
        For example, the wall ratio (#wall / area) for sokoban would only have valid values in the range [0, (area-3)/area]
        where the values can only be multiples of 1/area. Sometimes, we do not know one or both of the range bounds, so
        they can be ignored.
    2-  Define the tolerance for a given property at a given size. In other words, two values x and y are considered
        the same if |x-y| <= the tolerance.
    3-  A range (or an estimate for the range) of the possible values for a given property at a given size.

    NOTES:
    -   We do not need both the snapping function and the tolerance for each properties. Having only one of them is enough.
        They are used by the training algorithm to know if the requested and the generated level property are equal or not.
        If the snapping function is available, we can leave the tolerance at epsilon. If the tolerance is defined, we could
        leave the snapping function as the identity function (although we would lose the clamping to range but this does not
        cause any notable harm to the training process).
    -   The range estimate does not need to be accurate. Actually, you can leave it at [0, 1] if you don't want to override
        it. It is only used to defined a uniform distribution for the controls during the time where the replay buffer for
        the seed size is empty and the controls we give to the generator does not matter yet because it had not learned to
        generate anything playable. So, this function is basically here just for completeness.
    """

    def __init__(self) -> None:
        super().__init__()
        # The following function will be used by the snapping functions
        # They can be replaced by similar functions from pytorch or numpy to work on batches of data.
        self.min = lambda x, y: (x if y is None else (y if x is None else min(x, y)))
        self.max = lambda x, y: (x if y is None else (y if x is None else min(x, y)))
        self.clamp = lambda x, xmin, xmax: self.max(self.min(x, xmax), xmin)
        self.round = round
        self.mul = lambda x, y: x*y
        self.const = lambda x: x
    
    def for_torch(self):
        """Replace the math functions used by the snappers to ones from pytorch.
        """
        import torch
        self.round = torch.round
        self.clamp = torch.clamp
        self.min = torch.minimum
        self.max = torch.maximum
        self.const = torch.tensor
    
    def for_numpy(self):
        """Replace the math functions used by the snappers to ones from numpy.
        """
        import numpy as np
        self.round = np.round
        self.clamp = np.clip
        self.min = np.minimum
        self.max = np.maximum
        self.const = np.array
    
    def get_snapping_function(self, prop_name: str, size: Optional[Tuple[int, int]] = None) -> Union[Callable[[float, Tuple[int, int]], float], Callable[[float], float]]:
        """Create a snapping function for a given property and optionally, at a given size.

        Parameters
        ----------
        prop_name : str
            The property name.
        size : Optional[Tuple[int, int]], optional
            The level size, by default None

        Returns
        -------
        Union[Callable[[float, Tuple[int, int]], float], Callable[[float], float]]
            The snapping function. If the size is given, the snapping function will have the
            signature (property_value) -> snapped_property_value. Otherwise, the signature
            wiil be (property_value, size) -> snapped_property_value.
        """       
        return (lambda x, _: x) if size is None else (lambda x: x)

    def get_tolerence(self, prop_name: str, size: Tuple[int, int]) -> float:
        """Returns the tolerance for a given property at a given size.

        Parameters
        ----------
        prop_name : str
            The property name.
        size : Tuple[int, int]
            The level size.

        Returns
        -------
        float
            The tolerance value.
        """
        return 1e-8
    
    def get_range_estimates(self, prop_name: str, size: Tuple[int, int]) -> Tuple[float, float]:
        """Returns the range (or an estimate for it) for a given property at a given size.

        Parameters
        ----------
        prop_name : str
            The property name.
        size : Tuple[int, int]
            The level size.

        Returns
        -------
        Tuple[float, float]
            The range bounds.
        """
        return 0.0, 1.0

class Game(ABC):
    """The base class for all the games.
    """

    def __init__(self, name, tiles: str, sprite_atlas_path: str, dataset_postprocessors: Optional[Dict[str, DatasetPostProcessor]] = None) -> None:
        """The init function.

        Parameters
        ----------
        name : _type_
            The game's name.
        tiles : str
            A string containing the characters for all the game's tiles.
            The order is relevant since the integer assigned to each tile is its index in this string.
        sprite_atlas_path : str
            The path to the sprite atlas image. The sprites must have the same size and be arranged
            in a single row without padding in the same order of the tiles string.
        dataset_postprocessors : Optional[Dict[str, DatasetPostProcessor]], optional
            A dictionary of functions that can be used to postprocess a level dataset. (Default: None)
        """
        super().__init__()
        self.name = name
        self.__tiles = tiles
        self.__tile_to_index = {char.lower(): index for index, char in enumerate(tiles)}
        self.__sprite_atlas_path = sprite_atlas_path
        self.__sprite_atlas: np.ndarray = None
        self.__dataset_postprocessors = dataset_postprocessors or {}

    @property
    def tiles(self) -> str:
        """Returns a string containing the characters for all the game's tiles.
        """
        return self.__tiles

    def parse_level(self, level: Union[str, List[str]]) -> Optional[Level]:
        """Parses a string into a level. The inverse of this function is 'format_level'.

        Parameters
        ----------
        level : Union[str, List[str]]
            A string (or a list of string where each string is a line) containing the level
            and consisting of characters from 'tiles'.

        Returns
        -------
        Optional[Level]
            A level (or None if the input was empty).
        """
        if level is str:
            level = level.splitlines()
        if len(level) == 0: return None
        parsed = [[self.__tile_to_index.get(cell.lower()) for cell in row] for row in level]
        if any(cell is None for row in parsed for cell in row): return None
        if any(len(row) != len(parsed[0]) for row in parsed[1:]): return None
        return parsed
    
    def format_level(self, level: Level) -> str:
        """Formats a level as a string. The inverse of this function is 'parse_level'.

        Parameters
        ----------
        level : Level
            The level.

        Returns
        -------
        str
            A string representing the level.
        """
        return '\n'.join(''.join(self.tiles[cell] for cell in row) for row in level)
    
    def load_dataset(self, path: str) -> Tuple[List[Level], List[str]]:
        """Reads a level dataset from a file.

        The file should consists of zero or more levels where each level is defined as follows:

        ####### 
        #.....#
        #.01A.#     <- This is a 5x7 level for a game whose tiles contains '#', '.', '0', '1', 'A'.
        #.....#
        #######
        ; level name    <- This delimits the level and defines the level name (it can contain spaces).

        Empty lines are ignored.

        Parameters
        ----------
        path : str
            The path to the level dataset.

        Returns
        -------
        Tuple[List[Level], List[str]]
            A list of levels and a list of the corresponding names.
        """
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
        """Save a level dataset to a file. See docs 'load_dataset' to read about the file format.

        Parameters
        ----------
        path : str
            The path to the level dataset to be saved.
        levels : List[Level]
            A list of levels.
        names : Optional[List[str]], optional
            A list of the corresponding names. If None, the levle index will be used as its name. (Default: None).
        """
        names = names or [i for i in range(len(levels))]
        file_content = '\n\n'.join(f'{self.format_level(level)}\n; {name}' for name, level in zip(names, levels))
        with open(path, 'w') as f:
            f.write(file_content)

    @property
    def dataset_postprocessors(self) -> Dict[str, DatasetPostProcessor]:
        """A dictionary of functions that can be used to postprocess a level dataset.
        """
        return self.__dataset_postprocessors

    @abstractmethod
    def analyze(self, levels: List[Level], **kwargs) -> List[Dict[str, Any]]:
        """Given a list of level, this function analyzes the levels and returns a dictionary
        of information about each level.

        The dictionary should at least contain the following:
        - level: the level itself.
        - solvable: a boolean stating if this level can be won. If it is impossible to win, it
            should be false.

        Parameters
        ----------
        levels : List[Level]
            A list of levels to analyze.

        Returns
        -------
        List[Dict[str, Any]]
            A list of information about the given levels.
        """
        return [{"level": level, "solvable": False} for level in levels]

    @property
    def possible_augmentation_count(self) -> int:
        """The possible number of data augmentations for a level of this game.
        The number includes the original level, so if the level cannot be modified
        for data augmentation, return 1. 
        """
        return 1
    
    def augment_level(self, level: Level, augnmentation_index: Optional[int] = None) -> Level:
        """Apply a data augmentation process to the given level.
        This function does not modify the level in-place.

        Parameters
        ----------
        level : Level
           The level to be modified.
        augnmentation_index : Optional[int], optional
            The index of the augmentation process. If None, a process will be picked randomly. (Default: None)

        Returns
        -------
        Level
            The modified level.
        """
        return level
    
    def generate_random(self, level_count: int, size: Tuple[int, int], **kwargs) -> List[Level]:
        """Generate a random list of levels. This used as a baseline to make sure that the generators
        do not perform worse than random.

        Parameters
        ----------
        level_count : int
            The number of requested levels.
        size : Tuple[int, int]
            The sizes of the requested levels.

        Returns
        -------
        List[Level]
            A list of the randomly generated levels.
        """
        h, w = size
        return [[[random.randint(0, len(self.__tiles)-1) for _ in range(w)] for _ in range(h)] for _ in range(level_count)]

    def render(self, levels: np.ndarray, padding = 4) -> np.ndarray:
        """Renders an array of levels into an array of images.

        Parameters
        ----------
        levels : np.ndarray
            An array of level (dtype=int) with the shape (Batch_Size x Height x Width). 
        padding : int, optional
            The amount of white padding to add around the images. (Default: 4).

        Returns
        -------
        np.ndarray
            The generated images with the shape (Batch_Size x 3 x Sprite_Height * Height x Sprite_Width * Width)
        """
        if self.__sprite_atlas is None: self.__sprite_atlas = load_sprite_atlas(self.__sprite_atlas_path)
        sprite_atlas = self.__sprite_atlas
        assert levels.ndim >= 2, f"Expected levels to have 2 or more dimensions, get {levels.ndim} dimensions"
        if levels.ndim == 2: return self.render(levels[None,:,:])[0]
        *n, h, w = levels.shape
        *_, sh, sw = self.__sprite_atlas.shape
        images = np.empty((*n, 3, h*sh, w*sw), dtype=sprite_atlas.dtype)
        for index in np.ndindex(*n):
            level = levels[index]
            image = images[index]
            for i, row in enumerate(level):
                image_row = image[:, i*sw:(i+1)*sw]
                for j, tile in enumerate(row):
                    image_row[:, :, j*sh:(j+1)*sh] = sprite_atlas[tile]
        if padding is None:
            return images
        return np.pad(images, pad_width=((0,0),)*len(n) + ((0,0),(padding,padding),(padding,padding)), constant_values=255)

    @abstractproperty
    def condition_utility(self) -> ConditionUtility:
        """Creates and returns a condition utility class for this game.
        """
        pass