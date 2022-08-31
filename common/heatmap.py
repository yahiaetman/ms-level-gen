from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from common.config_tools import config

# Given a string for an expression that requires a size,
# wraps it in a lambda with the size as an argument then call it with the given size
# This is useful to convert an expression that requires a size into one that does not require it
def wrap_with_size(size: Tuple[int, int], fn: str):
    return f"(lambda size: {fn})({size})"

# This class represents a heatmap for a given set of levels
# The update function adds more levels to the heatmap
# The render function draws the heatmap as a matplotlib figure
class Heatmap:
    @config
    @dataclass
    class Config:
        labels: Tuple[str, str]                         # The labels of the y and x axes
        coordinates: Tuple[str, str]                    # The expression for the y and x values for a given level
        bounds: Tuple[Tuple[int, int], Tuple[int, int]] # The minimum and maximum bounds for the heatmap on the y and x axes
                                                        # The data represents ((y_min, y_max), (x_min, x_max))
    
    def __init__(self, config: Heatmap.Config) -> None:
        self.labels = config.labels
        self.y_fn = eval(f"lambda item: {config.coordinates[0]}")
        self.x_fn = eval(f"lambda item: {config.coordinates[1]}")
        
        (y_min, y_max), (x_min, x_max) = config.bounds
        h, w = (y_max - y_min + 1), (x_max - x_min + 1)

        self.items = [[[] for _ in range(w)] for _ in range(h)]
        self.bounds = config.bounds
    
    # Add levels to the heatmap. Only solvable levels are added. 
    def update(self, info: List[Dict]):
        items = self.items
        (y_min, y_max), (x_min, x_max) = self.bounds
        for item in info:
            if not item["solvable"]: continue
            y = min(max(self.y_fn(item), y_min), y_max) - y_min
            x = min(max(self.x_fn(item), x_min), x_max) - x_min
            items[y][x].append(item)

    # render the heatmap where the value is specified by the "stat_fn" and
    # the mask (if true, the cell is skipped) is specified by the "mask_fn"
    def render(self, stat_fn = len, mask_fn = (lambda l: len(l) == 0)):
        items = self.items
        (y_min, _), (x_min, _) = self.bounds
        
        stat = np.array([[stat_fn(cell) for cell in row] for row in items])
        mask = np.array([[mask_fn(cell) for cell in row] for row in items], dtype=bool)
        masked: np.ndarray = stat[~mask]
        if masked.size > 0:
            v_min = masked.min()
            v_max = masked.max()
        else:
            v_min, v_max = 0, 0

        fig = plt.figure()
        ax = sns.heatmap(stat, mask=mask, vmin=v_min, vmax=v_max, rasterized=True)
        ax.invert_yaxis()
        font_size = 6
        xticks, yticks = ax.get_xticks(), ax.get_yticks()
        ax.set_xticks(xticks)
        ax.set_xticklabels([int(i+x_min) for i in xticks], fontsize=font_size, rotation=90)
        ax.set_yticks(yticks)
        ax.set_yticklabels([int(i+y_min) for i in yticks], fontsize=font_size, rotation=0)
        ax.set_xlabel(self.labels[1], fontsize=font_size)
        ax.set_ylabel(self.labels[0], fontsize=font_size)
        
        return fig

# This class represents a set of heatmap for a given set of levels of multiple sizes
# The update function adds more levels to the heatmap of the corresponding size 
# The render function draws a heatmap as a matplotlib figure
class Heatmaps:
    @config
    @dataclass
    class Config:
        labels: Tuple[str, str]                                                 # The labels of the y and x axes
        coordinates: Tuple[str, str]                                            # The expression for the y and x values for a given level
        bounds: Dict[Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]]  # The minimum and maximum bounds for each heatmap on the y and x axes
                                                                                # The data represents {size: ((y_min, y_max), (x_min, x_max))}

    def __init__(self, config: Heatmaps.Config) -> None:
        self.config = config 
        self.heatmaps: Dict[Tuple[int, int], Heatmap] = {}
    
    def __getitem__(self, size: Tuple[int, int]) -> Heatmap:
        # The heatmap is lazily constructed when first requested
        heatmap = self.heatmaps.get(size)
        if heatmap is None:
            # If the configuration does not contain information about the heatmap bounds, the nearest size found is picked
            bounds = self.config.bounds.get(size)
            if bounds is None:
                h, w = size
                _, _, bounds = min((abs(hi-h)+abs(wi-w), (hi, wi), boundsi) for (hi, wi), boundsi in self.config.bounds.items())
            coordinates = tuple(wrap_with_size(size, fn) for fn in self.config.coordinates)
            heatmap = Heatmap(Heatmap.Config(self.config.labels, coordinates, bounds))
            self.heatmaps[size] = heatmap
        return heatmap
    
    # Added levels to the heatmap specified by the given size
    def update(self, size: Tuple[int, int], info: List[Dict]):
        heatmap = self[size]
        heatmap.update(info)
        
    # Render the heatmap for the given size where the value is specified by the "stat_fn" and
    # the mask (if true, the cell is skipped) is specified by the "mask_fn"
    def render(self, size: Tuple[int, int], stat_fn = len, mask_fn = (lambda l: len(l) == 0)):
        return self[size].render(stat_fn, mask_fn)

    # Render all the heatmaps (if the heatmap was not requested before, it will not be available in the results of this function)
    def render_all(self, stat_fn = len, mask_fn = (lambda l: len(l) == 0)):
        return {size:heatmap.render(stat_fn, mask_fn) for size, heatmap in self.heatmaps.items()}

