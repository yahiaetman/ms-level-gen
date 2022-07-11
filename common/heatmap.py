from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from common.config_tools import config

def wrap_with_size(size: Tuple[int, int], fn: str):
    return f"(lambda size: {fn})({size})"

class Heatmap:
    @config
    @dataclass
    class Config:
        labels: Tuple[str, str]
        coordinates: Tuple[str, str]
        bounds: Tuple[Tuple[int, int], Tuple[int, int]]
    
    def __init__(self, config: Heatmap.Config) -> None:
        self.labels = config.labels
        self.y_fn = eval(f"lambda item: {config.coordinates[0]}")
        self.x_fn = eval(f"lambda item: {config.coordinates[1]}")
        
        (y_min, y_max), (x_min, x_max) = config.bounds
        h, w = (y_max - y_min + 1), (x_max - x_min + 1)

        self.frequency_map = np.zeros((h, w), dtype=int)
        self.bounds = config.bounds
    
    def update(self, info: List[Dict]):
        frequency_map = self.frequency_map
        (y_min, y_max), (x_min, x_max) = self.bounds
        for item in info:
            if not item["solvable"]: continue
            y = min(max(self.y_fn(item), y_min), y_max) - y_min
            x = min(max(self.x_fn(item), x_min), x_max) - x_min
            frequency_map[y, x] += 1

    def render(self):
        frequency_map = self.frequency_map
        (y_min, _), (x_min, _) = self.bounds
        
        mask = frequency_map == 0
        v_min = 1
        v_max = max(1, frequency_map.max())

        fig = plt.figure()
        ax = sns.heatmap(frequency_map, mask=mask, vmin=v_min, vmax=v_max, rasterized=True)
        #ax.xaxis.set_ticks_position('top')
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

class Heatmaps:
    @config
    @dataclass
    class Config:
        labels: Tuple[str, str]
        coordinates: Tuple[str, str]
        bounds: Dict[Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]]

    def __init__(self, config: Heatmaps.Config) -> None:
        self.config = config 
        self.heatmaps: Dict[Tuple[int, int], Heatmap] = {}
    
    def __getitem__(self, size: Tuple[int, int]) -> Heatmap:
        heatmap = self.heatmaps.get(size)
        if heatmap is None:
            bounds = self.config.bounds.get(size)
            if bounds is None:
                h, w = size
                _, bounds = min((abs(hi-h)+abs(wi-w), boundsi) for (hi, wi), boundsi in self.config.bounds.items())
            coordinates = tuple(wrap_with_size(size, fn) for fn in self.config.coordinates)
            heatmap = Heatmap(Heatmap.Config(self.config.labels, coordinates, bounds))
            self.heatmaps[size] = heatmap
        return heatmap
        
    def update(self, size: Tuple[int, int], info: List[Dict]):
        heatmap = self[size]
        heatmap.update(info)
        
    
    def render(self, size: Tuple[int, int]):
        return self[size].render()

        
