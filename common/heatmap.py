from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from common.config_tools import config

class Heatmaps:
    @config
    @dataclass
    class Config:
        labels: Tuple[str, str]
        coordinates: Tuple[str, str]
        bounds: Dict[Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]]

    @dataclass
    class Heatmap:
        frequency_map: np.ndarray
        y_bounds: Tuple[int, int]
        x_bounds: Tuple[int, int]

    def __init__(self, config: Heatmaps.Config) -> None:
        self.labels = config.labels
        self.y_fn = eval(f"lambda item, size: {config.coordinates[0]}")
        self.x_fn = eval(f"lambda item, size: {config.coordinates[1]}")
        self.heatmaps: Dict[Tuple[int, int], Heatmaps.Heatmap] = {}
        for size, bounds in config.bounds.items():
            (y_min, y_max), (x_min, x_max) = bounds
            h, w = (y_max - y_min + 1), (x_max - x_min + 1)
            frequency_map = np.zeros((h, w), dtype=int)
            self.heatmaps[size] = Heatmaps.Heatmap(frequency_map, *bounds)
        
    def update(self, size: Tuple[int, int], info: List[Dict]):
        heatmap = self.heatmaps[size]
        frequency_map = heatmap.frequency_map
        (y_min, y_max) = heatmap.y_bounds
        (x_min, x_max) = heatmap.x_bounds
        for item in info:
            if not item["solvable"]: continue
            y = min(max(self.y_fn(item, size), y_min), y_max) - y_min
            x = min(max(self.x_fn(item, size), x_min), x_max) - x_min
            frequency_map[y, x] += 1
    
    def render(self, size: Tuple[int, int]):
        heatmap = self.heatmaps[size]

        frequency_map = heatmap.frequency_map
        (x_min, _) = heatmap.x_bounds
        (y_min, _) = heatmap.y_bounds

        mask = frequency_map == 0
        v_min = 1
        v_max = max(1, frequency_map.max())

        fig = plt.figure()
        ax = sns.heatmap(frequency_map, mask=mask, vmin=v_min, vmax=v_max)
        ax.xaxis.set_ticks_position('top')
        font_size = 6
        xticks, yticks = ax.get_xticks(), ax.get_yticks()
        ax.set_xticks(xticks)
        ax.set_xticklabels([int(i+x_min) for i in xticks], fontsize=font_size, rotation=90)
        ax.set_yticks(yticks)
        ax.set_yticklabels([int(i+y_min) for i in yticks], fontsize=font_size, rotation=0)
        ax.set_xlabel(self.labels[1], fontsize=font_size)
        ax.set_ylabel(self.labels[0], fontsize=font_size)
        

        return fig

        
