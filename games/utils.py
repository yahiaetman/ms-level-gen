from collections import Counter
import math
from typing import Optional, Tuple
from .game import Level

def entropy(level: Level, window_size: Tuple[int, int] = (2, 2), pad: Optional[int] = None) -> float:
    """Calculate the entropy for a level where the tokens are collected via a sliding window.

    Parameters
    ----------
    level : Level
        The level for which the entropy is computed.
    window_size : Tuple[int, int], optional
        The sliding window size (height, width). (Default: (2, 2))
    pad : Optional[int], optional
        A value to add as a padding around the level. If None, no padding is done. (Default: None)

    Returns
    -------
    float
        The entropy.
    """
    if len(level) == 0: return 0
    if pad is not None:
        level = [[pad, *row, pad] for row in level]
        extra = [pad]*len(level[0])
        level = [extra, *level, extra]
    th, tw = window_size
    col_groups = [list(zip(*[row[i:] for i in range(tw)])) for row in level]
    row_groups = list(zip(*[col_groups[i:] for i in range(th)]))
    groups = [group for bundle in row_groups for group in zip(*bundle)]
    total = len(groups)
    counter = Counter(groups)
    entropy = 0
    for _, frequency in counter.items():
        probability = frequency / total
        entropy += probability * math.log2(probability)
    return -entropy