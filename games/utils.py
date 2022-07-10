from collections import Counter
import math
from typing import List, Optional, Tuple

def entropy(level: List[List[int]], window_size: Tuple[int, int] = (2, 2), pad: Optional[int] = None):
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