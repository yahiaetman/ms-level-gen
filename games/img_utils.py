from typing import Optional, Tuple
import numpy as np
from PIL import Image
import math

def load_sprite_atlas(path: str) -> np.ndarray:
    sprite = Image.open(path)
    sprite = sprite.convert("RGB")
    sprite = np.array(sprite)
    h, nw, _ = sprite.shape
    assert nw%h==0, "The sprite must be square shaped"
    n = nw//h
    sprite = np.resize(sprite, (h, n, h, 3))
    sprite = np.transpose(sprite, (1, 3, 0, 2))
    return sprite

def save_images(path: str, images: np.ndarray, layout: Optional[Tuple[int, int]] = None):
    assert images.ndim >= 2, "Images must have at least 2 dimensions"
    h, w = images.shape[-2:]
    images = images.reshape((-1, 3, h, w))
    c = images.shape[0]
    if layout is None:
        r = math.floor(math.sqrt(c))
        while c%r != 0:
            r-=1
        layout = (r, c//r)
    images = images.transpose(2, 0, 3, 1)
    images = images.reshape(h, layout[0], layout[1]*w, 3)
    images = images.transpose(1, 0, 2, 3)
    images = images.reshape(layout[0]*h, layout[1]*w, 3)
    image = Image.fromarray(images)
    image.save(path)
