from typing import Optional, Tuple
import numpy as np
from PIL import Image
import math

def load_sprite_atlas(path: str) -> np.ndarray:
    """Read a sprite atlas from an image file and return it as
    a numby array in the shape (Sprites x Channels x Height x Width).

    Parameters
    ----------
    path : str
        The path to the sprite atlas image.

    Returns
    -------
    np.ndarray
        The sprite atlas as a numpy array in the shape (Sprites x Channels x Height x Width).
    """
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
    """Given an array of images, this function organized into a 2D layout
    and saves them to an image file.

    Parameters
    ----------
    path : str
        The path to the image to be saved.
    images : np.ndarray
        An array of images in the shape (* x Channels x Height x Width).
    layout : Optional[Tuple[int, int]], optional
        The 2D layout defined as (rows, columns). (rows * columns) must be
        equal the number of supplied images. If None, an automatic layout
        will be generated that is most square like (minimizes |rows - columns|)
        where rows <= columns. (Default: None)
    """
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
