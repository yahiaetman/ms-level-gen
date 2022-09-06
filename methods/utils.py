from typing import Iterable, Tuple

def snake(height: int, width: int):
    """Traverses a 2D map of size (height, width) row-wise  in 
    a snake like pattern and yields the current coordinate (x, y).

    Parameters
    ----------
    height : int
        The map height (number of rows).
    width : int
        The map width (number of columns).

    Yields
    ------
    Tuple[int, int]
        The x and y coordinates.
    """
    reverse_if = lambda cond, it: reversed(it) if cond else it
    for y in range(height):
        for x in reverse_if(y&1!=0, range(width)):
            yield (y, x)

def find_closest_size(size: Tuple[int, int], others: Iterable[Tuple[int, int]]) -> Tuple[int, int]:
    """Find the closest size in a list and return it.
    The distance metric is |Hi - H| + |Wi - W|

    Parameters
    ----------
    size : Tuple[int, int]
        The query size
    others : Iterable[Tuple[int, int]]
        A list of the available sizes

    Returns
    -------
    Tuple[int, int]
        The closest size found.
    """
    h, w = size
    _, _, closest = min(( (abs(h-hi)+abs(w-wi), i, (hi,wi)) for i, (hi, wi) in enumerate(others) ), default=(None, None, None))
    return closest

def find_closest_smaller_size(size: Tuple[int, int], others: Iterable[Tuple[int, int]]) -> Tuple[int, int]:
    """Find the closest size in a list where (Hi <= H) and (Wi <= W) and return it.
    The distance metric is |Hi - H| + |Wi - W|

    Parameters
    ----------
    size : Tuple[int, int]
        The query size
    others : Iterable[Tuple[int, int]]
        A list of the available sizes

    Returns
    -------
    Tuple[int, int]
        The closest smaller size found.
    """
    h, w = size
    _, _, closest = min(( (abs(h-hi)+abs(w-wi), i, (hi,wi)) for i, (hi, wi) in enumerate(others) if (hi <= h) and (wi <= w) ), default=(None, None, None))
    return closest