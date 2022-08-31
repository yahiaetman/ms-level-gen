import pathlib
from typing import Optional
from common import config_tools

def find_in_parents(path: str, file_name: str) -> Optional[str]:
    """Look for a file in any of the parent folders.

    Parameters
    ----------
    path : str
        The current path.
    file_name : str
        The file to search for.

    Returns
    -------
    Optional[str]
        A path to the file if found. None if not found.
    """
    parent = pathlib.Path(path)
    while not parent.joinpath(file_name).exists():
        new_parent = parent.parent
        if new_parent == parent:
            return None
        parent = new_parent
    return str(parent.joinpath(file_name))

def access_yaml(path: str):
    """Read a yaml config and return a field from it.

    Parameters
    ----------
    path : str
        A string in the one of 2 forms:
        1- "config_file_path":  reads the config file in the given path and returns it.
        2- "config_file_path@field_path":
                                reads the config file in the given path, get the field
                                identified by the field path, then return it.

    Returns
    -------
    Any
        The object defined by the path.
    """
    if '@' in path:
        file_path, key = path.rsplit('@', 1)
    else:
        file_path, key = path, ""
    file_path = file_path.strip()
    path = path.strip()
    return config_tools.access_object(config_tools.read_config(file_path), key)