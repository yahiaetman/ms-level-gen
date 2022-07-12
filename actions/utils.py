import pathlib, yaml
from typing import Optional
from common import config_tools

def find_in_parents(path: str, file_name: str) -> Optional[str]:
    parent = pathlib.Path(path)
    while not parent.joinpath(file_name).exists():
        new_parent = parent.parent
        if new_parent == parent:
            return None
        parent = new_parent
    return str(parent.joinpath(file_name))

def access_yaml(path: str):
    if '@' in path:
        file_path, key = path.rsplit('@', 1)
    else:
        file_path, key = path, ""
    return config_tools.access_object(config_tools.read_config_file(file_path), key)