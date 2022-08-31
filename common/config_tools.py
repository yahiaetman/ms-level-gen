"""
Config Tools
    A minimal and simple set of tools for configuration management

Q:  Why not use one of the popular config management packages?
A:  I wanted some features that I couldn't find in the packages I found.
    Namely, the features I wanted (and implemented) in this file are:
        1- Nested Configuration.
        2- Readable Config Files... so I use YAML.
        3- Support for Dataclasses (The class path can be defined in the config file using yaml tags).
        4- Support for merging configuration. This allows a config file to extend another config file.
        5- Support for including a config file into another one.

Q:  So, how does this work?
A:  These are the tools that you may need to use:
1- 'config' (decorator):        Use this to decorate a dataclass (regular classes are not supported). This
                                adds a yaml 'representer' for the class that give the serialized object
                                a yaml tag that the config loader will use to instantiate the correct class.

2- 'merge' (function):          Use this to merge two or more configurations. More info in the function docs.

3- 'update_object' (function):  Use this to update an object using one of more pairs of field paths and
                                values. For example, an object {'a':{'b':10}} updated using {'a.b':20} will
                                become {'a':{'b':20}}.

4- 'access_object' (function):  Allows you to access an object using a field path. For example, accessing
                                {'a':{'b':10}} using 'a.b' will return 10.

5- 'add_config_arguments' (function): Given an argument parser, this function will add arguments to receive
                                a list of config files (-cfg) and a list of overrides (-ovr). After parsing
                                the arguments, pass the namespace to 'get_config_from_namespace' to get the 
                                user specified config.

6- 'get_config_from_namespace' (function): Given the namespace, which is the result of a parsing an 
                                argument parser that contains the arguments added by 'add_config_arguments'.
                                The config files will be loaded and merged, then the overrides will be applied.

7- 'read_config' (function):    Given one or more file paths, the config files will be loaded, merged
                                then returned. This function adds some custom tags to the loader's constructors.
                                If any file contains multi documents, they will be merged into one.

***********************
** Custom YAML tags ***
***********************

- '!tup <list>':            The list will be converted to a python tuple
- '!range [*range args]:    The args will be passed to a range, then the range will be collected into a list.
- '!inc <path>':            The path string will be replaced by the config at the given path.
- '!obj:<module_path>/<class_path> <mapping>': 
                            This will instantiate the class defined by <module_path>.<class_path> 
                            and fill its fields using the mapping. Missing fields will be filled with 
                            dataclasses.MISSING, except if they have a default value, then they will
                            be filled with the default values.

******************************
*** Common Usage Patterns: ***
******************************

--- Define a Config Class ---
-----------------------------

File "models.py"
>>> @config
>>> @dataclass
>>> class Config:
>>>     learning_rate: float                                            # No default value
>>>     momentum: float = 0.9                                           # Default value = 0.9
>>>     hidden_sizes: List[int] = field(default_factory = lambda: [32]) # Default value = [32]

--- Create a YAML file for the config class ---
-----------------------------------------------

File "example.yaml"
>>> !obj:models/Config
>>> learning_rate: 0.001
>>> hidden_sizes: [128, 64]

--- Read the config file ---
----------------------------

>>> import config_tools as cfg
>>> config = cfg.read_config('example.yaml')
>>> print(config)
>>> # output: models.Config(learning_rate=0.001, momentum=0.9, hidden_sizes=[128, 64])

--- Read and merge multiple config files ---
--------------------------------------------

File "no_momentum.yaml"
>>> !obj:models/Config
>>> momentum: 0.0

>>> import config_tools as cfg
>>> config = cfg.read_config('example.yaml', 'no_momentum.yaml')
>>> print(config)
>>> # output: models.Config(learning_rate=0.001, momentum=0.0, hidden_sizes=[128, 64])

--- Read configuration from commandline arguments ---
-----------------------------------------------------

File "test.py"
>>> import argparse
>>> import config_tools as cfg
>>> parser = argparse.ArgumentParser()
>>> cfg.add_config_arguments(parser)
>>> args = parser.parse_args()
>>> config = cfg.get_config_from_namespace(args)
>>> print(config)

> python test.py -cfg example.yaml
models.Config(learning_rate=0.001, momentum=0.9, hidden_sizes=[128, 64])

> python test.py -cfg example.yaml no_momentum.yaml
models.Config(learning_rate=0.001, momentum=0.0, hidden_sizes=[128, 64])

> python test.py -cfg example.yaml no_momentum.yaml -ovr learning_rate:=0.1 "hidden_sizes:=[10, 20, 30]"
models.Config(learning_rate=0.1, momentum=0.0, hidden_sizes=[10, 20, 30])

--- Use "!inc" ---
------------------

File "hidden_sizes.yaml"
>>> [10, 20, 30]

File "example2.yaml"
>>> !obj:models/Config
>>> learning_rate: 0.001
>>> hidden_sizes: !inc 'hidden_sizes.yaml'

> python test.py -cfg example2.yaml
models.Config(learning_rate=0.001, momentum=0.9, hidden_sizes=[10, 20, 30])

--- Write a config file that extends another file ---
-----------------------------------------------------

File "example3.yaml"
>>> !inc: 'example.yaml'
>>> !obj:models/Config
>>> momentum: 0.0

> python test.py -cfg example.yaml
models.Config(learning_rate=0.001, momentum=0.0, hidden_sizes=[128, 64])


"""

from typing import Any, Dict, List, Union
import yaml
import dataclasses
import pathlib
import argparse

###################################
# Merge two or more configuations #
###################################

__CONFIG_ATTR = "__config_metadata"     # A class attribute storing info in Config classes
__CONFIG_OVERRIDE_ATTR = "__config_ovr" # An instance attribute storing which fields are set by the config loader (not just a default value).

# Internal function that merges two objects and returns the merged object.
def __merge_two(obj1, obj2):
    # Objects of different types are not merged.
    if type(obj1) != type(obj2):
        return obj2
    
    # Dictionaries are merged key-by-key.
    if isinstance(obj1, dict):
        for name, value in obj2.items():
            if name in obj1:
                obj1[name] = __merge_two(obj1[name], obj2[name])
            else:
                obj1[name] = value
        return obj1
    
    # If any of the objects is not a Config class, they are not merged.
    if not hasattr(obj1, __CONFIG_ATTR) or not hasattr(obj2, __CONFIG_ATTR):
        return obj2
    # If the config class is set to not be merged. 
    if not getattr(obj1, __CONFIG_ATTR).get("merge", False):
        return obj2
    
    # Merge the objects
    obj1_ovr = getattr(obj1, __CONFIG_OVERRIDE_ATTR, set())
    obj2_ovr = getattr(obj2, __CONFIG_OVERRIDE_ATTR, set())
    fields = dataclasses.fields(obj1)
    for f in fields:        
        o1 = f.name in obj1_ovr
        o2 = f.name in obj2_ovr
        v1 = getattr(obj1, f.name)
        v2 = getattr(obj2, f.name)
        if o1:
            if o2:
                if f.metadata.get("no_merge", False):
                    setattr(obj1, f.name, v2)
                else:
                    # if both fields were set, merge them.
                    setattr(obj1, f.name, __merge_two(v1, v2))
        else:
            if o2:
                setattr(obj1, f.name, v2)
                obj1_ovr.add(f.name)
    setattr(obj1, __CONFIG_OVERRIDE_ATTR, obj1_ovr)
    return obj1

def merge(*objs):
    """Merges two or more objects and returns the merged object.
    WARNING: This function could modify the first object.

    The objects are merged into the 1st object one by one.
    When two objects are being merged, the following rules are applied:
    1-  If they have different types, the 2nd is returned.
    2-  If they are dictionaries, the 1st is updated using the contents of the 2nd.
    3-  If they are config objects and 'merge' was set to True in the config decorator,
        then the fields are merged one by one. The fields are recursively merged, except
        the fields whose metadata contains 'no_merge' and it is set to True.
        Config objects store which fields had their value set by the loader (no missing
        or default), so that the set values have higher priority over unset ones. Only
        if both values were set that they would be merged recursively. This allow extending
        config for classes containing fields with default values, since it would tell
        us if the value is there by default or it was explicitly set by the user
        in the config file.
    4- Other data types are not merged, and the 2nd will always be returned.
    """
    obj = objs[0]
    for other in objs[1:]:
        obj = __merge_two(obj, other)
    return obj

############################################################################################
# A decorator to automatically register a class representer and add configuration metadata #
############################################################################################

__OBJ_TAG = "!obj:" # A custom tag for config classes
                    # The tag should be used as follows: !obj:<module_path>/<class_path> <mapping>
                    # where the class fields will be filled using the mapping.

def config(cls=None, /, *, merge=True):
    """A decorator for config classes.
    IMPORTANT: the class must be a dataclass.

    Parameters
    ----------
    cls : dataclass, optional
        The class to be decorated. If None, this function will return a decorator. (default: None)
    merge : bool, optional
        if True, the class instances will be merged recursively when sent to the 'merge' function. (default: True)
    """
    def wrap(cls):
        assert dataclasses.is_dataclass(cls), "a configuration class must be a dataclass"
        tag = f'{__OBJ_TAG}{cls.__module__}/{cls.__qualname__}'
        def representer(dumper: yaml.Dumper, obj):
            mapping = {field.name: getattr(obj, field.name) for field in dataclasses.fields(obj)}
            return dumper.represent_mapping(tag, mapping)
        yaml.add_representer(cls, representer)
        setattr(cls, __CONFIG_ATTR, {"merge": merge})
        return cls
    
    if cls is None:
        return wrap
    else:
        return wrap(cls)

################################################################
# Constructors for special tags (e.g. Objects, Includes, etc.) #
################################################################

# Internal function: construct a python tuple.
def __tup_cons(loader: yaml.Loader, node: yaml.Node):
    return loader.construct_python_tuple(node)

# Internal function: construct a list from a range initialized by the args in the yaml node.
def __range_cons(loader: yaml.Loader, node: yaml.Node):
    args = loader.construct_python_tuple(node)
    return list(range(*args))

# Internal function: construct a Config object whose class is defined in the suffix
#                    and the fields are defined in the yaml node.    
def __obj_cons(loader: yaml.Loader, suffix: str, node: yaml.Node):
    import importlib
    module_name, class_name = suffix.split('/')
    m = importlib.import_module(module_name)
    class_nesting = class_name.split('.')
    cls = m
    for name in class_nesting:
        cls = getattr(cls, name)
    fields = dataclasses.fields(cls)
    init, post_init, ovr = {}, {}, set()
    data = loader.construct_mapping(node)
    for f in fields:
        if f.init:
            if f.name in data:
                init[f.name] = data[f.name]
                ovr.add(f.name)
            elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
                init[f.name] = dataclasses.MISSING
        else:
            if f.name in data:
                post_init[f.name] = data[f.name]
                ovr.add(f.name)
    obj = cls(**init)
    for name, val in post_init.items():
        setattr(obj, name, val)
    setattr(obj, __CONFIG_OVERRIDE_ATTR, ovr)
    return obj

# Internal function: construct an object by loading the path defined in node
def __inc_cons(loader: yaml.Loader, node: yaml.Node):
    url: str = loader.construct_python_str(node).strip()
    if url.startswith("~/") or url.startswith("~\\"):
        url = url[2:].strip()
        stream = loader.stream
        stream_url = getattr(stream, "name", None)
        if stream_url is not None:
            url = str(pathlib.Path(stream_url).parent.joinpath(url))
    return read_config(url)

################################################
# Update An Object with Nested-Key Value Pairs #
################################################

# Internal function: Update a field in the object where the key is list of keys 
#                    to deeply access the field that should be modified.
def __update_object(obj, keys: List[str], value):
    if len(keys) == 0: return value
    top = keys[0]
    if obj is None:
        return value
    elif isinstance(obj, list):
        assert top.isdigit(), "a list accessor must be a number"
        index = int(top)
        obj[index] = __update_object(obj[index], keys[1:], value)
        return obj
    elif isinstance(obj, dict):
        obj[top] = __update_object(obj.get(top), keys[1:], value)
        return obj
    elif hasattr(obj, top):
        setattr(obj, top, __update_object(getattr(obj, top), keys[1:], value))
        return obj
    else:
        return obj

def update_object(obj, updates: Dict[str, Any]):
    """Update an object using a dictionary of paths and values.
    A path is a list of keys or attrs to be used to recursively access the field to be updated.
    e.g. a path 'a.b.c' would access an object 'obj' to reach 'obj.a.b.c' 
    or a nested dict of dicts 'd' to reach 'd["a"]["b"]["c"]'. 
    WARNING: The object could be modified.

    Parameters
    ----------
    obj : Any
        The object to update.
    updates : Dict[str, Any]
        A dictionary of path and values where each field identified by the path
        is updated to be the corresponding value.

    Returns
    -------
    Any
        The object after it is updated.
    """
    for key, value in updates.items():
        keys = key.split('.')
        obj = __update_object(obj, keys, value)
    return obj

def access_object(obj, key: Union[str, List[str]]):
    """Return a field from the object which is identified by the path.
    A path is a list of keys or attrs to be used to recursively access the field to be returned.
    e.g. a path 'a.b.c' would access an object 'obj' to reach 'obj.a.b.c' 
    or a nested dict of dicts 'd' to reach 'd["a"]["b"]["c"]'.

    Parameters
    ----------
    obj : Any
        The object to access.
    key : Union[str, List[str]]
        A path to reach the field to access

    Returns
    -------
    Any
        The object reached by following the path.
    """
    if isinstance(key, str): 
        if len(key) == 0: return obj
        key = key.split('.')
    if len(key) == 0: return obj
    top = key[0]
    if obj is None:
        return None
    elif isinstance(obj, list):
        assert top.isdigit(), "a list accessor must be a number"
        index = int(top)
        return access_object(obj[index], key[1:])
    elif isinstance(obj, dict):
        return access_object(obj[top], key[1:])
    elif hasattr(obj, top):
        return access_object(getattr(obj, top), key[1:])
    else:
        return obj

#################################################
# Argument Parsing For Configuration Management #
#################################################

def add_config_arguments(parser: argparse.ArgumentParser, config_file_args = None, override_args = None):
    """Add arguments to read a config from the user via the CLI.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to which the config arguments will be added.
    config_file_args : Optional[List[str]], optional
        The names or flags used to define the config files. If None, it will be ['-cfg', '--config']. (default: None)
    override_args : Optional[List[str]], optional
        The names or flags used to define the overrides. If None, it will be ['-ovr', '--override']. (default: None)
        Each override is defined as 'path=value' or 'path:=value'.
        The path defines how to access the object to reach the value to override.
        If the form 'path:=value' is used, value will be passed to the eval function. It is useful for any object that is not a string.
    """
    config_file_args = config_file_args or ['-cfg', '--config']
    override_args = override_args or ['-ovr', '--override']
    parser.add_argument(*config_file_args, nargs='+', default=[])
    parser.add_argument(*override_args, nargs='*', default=None)

def get_config_from_namespace(args: argparse.Namespace):
    """Get the config object as defined by the user via the CLI.

    Parameters
    ----------
    args : argparse.Namespace
        The namespace returned by the parser.

    Returns
    -------
    Any
        The config object as defined by the user via the CLI.
    """
    config_files = args.config
    config = read_config(*config_files)
    
    overrides = args.override
    if overrides is not None:
        updates = {}
        for override in overrides:
            key, value = override.split('=')
            if key.endswith(':'):
                key = key[:-1]
                value = eval(value)
            updates[key] = value
        config = update_object(config, updates)
    return config

###########################
# Read Configuration File #
###########################

def read_config(*paths):
    """Read one or more config files, then merge them.

    Returns
    -------
    str
        paths to config files.
    """
    config = []
    for path in paths:
        with open(path, 'r') as stream:
            loader = yaml.UnsafeLoader(stream)
        loader.add_constructor("!tup", __tup_cons)
        loader.add_constructor("!range", __range_cons)
        loader.add_constructor("!inc", __inc_cons)
        loader.add_multi_constructor(__OBJ_TAG, __obj_cons)
        try:
            while loader.check_data():
                config.append(loader.get_data())
        finally:
            loader.dispose()
    if config:
        return merge(*config)
    else:
        return {}