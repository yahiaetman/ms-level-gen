from typing import Any, Dict, List, Union
import yaml
import dataclasses
import argparse

###################################
# Merge two or more configuations #
###################################

__CONFIG_ATTR = "__config_metadata"

def __merge_two(obj1, obj2):
    if type(obj1) != type(obj2):
        return obj2
    if isinstance(obj1, dict):
        for name, value in obj2.items():
            if name in obj1:
                obj1[name] = __merge_two(obj1[name], obj2[name])
            else:
                obj1[name] = value
        return obj1
    if not hasattr(obj1, __CONFIG_ATTR) or not hasattr(obj2, __CONFIG_ATTR) or type(obj1) != type(obj2):
        return obj2
    config_options = getattr(obj1, __CONFIG_ATTR)
    should_merge = config_options.get("merge", False)
    if not should_merge:
        return obj2
    fields = dataclasses.fields(obj1)
    for f in fields:        
        v1 = getattr(obj1, f.name)
        v2 = getattr(obj2, f.name)
        if v1 == dataclasses.MISSING:
            setattr(obj1, f.name, v2)
        elif v2 != dataclasses.MISSING:
            if f.metadata.get("no_merge", False):
                setattr(obj1, f.name, v2)
            else:
                setattr(obj1, f.name, __merge_two(v1, v2))
    return obj1

def merge(*objs):
    obj = objs[0]
    for other in objs[1:]:
        obj = __merge_two(obj, other)
    return obj

############################################################################################
# A decorator to automatically register a class representer and add configuration metadata #
############################################################################################

__OBJ_TAG = "!obj:"

def config(cls=None, /, *, merge=True):
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

def __tup_cons(loader: yaml.Loader, node: yaml.Node):
    return loader.construct_python_tuple(node)

def __range_cons(loader: yaml.Loader, node: yaml.Node):
    args = loader.construct_python_tuple(node)
    return list(range(*args))

def __obj_cons(loader: yaml.Loader, suffix: str, node: yaml.Node):
    import importlib
    module_name, class_name = suffix.split('/')
    m = importlib.import_module(module_name)
    class_nesting = class_name.split('.')
    cls = m
    for name in class_nesting:
        cls = getattr(cls, name)
    fields = dataclasses.fields(cls)
    init, post_init = {}, {}
    data = loader.construct_mapping(node)
    for f in fields:
        if f.init:
            if f.name in data:
                init[f.name] = data[f.name]
            elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
                init[f.name] = dataclasses.MISSING
        else:
            if f.name in data:
                post_init[f.name] = data[f.name]
    obj = cls(**init)
    for name, val in post_init.items():
        setattr(obj, name, val)
    return obj

def __inc_cons(loader: yaml.Loader, node: yaml.Node):
    url = loader.construct_python_str(node)
    return yaml.unsafe_load(open(url, 'r'))

def register():
    yaml.add_constructor("!tup", __tup_cons)
    yaml.add_constructor("!range", __range_cons)
    yaml.add_constructor("!inc", __inc_cons)
    yaml.add_multi_constructor(__OBJ_TAG, __obj_cons)

################################################
# Update An Object with Nested-Key Value Pairs #
################################################

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
    for key, value in updates.items():
        keys = key.split('.')
        obj = __update_object(obj, keys, value)
    return obj

def access_object(obj, key: Union[str, List[str]]):
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
    config_file_args = config_file_args or ['-cfg', '--config']
    override_args = override_args or ['-ovr', '--override']
    parser.add_argument(*config_file_args, nargs='+', default=[])
    parser.add_argument(*override_args, nargs='*', action='store', default=None)

def get_config_from_namespace(args: argparse.Namespace):
    
    config_files = args.config
    configs = [c for config_file in config_files for c in yaml.unsafe_load_all(open(config_file, 'r'))]
    if configs:
        config = merge(*configs)
    else:
        config = {}
    
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

def read_config_file(path: str):
    config = yaml.unsafe_load_all(open(path, 'r'))
    return merge(*config)