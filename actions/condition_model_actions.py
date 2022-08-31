import argparse
from common import config_tools

#####################################
#####################################
'''
Generate a condition model (Typically, a GMM) from the given level data.

Arguments:
    * A glob pattern for the level files to use for fitting the condition model.
    * The path to which the condition model should be save (no extension since 
        2 files with the extensions will be created: '.yaml' for the model config & '.pt' for the model data)
    * -g, -game: a path to the game config. If not specified, a training config will be searched for in the levels' parents.
    * -c, ---conditions: a path to the condition config. If not specified, a training config will be searched for in the levels' parents.
    * -cfg & -ovr: The condition model configuration.
'''

def action_generate_condition_model_ms(args: argparse.Namespace):
    import json, pathlib, yaml, glob
    from collections import defaultdict
    from . import utils
    from games import create_game
    
    levels_glob_path: str = args.levels
    outputs_path: str = args.output
    game_path: str = args.game
    conditions_path: str = args.conditions
    config = config_tools.get_config_from_namespace(args)

    if "%END" in levels_glob_path:
        checkpoint_info_path = utils.find_in_parents(levels_glob_path, "checkpoint.yml")
        assert checkpoint_info_path is not None, "The checkpoint file is needed to know the last training step put it is not found"
        checkpoint_info = yaml.unsafe_load(open(checkpoint_info_path, 'r'))
        checkpoint_step = checkpoint_info["step"]
        levels_glob_path = levels_glob_path.replace("%END", str(checkpoint_step))

    if conditions_path == "" or game_path == "":
        training_config_path = utils.find_in_parents(levels_glob_path, "config.yml")
        assert training_config_path is not None, "The training configuration file is needed to know the conditions and/or the game"
        training_config = config_tools.read_config(training_config_path)
    if conditions_path == "":
        conditions = training_config.conditions
    else:
        conditions = utils.access_yaml(conditions_path)
    if game_path == "":
        game_config = training_config.game_config
    else:
        game_config = utils.access_yaml(game_path)

    game = create_game(game_config)
    cond_model = config.model_constructor(game, conditions)
    
    levels_files = glob.glob(levels_glob_path)
    level_groups = defaultdict(list)
    for levels_file in levels_files:
        info = json.load(open(levels_file, 'r'))
        for item in info:
            if not item["solvable"]: continue
            level = item["level"]
            h, w = len(level), len(level[0])
            level_groups[(h, w)].append(item)
    
    for size_index, (size, info) in enumerate(level_groups.items()):
        h, w = size
        print(f"{size_index+1}/{len(level_groups)}: Working on Size {h}x{w} ...")
        cond_model.update(size, info)
    
    print("Saving...")
    pathlib.Path(outputs_path).parent.mkdir(parents=True, exist_ok=True)
    cond_model.save(outputs_path + ".cm")
    yaml.dump({
        "game_config": game_config,
        "conditions": conditions,
        "condition_model_config": config
    }, open(outputs_path + ".yml", 'w'))
    print("Done")

def register_generate_condition_model_ms(parser: argparse.ArgumentParser):
    parser.add_argument("levels", type=str, help="path (pattern allowed) to the level data")
    parser.add_argument("output", type=str, help="path to the save the condition")
    parser.add_argument("-g", "--game", type=str, default="", help="the game configuration")
    parser.add_argument("-c", "--conditions", type=str, default="", help="the conditions")
    config_tools.add_config_arguments(parser)
    parser.set_defaults(func=action_generate_condition_model_ms)