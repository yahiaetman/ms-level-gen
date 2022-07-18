import argparse
from typing import Dict, List
from common import config_tools
import statistics

def summary(prefix: str, data):
    return {
        f"{prefix}-mean": statistics.mean(data),
        f"{prefix}-median": statistics.median(data),
        f"{prefix}-stdev": statistics.stdev(data),
        f"{prefix}-min": min(data),
        f"{prefix}-max": max(data),
    }

def compute_statistics(info: List[Dict], config: Dict):
    import games
    import numpy as np

    total = len(info)
    compilable = [item for item in info if item["compilable"]]
    solvable = [item for item in compilable if item["solvable"]]
    solvable_count = len(solvable)
    levels = [item["level"] for item in solvable]
    unique = set(tuple(tuple(row) for row in level) for level in levels)        

    levels_np = np.array(levels)
    _, h, w = levels_np.shape
    diversity = sum( np.sum((levels_np[i] != levels_np[i+1:]).astype(int)) for i in range(len(levels)-1) ) / ((w * h) * (solvable_count * (solvable_count - 1)) * 0.5)
    diversity = float(diversity)
    entropies = [games.utils.entropy(level) for level in levels]

    stats = {
        "Compilable%": len(compilable) / total * 100,
        "Solvable%": solvable_count / total * 100,
        "Duplicates%": (solvable_count - len(unique)) / max(solvable_count, 1) * 100,
        "Diversity": diversity,
        **summary("Entropy", entropies)
    }

    for prop_name in config.get("counter", []):
        prop_values = {tuple(item[prop_name]) for item in solvable}
        stats.update({
            f"unique-{prop_name}-count": len(prop_values),
            f"unique-{prop_name}-to-generated%": len(prop_values) / total * 100,
            f"unique-{prop_name}-to-solvable%": len(prop_values) / max(1, solvable_count) * 100,
        })

    for prop_name in config.get("summary", []):
        prop_values = [item[prop_name] for item in solvable]
        stats.update(summary(prop_name, prop_values))

    return stats

def action_compute_statistics(args: argparse.Namespace):
    import json, pathlib, yaml

    config_tools.register()
    levels_path: str = args.levels
    output_path: str = args.output
    config = config_tools.get_config_from_namespace(args)

    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    info = json.load(open(levels_path, 'r'))

    if len(info) == 0:
        print("The file is empty, No statistics will be generated.")

    yaml.dump(compute_statistics(info, config), open(output_path, 'w'), sort_keys=False)

def register_compute_statistics(parser: argparse.ArgumentParser):
    parser.add_argument("levels", type=str, help="path to the generated levels")
    parser.add_argument("output", type=str, help="path to the save the statistics")
    config_tools.add_config_arguments(parser)
    parser.set_defaults(func=action_compute_statistics)

def action_compute_statistics_ms(args: argparse.Namespace):
    import json, pathlib, glob, yaml
    from collections import defaultdict

    config_tools.register()
    levels_glob_path: str = args.levels
    output_path: str = args.output
    config = config_tools.get_config_from_namespace(args)

    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    levels_files = glob.glob(levels_glob_path)
    level_groups = defaultdict(list)
    for levels_file in levels_files:
        info = json.load(open(levels_file, 'r'))
        for item in info:
            level = item["level"]
            h, w = len(level), len(level[0])
            level_groups[(h, w)].append(item)

    if len(level_groups) == 0:
        print("The files are empty, No statistics will be generated.")
    
    stats = {}
    for size_index, (size, info) in enumerate(level_groups.items()):
        h, w = size
        print(f"{size_index+1}/{len(level_groups)}: Working on Size {h}x{w} ...")
        stats[size] = compute_statistics(info, config)

    yaml.dump(stats, open(output_path, 'w'), sort_keys=False)

def register_compute_statistics_ms(parser: argparse.ArgumentParser):
    parser.add_argument("levels", type=str, help="path (pattern) to the generated levels")
    parser.add_argument("output", type=str, help="path to the save the statistics")
    config_tools.add_config_arguments(parser)
    parser.set_defaults(func=action_compute_statistics_ms)

########################################
########################################

def compute_ctrl_statistics(info: List[Dict], tolerences: Dict[str, float]):
    from collections import defaultdict
    level_groups = defaultdict(list)
    for item in info:
        level_groups[item["control-name"]].append(item)

    stats = {}
    for control_name, info in level_groups.items():
        tolerence = tolerences.get(control_name, 1)
        total = len(info)
        compilable = [item for item in info if item["compilable"]]
        solvable = [item for item in compilable if item["solvable"]]
        solvable_count = len(solvable)
        control_values = [item["control-value"] for item in solvable]
        actual_values = [item[control_name] for item in solvable]
        errors = [abs(a - b) for a, b in zip(control_values, actual_values)]
        correct = sum(error==0 for error in errors)
        score = sum([1-min(tolerence, error)/tolerence for error in errors])

        stats[control_name] = {   
            "Compilable%": len(compilable) / total * 100,
            "Solvable%": solvable_count / total * 100,
            **summary("error", errors),
            "accuracy": correct / total * 100,
            "score": score / total * 100
        }

    return stats

def action_compute_ctrl_statistics(args: argparse.Namespace):
    import json, pathlib, yaml
    from methods.utils import find_closest_size

    config_tools.register()
    levels_path: str = args.levels
    output_path: str = args.output
    config = config_tools.get_config_from_namespace(args)

    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    info = json.load(open(levels_path, 'r'))

    if len(info) == 0:
        print("The file is empty, No statistics will be generated.")

    arb_level = info[0]["level"]
    h, w = len(arb_level), len[arb_level[0]]
    tolerences = {}
    for control_name, control_config in config["controlled_conditions"].items():
        control_tolerences = control_config.get("tolerence", {})
        closest_size = find_closest_size((h, w), control_tolerences.keys())
        if closest_size is None: continue
        tolerences[control_name] = control_tolerences[closest_size]

    yaml.dump(compute_ctrl_statistics(info, tolerences), open(output_path, 'w'), sort_keys=False)

def register_compute_ctrl_statistics(parser: argparse.ArgumentParser):
    parser.add_argument("levels", type=str, help="path to the generated levels")
    parser.add_argument("output", type=str, help="path to the save the statistics")
    config_tools.add_config_arguments(parser)
    parser.set_defaults(func=action_compute_ctrl_statistics)

def action_compute_ctrl_statistics_ms(args: argparse.Namespace):
    import json, pathlib, glob, yaml
    from collections import defaultdict
    from methods.utils import find_closest_size

    config_tools.register()
    levels_glob_path: str = args.levels
    output_path: str = args.output
    config = config_tools.get_config_from_namespace(args)

    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    levels_files = glob.glob(levels_glob_path)
    level_groups = defaultdict(list)
    for levels_file in levels_files:
        info = json.load(open(levels_file, 'r'))
        for item in info:
            level = item["level"]
            h, w = len(level), len(level[0])
            level_groups[(h, w)].append(item)

    if len(level_groups) == 0:
        print("The files are empty, No statistics will be generated.")
    
    stats = {}
    for size_index, (size, info) in enumerate(level_groups.items()):
        h, w = size
        print(f"{size_index+1}/{len(level_groups)}: Working on Size {h}x{w} ...")
        tolerences = {}
        for control_name, control_config in config["controlled_conditions"].items():
            control_tolerences = control_config.get("tolerence", {})
            closest_size = find_closest_size(size, control_tolerences.keys())
            if closest_size is None: continue
            tolerences[control_name] = control_tolerences[closest_size]
        stats[size] = compute_ctrl_statistics(info, tolerences)

    yaml.dump(stats, open(output_path, 'w'), sort_keys=False)

def register_compute_ctrl_statistics_ms(parser: argparse.ArgumentParser):
    parser.add_argument("levels", type=str, help="path (pattern) to the generated levels")
    parser.add_argument("output", type=str, help="path to the save the statistics")
    config_tools.add_config_arguments(parser)
    parser.set_defaults(func=action_compute_ctrl_statistics_ms)

########################################
########################################

def action_generate_expressive_ranges_ms(args: argparse.Namespace):
    import os, pathlib, json, glob
    from collections import defaultdict
    from common.heatmap import Heatmaps

    config_tools.register()
    levels_glob_path: str = args.levels
    output_path: str = args.output
    config = config_tools.get_config_from_namespace(args)

    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    levels_files = glob.glob(levels_glob_path)
    level_groups = defaultdict(list)
    for levels_file in levels_files:
        info = json.load(open(levels_file, 'r'))
        for item in info:
            level = item["level"]
            h, w = len(level), len(level[0])
            level_groups[(h, w)].append(item)

    if len(level_groups) == 0:
        print("The files are empty, No statistics will be generated.")
    
    heatmaps = Heatmaps(config)
    for size_index, (size, info) in enumerate(level_groups.items()):
        h, w = size
        print(f"{size_index+1}/{len(level_groups)}: Working on Size {h}x{w} ...")
        heatmaps.update(size, info)
        fig = heatmaps.render(size)
        fig.savefig(f"{output_path}_{h}x{w}.pdf")

def register_generate_expressive_ranges_ms(parser: argparse.ArgumentParser):
    parser.add_argument("levels", type=str, help="path (pattern) to the level files")
    parser.add_argument("output", type=str, help="the prefix of the used for expressive ranges")
    config_tools.add_config_arguments(parser)
    parser.set_defaults(func=action_generate_expressive_ranges_ms)

##########################################
##########################################

def action_render_level_sample_ms(args: argparse.Namespace):
    import json, pathlib, yaml, glob, random
    from collections import defaultdict
    import numpy as np
    from . import utils
    from games import create_game, img_utils
    
    config_tools.register()
    levels_glob_path: str = args.levels
    outputs_path: str = args.output
    game_path: str = args.game
    sample_count = args.count

    if "%END" in levels_glob_path:
        checkpoint_info_path = utils.find_in_parents(levels_glob_path, "checkpoint.yml")
        assert checkpoint_info_path is not None, "The checkpoint file is needed to know the last training step put it is not found"
        checkpoint_info = yaml.unsafe_load(open(checkpoint_info_path, 'r'))
        checkpoint_step = checkpoint_info["step"]
        levels_glob_path = levels_glob_path.replace("%END", str(checkpoint_step))

    if game_path == "":
        training_config_path = utils.find_in_parents(levels_glob_path, "config.yml")
        assert training_config_path is not None, "The training configuation file is need to know the conditions and/or the game"
        training_config = config_tools.read_config_file(training_config_path)
    if game_path == "":
        game_config = training_config.game_config
    else:
        game_config = utils.access_yaml(game_path)

    game = create_game(game_config)

    levels_files = glob.glob(levels_glob_path)
    level_groups = defaultdict(list)
    for levels_file in levels_files:
        info = json.load(open(levels_file, 'r'))
        for item in info:
            level = item["level"]
            h, w = len(level), len(level[0])
            level_groups[(h, w)].append(item)
    
    pathlib.Path(outputs_path).parent.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    for size_index, (size, info) in enumerate(level_groups.items()):
        h, w = size
        print(f"{size_index+1}/{len(level_groups)}: Working on Size {h}x{w} ...")
        solvable = [item["level"] for item in info if item["solvable"]]
        unsolvable = [item["level"] for item in info if not item["solvable"]]
        
        solvable_sample_count = min(len(solvable), sample_count)
        unsolvable_sample_count = min(len(unsolvable), sample_count)
        
        if solvable_sample_count != 0:
            sample = random.sample(solvable, solvable_sample_count)
            images = game.render(np.array(sample))
            img_utils.save_images(f"{outputs_path}_solvable_{h}x{w}.png", images, (1, solvable_sample_count))
        
        if unsolvable_sample_count != 0:
            sample = random.sample(unsolvable, unsolvable_sample_count)
            images = game.render(np.array(sample))
            img_utils.save_images(f"{outputs_path}_unsolvable_{h}x{w}.png", images, (1, unsolvable_sample_count))

def register_render_level_sample_ms(parser: argparse.ArgumentParser):
    parser.add_argument("levels", type=str, help="path to the level data")
    parser.add_argument("output", type=str, help="path to images")
    parser.add_argument("count", type=int, help="the number of images to generate")
    parser.add_argument("-g", "--game", type=str, default="", help="the game configuration")
    parser.add_argument("-s", "--seed", type=int, default=420, help="the random seed for sampling")
    parser.set_defaults(func=action_render_level_sample_ms)

##########################################
##########################################

def action_render_percentile_levels_ms(args: argparse.Namespace):
    import json, pathlib, yaml, glob, warnings
    from collections import defaultdict
    import numpy as np
    from . import utils
    from games import create_game, img_utils
    
    config_tools.register()
    levels_glob_path: str = args.levels
    outputs_path: str = args.output
    game_path: str = args.game
    sample_count: int = args.count
    properties = args.properties

    assert sample_count >= 1, "sample count cannot be less than 0"
    percentiles = [0.5]
    if sample_count >= 2:
        percentiles = [i/(sample_count-1) for i in range(sample_count)]

    if "%END" in levels_glob_path:
        checkpoint_info_path = utils.find_in_parents(levels_glob_path, "checkpoint.yml")
        assert checkpoint_info_path is not None, "The checkpoint file is needed to know the last training step put it is not found"
        checkpoint_info = yaml.unsafe_load(open(checkpoint_info_path, 'r'))
        checkpoint_step = checkpoint_info["step"]
        levels_glob_path = levels_glob_path.replace("%END", str(checkpoint_step))

    if game_path == "":
        training_config_path = utils.find_in_parents(levels_glob_path, "config.yml")
        assert training_config_path is not None, "The training configuation file is need to know the conditions and/or the game"
        training_config = config_tools.read_config_file(training_config_path)
    if game_path == "":
        game_config = training_config.game_config
    else:
        game_config = utils.access_yaml(game_path)

    game = create_game(game_config)

    levels_files = glob.glob(levels_glob_path)
    level_groups = defaultdict(list)
    for levels_file in levels_files:
        info = json.load(open(levels_file, 'r'))
        for item in info:
            if not item["solvable"]: continue
            level = item["level"]
            h, w = len(level), len(level[0])
            level_groups[(h, w)].append(item)
    
    pathlib.Path(outputs_path).parent.mkdir(parents=True, exist_ok=True)

    for size_index, (size, info) in enumerate(level_groups.items()):
        h, w = size
        print(f"{size_index+1}/{len(level_groups)}: Working on Size {h}x{w} ...")
        
        if len(info) < len(percentiles):
            warnings.warn(f"The number of solvable levels with size {h}x{w} are not enough to create a percentile map of size {len(percentiles)}")
            continue
        
        samples = []
        for prop_name in properties:
            info.sort(key=(lambda item: item[prop_name]))
            count = len(info)
            indices = [round(p*(count-1)) for p in percentiles]
            samples.append([info[index]["level"] for index in indices])
        
        images = game.render(np.array(samples))
        img_utils.save_images(f"{outputs_path}_{h}x{w}.png", images, (len(properties), len(percentiles)))

def register_render_percentile_levels_ms(parser: argparse.ArgumentParser):
    parser.add_argument("levels", type=str, help="path to the level data")
    parser.add_argument("output", type=str, help="path to images")
    parser.add_argument("count", type=int, help="the number of images to generate")
    parser.add_argument("properties", nargs="+", type=str, help="the properties on which the percentiles are selected")
    parser.add_argument("-g", "--game", type=str, default="", help="the game configuration")
    parser.set_defaults(func=action_render_percentile_levels_ms)