import argparse
from typing import Dict, List
from common import config_tools
import statistics

def summary(prefix: str, data: List[float]) -> Dict[str, float]:
    """Given a list of numbers, return some statistics about it

    Parameters
    ----------
    prefix : str
        a string to add as a prefix to the statistics.
    data : List[float]
        the data for which the statistics should be computed.

    Returns
    -------
    Dict[str, float]
        a dictionary of the statistics in the form {'{prefix}-{statistic_name}' : statistic_value}.
    """
    return {
        f"{prefix}-mean": statistics.mean(data),
        f"{prefix}-median": statistics.median(data),
        f"{prefix}-stdev": statistics.stdev(data),
        f"{prefix}-min": min(data),
        f"{prefix}-max": max(data),
    }

def compute_statistics(info: List[Dict], config: Dict):
    """Given a list of level data and a statistics config, compute the requested statistics for these levels

    Parameters
    ----------
    info : List[Dict]
        The list of level data (generated by a game analyze function).
    config : Dict
        The statistics config containing what items to count and what properties to produces statistics for.

    Returns
    -------
    Dict[str, float]
        The requested statistics. It contains:
        - "Compilable%": the percentage of levels that the game deems compilable.
        - "Solvable%": the percentage of levels that the game deems solvable (playable).
        - "Duplicates%": the percentage of levels removed while extracting the distinct playable levels.
        - "Diversity": the average tile-wise hamming distance between all pairs of playable levels divided by the level area.
        - "Entropy-{statistic}": a statistical summary of the 2x2 entropy over all the playable levels.
        - for each item in config['counter']:
            - "unique-{item}-count": the number of distinct items in all the playable levels.
            - "unique-{item}-to-generated%": 100% * "unique-{item}-count" / the level count.
            - "unique-{item}-to-solvable%": 100% * "unique-{item}-count" / the solvable level count.
        - for each property in config['summary']:
            - "{property}-{statistic}": a statistical summary of {property} over all the playable levels.
    """
    import games
    import numpy as np

    total = len(info)
    compilable = [item for item in info if item.get("compilable", True)]
    solvable = [item for item in compilable if item["solvable"]]
    solvable_count = len(solvable)
    levels = [item["level"] for item in solvable]
    unique = set(tuple(tuple(row) for row in level) for level in levels)        

    levels_np = np.array(levels)
    _, h, w = levels_np.shape
    diversity = sum( np.sum((levels_np[i] != levels_np[i+1:]).astype(np.int64)) for i in range(len(levels)-1) ) / ((w * h) * (solvable_count * (solvable_count - 1)) * 0.5)
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

#####################################
#####################################
'''
Analyze levels stored in a file

Arguments:
    * The game config for which the levels are designed.
    * The path to the levels (json or txt).
    * The path where the analysis results will be saved.

The output will be a Json file containing the analyzed levels returned by "game.analyze".
'''

def action_analyze_levels(args: argparse.Namespace):
    import json, pathlib, yaml
    from games import GameConfig

    game_config_path: str = args.game
    levels_path: str = args.levels
    output_path: str = args.output

    game_config: GameConfig = config_tools.read_config(game_config_path)
    game = game_config.create()

    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    if levels_path.endswith(".json"):
        info = json.load(open(levels_path, 'r'))
        levels = [item["level"] for item in info]
        names = [item.get("name", str(index)) for index, item in enumerate(info)]
    else:
        levels, names = game.load_dataset(levels_path) 

    if len(levels) == 0:
        print("The file is empty, No analysis will be generated.")
    
    info = game.analyze(levels)
    for name, item in zip(names, info):
        item["name"] = name

    json.dump(info, open(output_path, 'w'))

def register_analyze_levels(parser: argparse.ArgumentParser):
    parser.add_argument("game", type=str, help="path to the game configuration")
    parser.add_argument("levels", type=str, help="path to the generated levels")
    parser.add_argument("output", type=str, help="path to the save the level info")
    parser.set_defaults(func=action_analyze_levels)

#####################################
#####################################
'''
Compute statistics for level data with a single size.

Arguments:
    * A glob pattern for the level files.
    * The path where the statistics will be saved.
    * -cfg & -ovr: The statistics configuration.

The output will be a YAML file containing the statistics returned by the function "compute_statistics".
'''

def action_compute_statistics(args: argparse.Namespace):
    import json, pathlib, yaml

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

#####################################
#####################################
'''
Compute statistics for level data with different sizes.

Arguments:
    * A glob pattern for the level files.
    * The path where the statistics will be saved.
    * -cfg & -ovr: The statistics configuration.

The output will be a YAML file with a key for each size where the value is
the statistics returned by the function "compute_statistics" for that size.
'''

def action_compute_statistics_ms(args: argparse.Namespace):
    import json, pathlib, glob, yaml
    from collections import defaultdict

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
    """Given a list of level data and a statistics config, compute the requested control statistics for these levels

    Parameters
    ----------
    info : List[Dict]
        The list of level data (generated by a game analyze function).
    tolerences : Dict[str, float]
        The control score tolerance for each control.

    Returns
    -------
    Dict[str, float]
        The requested statistics. It contains:
        - "Compilable%": the percentage of levels that the game deems compilable.
        - "Solvable%": the percentage of levels that the game deems solvable (playable).
        - "error-{statistic}": a statistical summary of the absolute error between the requested and generated properties over all the playable levels.
        - "accuracy": the percentage of playable levels where the requested and the generated properties match.
        - "score": the control score (see https://ieeexplore.ieee.org/abstract/document/9779063 for more info).
        - "r2": the coefficient of determination of the requested and generated properties over all the playable levels.
    """
    from collections import defaultdict
    from sklearn.metrics import r2_score

    level_groups = defaultdict(list)
    for item in info:
        level_groups[item["control-name"]].append(item)

    stats = {}
    for control_name, info in level_groups.items():
        tolerence = tolerences.get(control_name, 1)
        total = len(info)
        compilable = [item for item in info if item.get("compilable", True)]
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
            "score": score / total * 100,
            "r2": float(r2_score(control_values, actual_values)),
        }

    return stats

#####################################
#####################################
'''
Compute control statistics for level data with a single size.

Arguments:
    * A glob pattern for the level files.
    * The path where the statistics will be saved.
    * -cfg & -ovr: The statistics configuration.

The output will be a YAML file containing the statistics returned by the function "compute_ctrl_statistics".
'''

def action_compute_ctrl_statistics(args: argparse.Namespace):
    import json, pathlib, yaml
    from methods.utils import find_closest_size

    levels_path: str = args.levels
    output_path: str = args.output
    config = config_tools.get_config_from_namespace(args)

    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    info = json.load(open(levels_path, 'r'))

    if len(info) == 0:
        print("The file is empty, No statistics will be generated.")

    arb_level = info[0]["level"]
    h, w = len(arb_level), len(arb_level[0])
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

#####################################
#####################################
'''
Compute control statistics for level data with different sizes.

Arguments:
    * A glob pattern for the level files.
    * The path where the statistics will be saved.
    * -cfg & -ovr: The statistics configuration.

The output will be a YAML file with a key for each size where the value is
the statistics returned by the function "compute_ctrl_statistics" for that size.
'''

def action_compute_ctrl_statistics_ms(args: argparse.Namespace):
    import json, pathlib, glob, yaml
    from collections import defaultdict
    from methods.utils import find_closest_size

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

#####################################
#####################################
'''
Render the expressive range for level data with different sizes.

Arguments:
    * A glob pattern for the level files.
    * The path where the statistics will be saved (without extension since the suffix "_{height}x{width}.pdf" will be added).
    * -cfg & -ovr: The heatmap configuration.

The output will be a PDF for each level size.
'''

def action_generate_expressive_ranges_ms(args: argparse.Namespace):
    import pathlib, json, glob
    from collections import defaultdict
    from common.heatmap import Heatmaps

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

#####################################
#####################################
'''
Render a sample of the levels at different sizes.

Arguments:
    * A glob pattern for the level files.
    * The path where the statistics will be saved (without extension since a suffix will be added).
    * The number of levels to render in a sample.
    * -g, --game: a path to the game config. If not specified, a training config will be searched for in the levels' parents.
    * -s, --seed: a random seed. (Default: 420)

The output will be two pngs for each level size.
One png with a sample of playable levels suffixed "_solvable_{height}x{width}.png"
The other with a sample of unplayable levels suffixed "_unsolvable_{height}x{width}.png"
'''

def action_render_level_sample_ms(args: argparse.Namespace):
    import json, pathlib, yaml, glob, random
    from collections import defaultdict
    import numpy as np
    from . import utils
    from games import GameConfig, img_utils
    
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
        assert training_config_path is not None, "The training configuation file is needed to know the conditions and/or the game"
        training_config = config_tools.read_config(training_config_path)
    if game_path == "":
        game_config: GameConfig = training_config.game_config
    else:
        game_config: GameConfig = utils.access_yaml(game_path)

    game = game_config.create()

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
    parser.add_argument("count", type=int, help="the number of levels to render")
    parser.add_argument("-g", "--game", type=str, default="", help="the game configuration")
    parser.add_argument("-s", "--seed", type=int, default=420, help="the random seed for sampling")
    parser.set_defaults(func=action_render_level_sample_ms)

#####################################
#####################################
'''
Render a sample of the levels at different sizes, where the levels are picked to be a certain percentiles of given properties.

Arguments:
    * A glob pattern for the level files.
    * The path where the statistics will be saved (without extension since a suffix will be added).
    * The number of levels to render in a sample (this specifies the number of percentiles).
    * A list of properties at which the percentiles will be picked.
    * -g, --game: a path to the game config. If not specified, a training config will be searched 
                for in the levels' parents.
    * -f, --figure: Pick whether a matplotlib figure where the property names and values will be
                rendered alongside the level should be generated. (Default: False).
    * -s, --seed: a random seed for shuffling before sorting. (Default: 69)

If the number of levels is:
1: The median will be rendered.
2: The minimum and the maximum will be rendered.
3: The minimum, the median and the maximum will be rendered.
n: The levels at [1, 2, .., n]/n*100% percentiles will be rendered. 

The output will be a PNG (if figure=false) or a PDF (if figure=true) for each level size.
The generated file will be suffixed with "_{height}x{width}.png" or "_{height}x{width}.pdf".
'''

def action_render_percentile_levels_ms(args: argparse.Namespace):
    import json, pathlib, yaml, glob, warnings, random
    from collections import defaultdict
    import numpy as np
    from . import utils
    from games import GameConfig, img_utils
    
    levels_glob_path: str = args.levels
    outputs_path: str = args.output
    game_path: str = args.game
    sample_count: int = args.count
    properties = args.properties
    as_figure = args.figure
    random.seed(args.seed)

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
        training_config = config_tools.read_config(training_config_path)
    if game_path == "":
        game_config: GameConfig = training_config.game_config
    else:
        game_config: GameConfig = utils.access_yaml(game_path)

    game = game_config.create()

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
        values = []
        for prop_name in properties:
            random.shuffle(info)
            info.sort(key=(lambda item: item[prop_name]))
            count = len(info)
            indices = [round(p*(count-1)) for p in percentiles]
            samples.append([info[index]["level"] for index in indices])
            values.append([info[index][prop_name] for index in indices])
        
        images = game.render(np.array(samples))

        if as_figure:
            import matplotlib.pyplot as plt
            plt.rcParams["font.family"] = "Times New Roman"

            images = images.transpose([0, 1, 3, 4, 2])

            fig = plt.figure(tight_layout=True)

            width_ratios = [0.25] + [w/h]*sample_count

            subfigs = fig.subfigures(nrows=len(properties), ncols=1)
            for row, subfig in enumerate(subfigs):
                prop_name: str = properties[row]
                prop_name = '\n'.join(s.capitalize() for s in prop_name.split('-'))
                subfig.suptitle(prop_name, x=0.05, y=0.5, rotation=90, va='center')

                # create 1x3 subplots per subfig
                axs = subfig.subplots(nrows=1, ncols=sample_count+1, gridspec_kw={'width_ratios': width_ratios})
                axs[0].axis('off')
                for col, ax in enumerate(axs[1:]):
                    ax.imshow(images[row,col])
                    ax.axis('off')
                    if col == 0:
                        prefix = "min"
                    elif col == sample_count - 1:
                        prefix = "max"
                    elif col == (sample_count-1)/2:
                        prefix = "median"
                    else:
                        prefix = f"{percentiles[col]*100}%" 
                    ax.set_title(f'{prefix}: {values[row][col]}')
            
            fig.savefig(f"{outputs_path}_{h}x{w}.pdf", bbox_inches='tight')
        else:
            img_utils.save_images(f"{outputs_path}_{h}x{w}.png", images, (len(properties), len(percentiles)))

def register_render_percentile_levels_ms(parser: argparse.ArgumentParser):
    parser.add_argument("levels", type=str, help="path to the level data")
    parser.add_argument("output", type=str, help="path to images")
    parser.add_argument("count", type=int, help="the number of images to generate")
    parser.add_argument("properties", nargs="+", type=str, help="the properties on which the percentiles are selected")
    parser.add_argument("-g", "--game", type=str, default="", help="the game configuration")
    parser.add_argument("-f", "--figure", action="store_true", default=False, help="Pick if the output should be printed as a figure")
    parser.add_argument("-s", "--seed", type=int, default=69, help="the random seed for shuffling before sorting")
    parser.set_defaults(func=action_render_percentile_levels_ms)

#####################################
#####################################
'''
Profile the generator speed.

Arguments:
    * The path to the generator config.
    * The tileset size (type: int).
    * The condition (control vector) size (type: int).
    * -b, --batch: the size of the test batch. (Default: 1).
    * -r, --repeat: The number of profiling repetitions. Increase to improve the precision. (Default: 1000).
    * -cfg & -ovr: The generation configuration used for profiling.

This action does not save anyting. The results are printed on the console.
'''

def action_profile_generator_time_ms(args: argparse.Namespace):
    import torch, time
    from . import utils
    from methods.generator import MSGenerator
    
    generator_path: str = args.generator
    tile_count: str = args.tileset
    condition_size: str = args.condition
    batch_size: int = args.batch
    repeat: int = args.repeat
    config = config_tools.get_config_from_namespace(args)

    generator_config = utils.access_yaml(generator_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator: MSGenerator = generator_config.model_constructor(tile_count, condition_size).to(device)
    
    generation_sizes = config.get("generation_sizes", [])
    
    conditions = torch.rand((batch_size, condition_size), device=device)

    A = torch.ones((len(generation_sizes), 2))
    b = torch.zeros(len(generation_sizes))

    for size_index, size in enumerate(generation_sizes):
        h, w = size
        print(f"{size_index+1}/{len(generation_sizes)}: Working on Size {h}x{w} ...")

        # run once first to ensure initialization is not computed in the time
        generator.generate(conditions, size)

        start = time.time()

        for _ in range(repeat):
            generator.generate(conditions, size)
        
        elapsed = (time.time() - start) / repeat

        A[size_index, 1] = h * w
        b[size_index] = elapsed

        print(f"Generation time / Level = {elapsed} secs")
    
    x, *_ = torch.linalg.lstsq(A, b)

    print(f"Time Estimate = {x[0]} + wh * {x[1]} secs")
    print(f"Correlation Coefficient = { torch.corrcoef(torch.stack([A[:,1], b]))[0, 1] }")

def register_profile_generator_time_ms(parser: argparse.ArgumentParser):
    parser.add_argument("generator", type=str, help="path to the generator config")
    parser.add_argument("tileset", type=int, help="the tileset size")
    parser.add_argument("condition", type=int, help="the condition size")
    parser.add_argument("--batch", "-b", type=int, default=1, help="the batch size")
    parser.add_argument("--repeat", "-r", type=int, default=1000, help="the number of repetitions")
    config_tools.add_config_arguments(parser)
    parser.set_defaults(func=action_profile_generator_time_ms)