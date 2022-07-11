import argparse
from common import config_tools

def compute_statistics_ms(args: argparse.Namespace):
    import json, os, warnings, pathlib, statistics, yaml
    import games
    import numpy as np

    config_tools.register()
    path = args.path
    config = config_tools.get_config_from_namespace(args)

    training_config_path = os.path.join(path, "config.yml")
    if not os.path.exists(training_config_path):
        warnings.warn("The training config is missing.")
        return
    training_config = yaml.unsafe_load(open(training_config_path, 'r'))

    sizes = config.get("sizes", training_config.sizes)
    generation_sizes = sizes + config.get("extra_sizes", [])
    prefix = config.get("prefix", "")
    if prefix: prefix += "_"

    output_path = os.path.join(path, "output")
    statistics_path = os.path.join(path, "statistics")
    pathlib.Path(statistics_path).mkdir(parents=True, exist_ok=True)
    
    def summary(prefix: str, data):
        return {
            f"{prefix}-mean": statistics.mean(data),
            f"{prefix}-median": statistics.median(data),
            f"{prefix}-stdev": statistics.stdev(data),
            f"{prefix}-min": min(data),
            f"{prefix}-max": max(data),
        }

    for size_index, size in enumerate(generation_sizes):
        h, w = size
        print(f"{size_index+1}/{len(generation_sizes)}: Working on Size {h}x{w} ...")
        info = json.load(open(os.path.join(output_path, f"{prefix}levels_{h}x{w}.json"), 'r'))
        
        total = len(info)
        compilable = [item for item in info if item["compilable"]]
        solvable = [item for item in compilable if item["solvable"]]
        solvable_count = len(solvable)
        levels = [item["level"] for item in solvable]
        unique = set(tuple(tuple(row) for row in level) for level in levels)        
        unique_solutions = set(item["solution"] for item in solvable)
        unique_push_signatures = set(tuple(item["push-signature"]) for item in solvable)
        unique_crate_signatures = set(tuple(item["crate-signature"]) for item in solvable)

        levels_np = np.array(levels)
        #diversity = np.mean((levels_np[None, :, :, :] != levels_np[:, None, :, :]).astype(float))
        diversity = sum( np.sum((levels_np[i] != levels_np[i+1:]).astype(int)) for i in range(len(levels)-1) ) / ((w * h) * (solvable_count * (solvable_count - 1)) * 0.5)
        entropies = [games.utils.entropy(level) for level in levels]

        walls = [item["walls"] for item in solvable]
        crates = [item["crates"] for item in solvable]
        pushed_crates = [item["pushed-crates"] for item in solvable]
        solution_lengths = [item["solution-length"] for item in solvable]

        stats = {
            "Compilable%": len(compilable) / total * 100,
            "Solvable%": solvable_count / total * 100,
            "Duplicates%": (solvable_count - len(unique)) / max(solvable_count, 1) * 100,
            "Unique-Solutions": len(unique_solutions),
            "Unique-Push-Signatures": len(unique_push_signatures),
            "Unique-Crate-Signatures": len(unique_crate_signatures),
            "Diversity": diversity,
            **summary("Entropy", entropies),
            **summary("Walls", walls),
            **summary("Crates", crates),
            **summary("Pushed-crates", pushed_crates),
            **summary("Solution-lengths", solution_lengths),
        }

        json.dump(stats, open(os.path.join(statistics_path, f"{prefix}statistics_{h}x{w}.json"), 'w'), indent=1)

def register_compute_statistics_ms(parser: argparse.ArgumentParser):
    parser.add_argument("path", type=str, help="path to the training results folder")
    config_tools.add_config_arguments(parser)
    parser.set_defaults(func=compute_statistics_ms)

########################################
########################################

def generate_expressive_ranges_ms(args: argparse.Namespace):
    import os, warnings, pathlib, yaml, json
    from common.heatmap import Heatmaps

    config_tools.register()
    path = args.path
    prefix = args.prefix
    if prefix: prefix += "_"
    config = config_tools.get_config_from_namespace(args)

    output_path = os.path.join(path, "output")
    statistics_path = os.path.join(path, "statistics")
    pathlib.Path(statistics_path).mkdir(parents=True, exist_ok=True)

    generation_stats_path = os.path.join(output_path, f"{prefix}generation_stats.yml")
    if not os.path.exists(generation_stats_path):
        warnings.warn("No generation statistics were found")
        return
    generation_stats = yaml.unsafe_load(open(generation_stats_path, 'r'))
    generation_sizes = list(generation_stats.keys())

    heatmaps = Heatmaps(config)
    for size_index, size in enumerate(generation_sizes):
        h, w = size
        print(f"{size_index+1}/{len(generation_sizes)}: Working on Size {h}x{w} ...")
        info = json.load(open(os.path.join(output_path, f"{prefix}levels_{h}x{w}.json"), 'r'))
        heatmaps.update(size, info)
        fig = heatmaps.render(size)
        fig.savefig(os.path.join(output_path, f"{prefix}expressive_range_{h}x{w}.pdf"))

def register_generate_expressive_ranges_ms(parser: argparse.ArgumentParser):
    parser.add_argument("path", type=str, help="path to the training results folder")
    parser.add_argument("-p", "--prefix", type=str, default="", help="the prefix of the generated levels files")
    config_tools.add_config_arguments(parser)
    parser.set_defaults(func=generate_expressive_ranges_ms)