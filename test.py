import argparse
import json
import os
import warnings
import pathlib
import torch
import statistics

import yaml
from common import config_tools
from games import create_game
import games
from methods.generator import MSGenerator


def generate_ms_condition_model(args: argparse.Namespace):
    config_tools.register()
    path = args.path
    config = config_tools.get_config_from_namespace(args)

    training_config_path = os.path.join(path, "config.yml")
    if not os.path.exists(training_config_path):
        warnings.warn("The training config is missing.")
        return
    training_config = yaml.unsafe_load(open(training_config_path, 'r'))

    checkpoint_path = os.path.join(path, "checkpoints")
    checkpoint_info_path = os.path.join(checkpoint_path, "checkpoint.yml")
    if not os.path.exists(checkpoint_info_path):
        warnings.warn("There are no checkpoints in the given path")
        return
    checkpoint_info = yaml.unsafe_load(open(checkpoint_info_path, 'r'))
    checkpoint_step = checkpoint_info["step"]

    game = create_game(training_config.game_config)
    conditions = training_config.conditions
    sizes = training_config.sizes
    cond_model_config = config["condition_model_config"]

    cond_model = cond_model_config.model_constructor(game, conditions, sizes)

    for size_index, size in enumerate(sizes):
        h, w = size
        print(f"{size_index+1}/{len(sizes)}: Working on Size {h}x{w} ...")
        dataset_path = os.path.join(checkpoint_path, f"dataset_{checkpoint_step}_{h}x{w}.json")
        if not os.path.exists(dataset_path):
            warnings.warn(f"Couldn't find the dataset file {dataset_path}, skipping this size.")
            continue
        info = json.load(open(dataset_path, 'r'))
        cond_model.update(size, info)
    
    print("Saving...")
    condition_model_path = os.path.join(path, "condition_models")
    pathlib.Path(condition_model_path).mkdir(parents=True, exist_ok=True)
    file_name = config.get("file_name", "%NAME.cm")
    file_name = file_name.replace("%NAME", cond_model.name)
    config["file_name"] = file_name
    file_path = os.path.join(condition_model_path, file_name)
    cond_model.save(file_path)
    yaml.dump(config, open(os.path.splitext(file_path)[0] + "_config.yml", 'w'))
    print("Done")

def generate_ms_levels(args: argparse.Namespace):
    config_tools.register()
    path = args.path
    cm_file_name = os.path.splitext(args.cm)[0]
    config = config_tools.get_config_from_namespace(args)

    training_config_path = os.path.join(path, "config.yml")
    if not os.path.exists(training_config_path):
        warnings.warn("The training config is missing.")
        return
    training_config = yaml.unsafe_load(open(training_config_path, 'r'))

    checkpoint_path = os.path.join(path, "checkpoints")
    checkpoint_info_path = os.path.join(checkpoint_path, "checkpoint.yml")
    if not os.path.exists(checkpoint_info_path):
        warnings.warn("There are no checkpoints in the given path")
        return
    checkpoint_info = yaml.unsafe_load(open(checkpoint_info_path, 'r'))
    checkpoint_step = checkpoint_info["step"]

    condition_model_path = os.path.join(path, "condition_models")
    condition_model_config_path = os.path.join(condition_model_path, f"{cm_file_name}_config.yml")
    if not os.path.exists(condition_model_config_path):
        warnings.warn(f"The requested condition model {condition_model_config_path} does not exist")
        return
    condition_model_config = yaml.unsafe_load(open(condition_model_config_path, 'r'))
    condition_model_file_name = condition_model_config["file_name"]
    condition_model_file_path = os.path.join(condition_model_path, condition_model_file_name)
    if not os.path.exists(condition_model_file_path):
        warnings.warn(f"The requested condition model {condition_model_file_path} does not exist")
        return
    
    model_checkpoint_path = os.path.join(checkpoint_path, f"model_{checkpoint_step}.pt")
    if not os.path.exists(model_checkpoint_path):
        warnings.warn("The requested model weigth does not exist")
        return

    game = create_game(training_config.game_config)
    conditions = training_config.conditions
    sizes = config.get("sizes", training_config.sizes)
    generation_sizes = sizes + config.get("extra_sizes", [])
    generation_amount = config.get("generation_amount", 10000)
    trials = config.get("trials", 1)
    prefix = config.get("prefix", "")
    if prefix: prefix += "_"

    cond_model = condition_model_config["condition_model_config"].model_constructor(game, conditions, sizes)
    cond_model.load(condition_model_file_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator: MSGenerator = training_config.generator_config.model_constructor(len(game.tiles), len(conditions)).to(device)
    weights = torch.load(model_checkpoint_path)
    generator.load_state_dict(weights["netG"])

    output_path = os.path.join(path, "output")
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    for size_index, size in enumerate(generation_sizes):
        h, w = size
        print(f"{size_index+1}/{len(generation_sizes)}: Working on Size {h}x{w} ...")
        results = []
        found = 0
        requests = 0
        for trial in range(1, trials+1):
            remaining = generation_amount - len(results)
            requests += remaining
            conditions = cond_model.sample(size, remaining).to(device)
            levels = generator.generate(conditions, size)
            info = game.analyze(levels.tolist())
            solvable = [item for item in info if item["solvable"]]
            found += len(solvable)
            if trial == trials:
                results.extend(info)
            else:
                results.extend(solvable)
            print(f"Trial {trial}/{trials}: Progress = {found}/{generation_amount} levels")
            if found == generation_amount:
                break
        print(f"Done in {trial} trials using {requests} requests")
        json.dump(results, open(os.path.join(output_path, f"{prefix}levels_{h}x{w}.json"), 'w'))
        
def compute_ms_statistics(args: argparse.Namespace):
    import numpy as np

    config_tools.register()
    path = args.path
    config = config_tools.get_config_from_namespace(args)

    training_config_path = os.path.join(path, "config.yml")
    if not os.path.exists(training_config_path):
        warnings.warn("The training config is missing.")
        return
    training_config = yaml.unsafe_load(open(training_config_path, 'r'))

    game = create_game(training_config.game_config)
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

def main():
    parser = argparse.ArgumentParser("Run Training")
    subparsers = parser.add_subparsers()

    # Generate Conditon Model Options
    cm_parser = subparsers.add_parser("gen-mscondmodel", aliases=["mscm"])
    cm_parser.add_argument("path", type=str, help="path to the training results folder")
    config_tools.add_config_arguments(cm_parser)
    cm_parser.set_defaults(func=generate_ms_condition_model)

    # Generate Levels Options
    gen_parser = subparsers.add_parser("generate-ms-levels", aliases=["msgen"])
    gen_parser.add_argument("path", type=str, help="path to the training results folder")
    gen_parser.add_argument("cm", type=str, help="name of the condition model file")
    config_tools.add_config_arguments(gen_parser)
    gen_parser.set_defaults(func=generate_ms_levels)

    # Generate Levels Options
    stats_parser = subparsers.add_parser("compute-ms-statistics", aliases=["msstats"])
    stats_parser.add_argument("path", type=str, help="path to the training results folder")
    config_tools.add_config_arguments(stats_parser)
    stats_parser.set_defaults(func=compute_ms_statistics)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()