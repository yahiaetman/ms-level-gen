import argparse
from common import config_tools

def generate_condition_model_ms(args: argparse.Namespace):
    import json, os, warnings, pathlib, yaml
    from games import create_game
    
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

def register_generate_condition_model_ms(parser: argparse.ArgumentParser):
    parser.add_argument("path", type=str, help="path to the training results folder")
    config_tools.add_config_arguments(parser)
    parser.set_defaults(func=generate_condition_model_ms)