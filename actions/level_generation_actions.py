import argparse
from common import config_tools

def generate_levels_ms(args: argparse.Namespace):
    import json, os, warnings, pathlib, torch, yaml
    from games import create_game
    from methods.generator import MSGenerator

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

def register_generate_levels_ms(parser: argparse.ArgumentParser):
    parser.add_argument("path", type=str, help="path to the training results folder")
    parser.add_argument("cm", type=str, help="name of the condition model file")
    config_tools.add_config_arguments(parser)
    parser.set_defaults(func=generate_levels_ms)