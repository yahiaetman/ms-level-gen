import argparse
from common import config_tools

#####################################
#####################################
'''
Generate levels at multiple sizes from a pre-trained model 
where the controls are sampled unconditionally from a condition model.

Arguments:
    * path to the model weights.
    * path to the condition model (without extension).
    * path to which the output will be written.
    * -gen, --generator: path to the generator config. If not specified, a training config will be searched for in the weight's parents.
    * -cfg & -ovr: The generation configuration.

The generation configurations contains:
- The sizes to generate at.
- The number of levels to generate for each size.
- The number of trials to generate a playable level before giving up. (Default: 1)
- A prefix to add before the generated level's files names. (Default: "")

This action checks if the generated levels are playable or not.
If a generated level is unplayable, a replacement is requested in the next trial.

The action writes a separate file for each level sizes named as follows: {prefix}_levels_{width}x{height}.json
The files will not only store the level, they will also store all the information returned by the game during the analysis.
In addition, a statistics file is saved to store the generation time, the number of playable levels generated, the number of requests needed, etc.
'''

def action_generate_levels_ms(args: argparse.Namespace):
    import json, os, pathlib, torch, yaml, time
    from games import create_game
    from . import utils
    from methods.generator import MSGenerator

    weights_path: str = args.weights
    condition_model_path: str = args.cm
    generator_path: str = args.generator
    output_path: str = args.output
    config = config_tools.get_config_from_namespace(args)

    if "%END" in weights_path:
        checkpoint_info_path = utils.find_in_parents(weights_path, "checkpoint.yml")
        assert checkpoint_info_path is not None, "The checkpoint file is needed to know the last training step put it is not found"
        checkpoint_info = yaml.unsafe_load(open(checkpoint_info_path, 'r'))
        checkpoint_step = checkpoint_info["step"]
        weights_path = weights_path.replace("%END", str(checkpoint_step))
    
    if "@" in weights_path:
        weights_path, key = weights_path.rsplit('@', 1)
    else:
        key = ""
    

    if generator_path == "":
        training_config_path = utils.find_in_parents(weights_path, "config.yml")
        assert training_config_path is not None, "The training configuration file is needed to know the generator"
        training_config = config_tools.read_config(training_config_path)
    if generator_path == "":
        generator_config = training_config.generator_config
    else:
        generator_config = utils.access_yaml(generator_path)

    assert os.path.exists(condition_model_path + ".cm"), "The condition model file must exist"
    assert os.path.exists(condition_model_path + ".yml"), "The condition model config file must exist"

    condition_model_config = config_tools.read_config(condition_model_path + ".yml")
    
    game = create_game(condition_model_config["game_config"])
    conditions = condition_model_config["conditions"]

    generation_sizes = config.get("generation_sizes", [])
    generation_amount = config.get("generation_amount", 10000)
    trials = config.get("trials", 1)
    prefix = config.get("prefix", "")
    if prefix: prefix += "_"

    cond_model = condition_model_config["condition_model_config"].model_constructor(game, conditions)
    cond_model.load(condition_model_path + ".cm")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator: MSGenerator = generator_config.model_constructor(len(game.tiles), len(conditions)).to(device)
    weights = config_tools.access_object(torch.load(weights_path), key)
    generator.load_state_dict(weights["netG"])

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    generation_stats = {}
    for size_index, size in enumerate(generation_sizes):
        h, w = size
        print(f"{size_index+1}/{len(generation_sizes)}: Working on Size {h}x{w} ...")
        results = []
        found = 0
        requests = 0
        start_time = time.time()
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
        elapsed_time = time.time() - start_time
        print(f"Done in {elapsed_time} seconds ({trial} trials) using {requests} requests")
        json.dump(results, open(os.path.join(output_path, f"{prefix}levels_{h}x{w}.json"), 'w'))
        generation_stats[size] = {
            "trials": trials,
            "requests": requests,
            "solvable": found,
            "elapsed_seconds": elapsed_time
        }
    yaml.dump(generation_stats, open(os.path.join(output_path, f"{prefix}generation_stats.yml"), 'w'))

def register_generate_levels_ms(parser: argparse.ArgumentParser):
    parser.add_argument("weights", type=str, help="path to the model weight")
    parser.add_argument("cm", type=str, help="path to the condition model file")
    parser.add_argument("output", type=str, help="path to which the levels will be saved")
    parser.add_argument("-gen", "--generator", type=str, default="", help="path to the condition model file")
    config_tools.add_config_arguments(parser)
    parser.set_defaults(func=action_generate_levels_ms)

#####################################
#####################################
'''
Generate levels at multiple sizes from a pre-trained model 
where the controls are sampled conditionally from a condition model given user-supplied controls.

Arguments:
    * path to the model weights.
    * path to the condition model (without extension).
    * path to which the output will be written.
    * -gen, --generator: path to the generator config. If not specified, a training config will be searched for in the weight's parents.
    * -cfg & -ovr: The generation configuration.

The generation configurations contains for each requested control:
- The control name and denomerator (as a function so for flexibility).
- The sizes to generate at and the control range for each size.
- The number of levels to generate for each size.
- The number of trials to generate a playable level before giving up. (Default: 1)
- A prefix to add before the generated level's files names. (Default: "")

This action checks if the generated levels are playable or not.
The best level across trials (nearest to the requested controls) is returned.

The action writes a separate file for each level sizes named as follows: {prefix}_ctrl_levels_{width}x{height}.json
The files will not only store the level, they will also store all the information returned by the game during the analysis.
In addition, a statistics file is saved to store the generation time, the number of playable levels generated, the number of requests needed, etc.
'''

def action_generate_levels_controllable_ms(args: argparse.Namespace):
    import json, os, pathlib, torch, yaml, time, tqdm
    from games import create_game
    from . import utils
    from methods.generator import MSGenerator
    from methods.ms_conditions import ControllableConditionModel

    weights_path: str = args.weights
    condition_model_path: str = args.cm
    generator_path: str = args.generator
    output_path: str = args.output
    config = config_tools.get_config_from_namespace(args)

    if "%END" in weights_path:
        checkpoint_info_path = utils.find_in_parents(weights_path, "checkpoint.yml")
        assert checkpoint_info_path is not None, "The checkpoint file is needed to know the last training step put it is not found"
        checkpoint_info = yaml.unsafe_load(open(checkpoint_info_path, 'r'))
        checkpoint_step = checkpoint_info["step"]
        weights_path = weights_path.replace("%END", str(checkpoint_step))
    
    if "@" in weights_path:
        weights_path, key = weights_path.rsplit('@', 1)
    else:
        key = ""
    
    if generator_path == "":
        training_config_path = utils.find_in_parents(weights_path, "config.yml")
        assert training_config_path is not None, "The training configuration file is needed to know the generator"
        training_config = config_tools.read_config(training_config_path)
    if generator_path == "":
        generator_config = training_config.generator_config
    else:
        generator_config = utils.access_yaml(generator_path)

    assert os.path.exists(condition_model_path + ".cm"), "The condition model file must exist"
    assert os.path.exists(condition_model_path + ".yml"), "The condition model config file must exist"

    condition_model_config = config_tools.read_config(condition_model_path + ".yml")
    
    game = create_game(condition_model_config["game_config"])
    conditions = condition_model_config["conditions"]

    generation_sizes = config.get("generation_sizes", [])
    controlled_conditions = config.get("controlled_conditions", [])
    trials = config.get("trials", 1)
    prefix = config.get("prefix", "")
    if prefix: prefix += "_"

    cond_model: ControllableConditionModel = condition_model_config["condition_model_config"].model_constructor(game, conditions)
    cond_model.load(condition_model_path + ".cm")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator: MSGenerator = generator_config.model_constructor(len(game.tiles), len(conditions)).to(device)
    weights = config_tools.access_object(torch.load(weights_path), key)
    generator.load_state_dict(weights["netG"])

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    generation_stats = {}
    for size_index, size in enumerate(generation_sizes):
        h, w = size
        generation_stats[size] = {}
        print(f"{size_index+1}/{len(generation_sizes)}: Working on Size {h}x{w} ...")
        
        all_results = []
        for condition_index, controlled_condition in enumerate(controlled_conditions):
            control_name = controlled_condition["control_name"]
            condition_name = controlled_condition["condition"]
            mapping = controlled_condition.get("mapping", "value")
            mapping = eval(f"lambda value, size: {mapping}")
            amount_per_pin = controlled_condition["amount_per_pin"]
            control_values = controlled_condition["values"][size]
            print(f"Condition {condition_index+1}/{len(controlled_conditions)}: {control_name} ...")
        
            pbar = tqdm.tqdm(control_values, desc="Starting", dynamic_ncols=True)
            start_time = time.time()
            for control_value in pbar:
                condition_value: float = mapping(control_value, size)
                
                results = None
                for trial in range(1, trials+1):
                    pbar.set_description(f"Requested: {control_value} - Trial {trial}/{trials}")
                    conditions = cond_model.sample_given(size, amount_per_pin, {condition_name:condition_value}).to(device)
                    levels = generator.generate(conditions, size)
                    info = game.analyze(levels.tolist())
                    if results is None:
                        results = info
                    else:
                        for index, item in enumerate(info):
                            if not item["solvable"]: continue
                            if not results[index]["solvable"] or abs(results[index][condition_name] - condition_value) > abs(item[condition_name] - condition_value):
                                results[index] = item
                for item in results:
                    item["control-name"] = control_name
                    item["control-value"] = control_value
                all_results.extend(results)
            elapsed_time = time.time() - start_time
            generation_stats[size][control_name] = {
                "trials": trials,
                "elapsed_seconds": elapsed_time
            }
        json.dump(all_results, open(os.path.join(output_path, f"{prefix}ctrl_levels_{h}x{w}.json"), 'w'))
    yaml.dump(generation_stats, open(os.path.join(output_path, f"{prefix}ctrl_generation_stats.yml"), 'w'))

def register_generate_levels_controllable_ms(parser: argparse.ArgumentParser):
    parser.add_argument("weights", type=str, help="path to the model weight")
    parser.add_argument("cm", type=str, help="path to the condition model file")
    parser.add_argument("output", type=str, help="path to which the levels will be saved")
    parser.add_argument("-gen", "--generator", type=str, default="", help="path to the condition model file")
    config_tools.add_config_arguments(parser)
    parser.set_defaults(func=action_generate_levels_controllable_ms)

#####################################
#####################################
'''
Generate random levels at multiple sizes.

Arguments:
    * path to the game configuration.
    * path to which the output will be written.
    * -m, --mode: the random generator's mode (this depends on what is implemented for each game). (Default: "basic")
    * -cfg & -ovr: The generation configuration.

The generation configurations contains:
- The sizes to generate at.
- The number of levels to generate for each size.
- The number of trials to generate a playable level before giving up. (Default: 1)
- A prefix to add before the generated level's files names. (Default: "")

This action checks if the generated levels are playable or not.
If a generated level is unplayable, a replacement is requested in the next trial.

The action writes a separate file for each level sizes named as follows: {prefix}_levels_{height}x{width}.json
The files will not only store the level, they will also store all the information returned by the game during the analysis.
In addition, a statistics file is saved to store the generation time, the number of playable levels generated, the number of requests needed, etc.
'''

def action_random_generate_levels_ms(args: argparse.Namespace):
    import json, os, pathlib, yaml, time
    from games import create_game

    game_config_path: str = args.game
    output_path: str = args.output
    rng_mode = args.mode
    config = config_tools.get_config_from_namespace(args)
    
    game = create_game(config_tools.read_config(game_config_path))
    
    generation_sizes = config.get("generation_sizes", [])
    generation_amount = config.get("generation_amount", 10000)
    trials = config.get("trials", 1)
    prefix = config.get("prefix", "")
    if prefix: prefix += "_"

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    generation_stats = {}
    for size_index, size in enumerate(generation_sizes):
        h, w = size
        print(f"{size_index+1}/{len(generation_sizes)}: Working on Size {h}x{w} ...")
        results = []
        found = 0
        requests = 0
        start_time = time.time()
        for trial in range(1, trials+1):
            remaining = generation_amount - len(results)
            requests += remaining
            levels = game.generate_random(remaining, size, mode=rng_mode)
            info = game.analyze(levels)
            solvable = [item for item in info if item["solvable"]]
            found += len(solvable)
            if trial == trials:
                results.extend(info)
            else:
                results.extend(solvable)
            print(f"Trial {trial}/{trials}: Progress = {found}/{generation_amount} levels")
            if found == generation_amount:
                break
        elapsed_time = time.time() - start_time
        print(f"Done in {elapsed_time} seconds ({trial} trials) using {requests} requests")
        json.dump(results, open(os.path.join(output_path, f"{prefix}rnd_levels_{h}x{w}.json"), 'w'))
        generation_stats[size] = {
            "trials": trials,
            "requests": requests,
            "solvable": found,
            "elapsed_seconds": elapsed_time
        }
    yaml.dump(generation_stats, open(os.path.join(output_path, f"{prefix}rnd_generation_stats.yml"), 'w'))

def register_random_generate_levels_ms(parser: argparse.ArgumentParser):
    parser.add_argument("game", type=str, help="path to the game configuration")
    parser.add_argument("output", type=str, help="path to which the levels will be saved")
    parser.add_argument("--mode", "-m", type=str, default="basic", help="the generator mode")
    config_tools.add_config_arguments(parser)
    parser.set_defaults(func=action_random_generate_levels_ms)