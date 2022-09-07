# Start Small: Training Level Generator from Nothing by Learning at Multiple Sizes

<p align="center">
  <img src="docs/cover.png" />
</p>

## About

This repository contains the code for the paper "Start Small: Training Level Generator from Nothing by Learning at Multiple Sizes". It proposes as method to train level generators from nothing by starting at a small size. The method makes the following assumption:
- The probability of finding playable levels via random exploration is sufficiently high at small level sizes.
- The generator can gain knowledge from learning at small sizes to start generating levels at lerger sizes.

For the first assumption, we are yet to face a game that does not satisfy it. For Sokoban, the probability that a random level is playable is ~0.6% which is sufficiently high for our method. The probability was even higher in the other games we tried. For the second assumption, the results shows that it is true since the generator learns to generate diverse levels for sizes it had not seen during training. Overall, the method presents the following advantages:
- Train the generator without reward shaping or training examples, so it should be easier to apply on new games.
- Learn to generate diverse playable levels at a variety of sizes within a relatively short amount of training time.
- Generate levels at sizes that were not seen during training (but the performance is usually better if the model was trained on the targeted size).
- Control the generator's output by supplying the desired properties as inputs to the model.

## Code Structure

The code is divided as follows:
- `cli.py`: whichs is the main entry point that can be used to invoke the different actions defined in the folder `actions`.
- `actions`: where the different actions that `cli.py` can invoke. The actions are organized into different files by their purpose (training, level generation, analysis, etc).
- `common`: which contains some common utilities such as the configuration tools and the heatmap generator.
- `games`: which contains the implementations of the different games. The games are listed in a dictionary inside `games/__init__.py`.
- `methods`: which contains the implementations of the different methods (includng the method defined in the paper). Each method is defined in a separate subfolder.
- `datasets`: contains some dataset for some of the games. It is used by one of the methods, but not by the method defined in the paper.
- `configs`: which contains the configurations for all the experiment and analysis for the different games.
- `analyze.ps1`: which is a powershell script that runs all the analysis actions on an experiment.

## Method Structure

```mermaid
flowchart  TD;
T[Trainer]-->O[Optimizer];
O-->G[Generator];
T-->D[Dataset];
T-->C[Condition Model];
```

The main components are:
- `Generator`: which is the module that learns to generate the levels.
- `Optimizer`: which is responsible for optimizing the generator. It's interface depends on the method, so each optimizer can only be used by a compatible trainer and for a compatible generator.
- `Dataset`: where the levels are stored. It also differs depending on the method so it can only be used by a compatible trainer. For some methods (such as the GFlowNet), the dataset is initially empty and is filled over time. In that case, it acts as an experience replay buffer.
- `Condition Model`: which is used to sample conditions. The trainer should update it after it updates the dataset. Currently, the implemented condition models are trainer-agnostic and can be used with all the trainer available in the repository. Therefore, it is defined in the root folder of the methods. 
- `Trainer`: which is the class that orchestrates the whole training process.

## Adding a new Game

Each game is defined using two classes:
- A `Game` class which defines the game tiles, sprite images and the game's functional requirements. The functional requirements should be defined in the function `analyze` which takes a list of levels and returns an information dictionary for each level that should at least contain the following:
  - `"level"`: which is the level itself (without any changes).
  - `"solvable"`: which is a boolean that indicates if the level can played and won or not.
- A `ConditionUtility` class which is used to retrieve some tools and values to be used with the controls (conditions). Look at the class documentation in `games/game.py` to know more about it.

## Installation

This repository's requirements are defined in [environment.yml](environment.yml).

you can either create a new environment from it using:

    conda env create -f environment.yml
    conda activate msgen

or update an existing envionment using:

    conda env update --file environment.yml

**Note:** The dependencies installs a package from a github repo ([sokosolve](https://github.com/yahiaetman/sokosolve)) which is a sokoban solver required to run the sokoban experiments.

## Usage

To start a training session, you can use:

    python cli.py train -cfg ./configs/sokoban/gflownet/div-tuple.yml

This will start a sokoban training session and will automatically create a folder `runs/%TIME_%NAME` where `%TIME` is replaced by a timestamp containing the training start time and `%NAME` is replaced by a automatically created name that represents the experiment configuration. This folder will hold all the experiment results including a version of the training configuration and the checkpoints. If you wish, you can override the save path as follows:

    python cli.py train -cfg ./configs/sokoban/gflownet/div-tuple.yml -ovr save_path="./experiment"

This will save the experiment results in the folder `./experiment`. You can also override other configuration options similarly. For example, we can turn off data augmentation using:

    python cli.py train -cfg ./configs/sokoban/gflownet/div-tuple.yml -ovr save_path="./experiment" dataset_config.data_augmentation:=False

The `:=` sign means that the value after it is not necessarily a string and that it will be passed to the python builtin function `eval`. If the training was interrupted, you can resume it using:

    python cli.py resume ./experiment
 
This command will automatically read the correct configuration from the given path. After you finish the training process, it is still not ready to be used generation. You have to build a condition model for it. In the paper, the condition model is fit to the data in the final version of the dataset checkpoint. To do that, you can use:

    python cli.py cmms ./experiment/checkpoints/dataset_%END_*x*.json ./experiment/condition_models/GMM -cfg configs/sokoban/ms_conditions/gmm.yml

Notice that we use a glob pattern to give the script the final dataset for all the sizes. The script automatically replaces `%END` with the number of steps at the last checkpoint. After that you can use the generator and the condition model to generate a batch of levels as follows:

    python cli.py genms ./experiment/checkpoints/model_%END.pt ./experiment/condition_models/GMM ./experiment/output -cfg configs/sokoban/analysis/generate.yml

The level count and sizes are defined in the config file [configs/sokoban/analysis/generate.yml](configs/sokoban/analysis/generate.yml). You can override any of these configurations using `-ovr` or by writing another config file. After generating the files, you may want to collect some statistics about the generated levels. For that you can use:

    python cli.py statsms ./experiment/output/levels_*x*.json ./experiment/statistics/statistics.yml -cfg configs/sokoban/analysis/statistics.yml

There are still more scripts that are available, so it would become tedious to repeat them after every experiment. So after the training is done, you can do all the analysis steps (including the condition model and level generation) using the [analyze.ps1](analyze.ps1) powershell script which will save the analysis results in the given experiment folder. To run this script, use the following:

    ./analyze.ps1 sokoban ./experiment

Note that this script requires the game name to pick some analysis options. If you want to run all the experiments in the paper (including the analysis steps), you can just run [paper_experiments.ps1](paper_experiments.ps1) but note that it runs 30 experiments so it would probably require multiple days to finish depending on your machine.

To explore the results of the generators by hand, you can run them and try different control values and level sizes in the jupyter notebook [explore.ipynb](explore.ipynb).

## License

The project is available as open source under the terms of the [MIT License](LICENSE).