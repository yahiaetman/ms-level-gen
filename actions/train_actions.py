import argparse
from common import config_tools

#####################################
#####################################
'''
Train a model from the scratch.

Arguments:
    * -cfg & -ovr: The training configuration.
'''

def train(args: argparse.Namespace):
    config = config_tools.get_config_from_namespace(args)

    trainer = config.create_trainer()
    trainer.train()

def register_train(parser: argparse.ArgumentParser):
    config_tools.add_config_arguments(parser)
    parser.set_defaults(func=train)

#####################################
#####################################
'''
Resume training a model.

Arguments:
    * path to the folder containing the training config file and the checkpoints folder.
'''

def resume(args: argparse.Namespace):
    import os

    config = config_tools.read_config(os.path.join(args.path, "config.yml"))

    trainer = config.create_trainer()
    trainer.resume()

def register_resume(parser: argparse.ArgumentParser):
    parser.add_argument("path", type=str)
    parser.set_defaults(func=resume)