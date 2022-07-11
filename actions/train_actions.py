import argparse
from common import config_tools

#####################################
#####################################

def train(args: argparse.Namespace):
    config_tools.register()
    config = config_tools.get_config_from_namespace(args)

    trainer = config.create_trainer()
    trainer.train()

def register_train(parser: argparse.ArgumentParser):
    config_tools.add_config_arguments(parser)
    parser.set_defaults(func=train)

#####################################
#####################################

def resume(args: argparse.Namespace):
    import os, yaml

    config_tools.register()
    config = yaml.unsafe_load(open(os.path.join(args.path, "config.yml"), 'r'))

    trainer = config.create_trainer()
    trainer.resume()

def register_resume(parser: argparse.ArgumentParser):
    parser.add_argument("path", type=str)
    parser.set_defaults(func=resume)