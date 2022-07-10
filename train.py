import argparse
import os

import yaml
from common import config_tools

def train(args: argparse.Namespace):
    config_tools.register()
    config = config_tools.get_config_from_namespace(args)

    trainer = config.create_trainer()
    trainer.train()

def resume(args: argparse.Namespace):
    config_tools.register()
    config = yaml.unsafe_load(open(os.path.join(args.path, "config.yml"), 'r'))

    trainer = config.create_trainer()
    trainer.resume()

def main():
    parser = argparse.ArgumentParser("Run Training")
    subparsers = parser.add_subparsers()

    # Train Options
    train_parser = subparsers.add_parser("train")
    config_tools.add_config_arguments(train_parser)
    train_parser.set_defaults(func=train)

    # Resume Options
    resume_parser = subparsers.add_parser("resume")
    resume_parser.add_argument("path", type=str)
    resume_parser.set_defaults(func=resume)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()