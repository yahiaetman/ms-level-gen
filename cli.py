import argparse
from actions.train_actions import register_train, register_resume
from actions.condition_model_actions import register_generate_condition_model_ms
from actions.level_generation_actions import register_generate_levels_ms
from actions.statistics_actions import register_statistics_ms

def main():
    parser = argparse.ArgumentParser("Run Training")
    subparsers = parser.add_subparsers()

    register_train(subparsers.add_parser("train"))
    register_resume(subparsers.add_parser("resume"))

    register_generate_condition_model_ms(subparsers.add_parser("gen-mscondmodel", aliases=["mscm"]))
    register_generate_levels_ms(subparsers.add_parser("generate-ms-levels", aliases=["msgen"]))
    register_statistics_ms(subparsers.add_parser("compute-ms-statistics", aliases=["msstats"]))
    
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()