import argparse
from actions.train_actions import register_train, register_resume
from actions.condition_model_actions import register_generate_condition_model_ms
from actions.level_generation_actions import register_generate_levels_ms, register_generate_levels_controllable_ms, register_random_generate_levels_ms
from actions.statistics_actions import register_compute_statistics, register_compute_statistics_ms
from actions.statistics_actions import register_compute_ctrl_statistics, register_compute_ctrl_statistics_ms
from actions.statistics_actions import register_generate_expressive_ranges_ms, register_render_level_sample_ms, register_render_percentile_levels_ms
from actions.statistics_actions import register_profile_generator_time_ms

# This file is the main entry point of the project
# from here you can call all the subcommands

def main():
    parser = argparse.ArgumentParser("Multi-Size Level Generator")
    subparsers = parser.add_subparsers()

    # Here we register all the subcommands which can be called as follows:
    # >>> python cli.py subcommand-name [args]

    # NOTE: If the subcommand name ends with "ms" then it deals with levels of multiple sizes in a single call
    # e.g.
    #   - "compute-statistics" runs on a single file where all the levels are of the same size
    #   - "compute-statistics-ms" runs on multiple files where the levels can have different sizes
    
    # Start a new training session
    register_train(subparsers.add_parser("train"))
    # Resume a training session
    register_resume(subparsers.add_parser("resume"))

    # Generate a condition model
    register_generate_condition_model_ms(subparsers.add_parser("gen-condmodel-ms", aliases=["cmms"]))

    # Generate levels from a model unconditionally
    register_generate_levels_ms(subparsers.add_parser("generate-levels-ms", aliases=["genms"]))
    # Generate levels from a model under the control of a given set of values
    register_generate_levels_controllable_ms(subparsers.add_parser("generate-ctrl-levels-ms", aliases=["cgenms"]))
    # Generate levels randomly
    register_random_generate_levels_ms(subparsers.add_parser("random-generate-levels-ms", aliases=["rngms"]))
    
    # Compute statistics for a certain file of generated levels
    register_compute_statistics(subparsers.add_parser("compute-statistics", aliases=["stats"]))
    # Compute statistics for one or more files of generated levels
    register_compute_statistics_ms(subparsers.add_parser("compute-statistics-ms", aliases=["statsms"]))
    # Compute statistics about the model controllability for a certain file of levels generated from a set of control values
    register_compute_ctrl_statistics(subparsers.add_parser("compute-ctrl-statistics", aliases=["cstats"]))
    # Compute statistics about the model controllability for one or more files of levels generated from a set of control values
    register_compute_ctrl_statistics_ms(subparsers.add_parser("compute-ctrl-statistics-ms", aliases=["cstatsms"]))
    # Create and save the expressive range figures for one or more files of generated levels
    register_generate_expressive_ranges_ms(subparsers.add_parser("draw-expressive-range-ms", aliases=["erms"]))
    # Render and save a sample of generated levels from one or more files
    register_render_level_sample_ms(subparsers.add_parser("draw-level-sample-ms", aliases=["imms"]))
    # Render and save the levels at equally distributed percentiles of specific properties from one or more files
    register_render_percentile_levels_ms(subparsers.add_parser("draw-level-percentiles-ms", aliases=["perimms"]))
    # Profile the time needed to generate level from a model
    register_profile_generator_time_ms(subparsers.add_parser("profile-generation-ms", aliases=["profgenms"]))
    
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()