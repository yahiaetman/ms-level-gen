$ROOT = "runs/2022-07-10_03-13-43_SOKOBAN_DIVS_DIVR_PREW_AUG_DATASET_DIV_KDECOND_GFLOW_SNAKE_MSCELSTM_GEN"
$GAME = "sokoban"

# generate GMMs (assumes the game and condition config in weights_path/../config.yml)
python cli.py cmms "$ROOT/checkpoints/dataset_%END_*x*.json" "$ROOT/condition_models/GMM" -cfg "configs/$GAME/ms_conditions/gmm.yml"
# generate levels (assumes the generator config in weights_path/../config.yml)
python cli.py genms "$ROOT/checkpoints/model_%END.pt" "$ROOT/condition_models/GMM" "$root/output" -cfg "configs/$GAME/analysis/generate.yml"
# generate statistics for all sizes together
python cli.py statsms "$ROOT/output/levels_*x*.json" "$ROOT/statistics\statistics.yml" -cfg "configs/$GAME/analysis/statistics.yml"

# generate a single statistics set (size: 9x9)
# python cli.py stats "$ROOT/output/levels_9x9.json" "$ROOT/statistics/statistics_9x9.json" -cfg "configs/$GAME/analysis/statistics.yml"

# generate ctrl levels
python cli.py cgenms "$ROOT/checkpoints/model_%END.pt" "$ROOT/condition_models/GMM" "$root/output" -cfg "configs/$GAME/analysis/generate_ctrl.yml" 
# generate statistics for all sizes together
python cli.py cstatsms "$ROOT/output/ctrl_levels_*x*.json" "$ROOT/statistics\statistics_ctrl.yml" -cfg "configs/$GAME/analysis/statistics_ctrl.yml"