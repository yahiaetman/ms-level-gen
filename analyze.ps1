$GAME = $args[0] #"sokoban"
$ROOT = $args[1] #"runs/2022-07-27_18-49-22_SOKOBAN_DIVS_DIVR_PREW_AUG_DATASET_DIV_KDECOND_GFLOW_SNAKE_MSCEGRU_GEN"
if($GAME -eq "sokoban") {
    $CONDITIONS = @("solution-length", "pushed-crates")
} elseif($GAME -eq "zelda") {
    $CONDITIONS = @("path-length", "nearest-enemy-distance", "enemies")
} elseif($GAME -eq "dave") {
    $CONDITIONS = @("solution-length", "jumps", "spikes")
} else {
    Write-Error "Unknown game: $GAME"
    exit
}

# # generate GMMs (assumes the game and condition config in weights_path/../config.yml)
python cli.py cmms "$ROOT/checkpoints/dataset_%END_*x*.json" "$ROOT/condition_models/GMM" -cfg "configs/$GAME/ms_conditions/gmm.yml"

# # generate levels (assumes the generator config in weights_path/../config.yml)
python cli.py genms "$ROOT/checkpoints/model_%END.pt" "$ROOT/condition_models/GMM" "$root/output" -cfg "configs/$GAME/analysis/generate.yml"

# # generate statistics for all sizes together
python cli.py statsms "$ROOT/output/levels_*x*.json" "$ROOT/statistics/statistics.yml" -cfg "configs/$GAME/analysis/statistics.yml"

# # generate er for all sizes together
python cli.py erms "$ROOT/output/levels_*x*.json" "$ROOT/statistics/er" -cfg "configs/$GAME/heatmap.yml"

python cli.py imms "$ROOT/output/levels_*x*.json" "$ROOT/statistics/images" 5

python cli.py perimms "$ROOT/output/levels_*x*.json" "$ROOT/statistics/percentile_images" 5 $CONDITIONS

# python cli.py imms "$ROOT/output/levels_*x*.json" "$ROOT/extra_statistics/images" 4

# python cli.py perimms "$ROOT/output/levels_*x*.json" "$ROOT/extra_statistics/fig_percentile_images" 3 $CONDITIONS -f

# generate ctrl levels
python cli.py cgenms "$ROOT/checkpoints/model_%END.pt" "$ROOT/condition_models/GMM" "$root/output" -cfg "configs/$GAME/analysis/generate_ctrl.yml" 

# generate statistics for all sizes together
python cli.py cstatsms "$ROOT/output/ctrl_levels_*x*.json" "$ROOT/statistics\statistics_ctrl.yml" -cfg "configs/$GAME/analysis/statistics_ctrl.yml"