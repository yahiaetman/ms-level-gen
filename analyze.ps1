param ($GAME, $ROOT, [switch] $TAILOREDGMM=$false)
# Parameters:
#  - $GAME: the game's name (e.g. sokoban, zelda, dave)
#  - $ROOT: the path to the experiment folder (which contains the experiment config file "config.yml")
#  - $TAILOREDGMM: whether to use tailored GMMs for out-of-training sizes or not.

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

# generate GMMs (assumes the game and condition config in weights_path/../config.yml)
Write-Output "Fitting the GMMs ..."
python cli.py cmms "$ROOT/checkpoints/dataset_%END_*x*.json" "$ROOT/condition_models/GMM" -cfg "configs/$GAME/ms_conditions/gmm.yml"

if($TAILOREDGMM){
    # generate levels for oot sizes to fit GMM (assumes the generator config in weights_path/../config.yml)
    Write-Output "Generating levels for out-of-training sizes ..."
    python cli.py genms "$ROOT/checkpoints/model_%END.pt" "$ROOT/condition_models/GMM" "$root/initial_output" -f oot -cfg "configs/$GAME/analysis/generate.yml"
    
    # continue generating GMMs (assumes the game and condition config in weights_path/../config.yml)
    Write-Output "Fitting GMMs for out-of-training sizes ..."
    python cli.py cmms "$ROOT/initial_output/levels_*x*.json" "$ROOT/condition_models/GMM" -e "$ROOT/condition_models/GMM.cm" -cfg "configs/$GAME/ms_conditions/gmm.yml"
}

# generate levels (assumes the generator config in weights_path/../config.yml)
Write-Output "Generating levels for all the sizes ..."
python cli.py genms "$ROOT/checkpoints/model_%END.pt" "$ROOT/condition_models/GMM" "$root/output" -cfg "configs/$GAME/analysis/generate.yml"
# generate statistics for all sizes together
Write-Output "Computing statistics for all the sizes ..."
python cli.py statsms "$ROOT/output/levels_*x*.json" "$ROOT/statistics/statistics.yml" -cfg "configs/$GAME/analysis/statistics.yml"

# generate er for all sizes together
Write-Output "Rendering the ER for all the sizes ..."
python cli.py erms "$ROOT/output/levels_*x*.json" "$ROOT/statistics/er" -cfg "configs/$GAME/heatmap.yml"
# sample levels and render them to an image
Write-Output "Sampling level images for all the sizes ..."
python cli.py imms "$ROOT/output/levels_*x*.json" "$ROOT/statistics/images" 5
# sample images at percentiles and render them to a figure
Write-Output "Sampling level images at percentiles for all the sizes ..."
python cli.py perimms "$ROOT/output/levels_*x*.json" "$ROOT/extra_statistics/fig_percentile_images" 3 $CONDITIONS -f

# generate ctrl levels
Write-Output "Generating levels to test the controllability for all the sizes ..."
python cli.py cgenms "$ROOT/checkpoints/model_%END.pt" "$ROOT/condition_models/GMM" "$root/output" -cfg "configs/$GAME/analysis/generate_ctrl.yml" 
# generate statistics for all sizes together
Write-Output "Computing controllability statistics for all the sizes ..."
python cli.py cstatsms "$ROOT/output/ctrl_levels_*x*.json" "$ROOT/statistics\statistics_ctrl.yml" -cfg "configs/$GAME/analysis/statistics_ctrl.yml"
