$RUNS = 5

# This script runs 5 * 10 = 50 experiments so it will run for a REALLY LONG time.
# It is meant to be used to replicate all the experiments in the paper.

for($i = 1; $i -le $RUNS; $i++){

    # Sokoban Experiments

    $GAME = "sokoban"

    Write-Output "${i}/${RUNS}: $GAME (Base Experiment) ..."
    $PATH = ".\experiments\$GAME\GFLOW\${i}"
    python .\cli.py train -cfg .\configs\$GAME\gflownet\base.yml -ovr save_path=$PATH
    ./analyze.ps1 $GAME $PATH

    Write-Output "${i}/${RUNS}: $GAME (Diversity Sampling: Push Signature) ..."
    $PATH = ".\experiments\$GAME\DIVPUSHSIG_GFLOW\${i}"
    python .\cli.py train -cfg .\configs\$GAME\gflownet\div-pushsig.yml -ovr save_path=$PATH
    ./analyze.ps1 $GAME $PATH

    Write-Output "${i}/${RUNS}: $GAME (Diversity Sampling: Tuple) ..."
    $PATH = ".\experiments\$GAME\DIV_GFLOW\${i}"
    python .\cli.py train -cfg .\configs\$GAME\gflownet\div-tuple.yml -ovr save_path=$PATH dataset_config.property_reward:=None dataset_config.data_augmentation:=False
    ./analyze.ps1 $GAME $PATH

    Write-Output "${i}/${RUNS}: $GAME (Diversity Sampling: Tuple + Property Reward) ..."
    $PATH = ".\experiments\$GAME\DIV_PR_GFLOW\${i}"
    python .\cli.py train -cfg .\configs\$GAME\gflownet\div-tuple.yml -ovr save_path=$PATH dataset_config.data_augmentation:=False
    ./analyze.ps1 $GAME $PATH

    Write-Output "${i}/${RUNS}: $GAME (Diversity Sampling: Tuple + Data Augmentation) ..."
    $PATH = ".\experiments\$GAME\DIV_AUG_GFLOW\${i}"
    python .\cli.py train -cfg .\configs\$GAME\gflownet\div-tuple.yml -ovr save_path=$PATH dataset_config.property_reward:=None
    ./analyze.ps1 $GAME $PATH

    Write-Output "${i}/${RUNS}: $GAME (Diversity Sampling: Tuple + Property Reward + Data Augmentation) ..."
    $PATH = ".\experiments\$GAME\DIV_PR_AUG_GFLOW\${i}"
    python .\cli.py train -cfg .\configs\$GAME\gflownet\div-tuple.yml -ovr save_path=$PATH
    ./analyze.ps1 $GAME $PATH

    # Zelda Experiments

    $GAME = "zelda"

    Write-Output "${i}/${RUNS}: $GAME (Diversity Sampling: Tuple + Property Reward) ..."
    $PATH = ".\experiments\$GAME\DIV_PR_GFLOW\${i}"
    python .\cli.py train -cfg .\configs\$GAME\gflownet\config.yml -ovr save_path=$PATH dataset_config.data_augmentation:=False
    ./analyze.ps1 $GAME $PATH -TAILOREDGMM

    Write-Output "${i}/${RUNS}: $GAME (Diversity Sampling: Tuple + Property Reward + Data Augmentation) ..."
    $PATH = ".\experiments\$GAME\DIV_PR_AUG_GFLOW\${i}"
    python .\cli.py train -cfg .\configs\$GAME\gflownet\config.yml -ovr save_path=$PATH
    ./analyze.ps1 $GAME $PATH -TAILOREDGMM

    # Danger Dave Experiments

    $GAME = "dave"

    Write-Output "${i}/${RUNS}: $GAME (Diversity Sampling: Tuple) ..."
    $PATH = ".\experiments\$GAME\DIV_GFLOW\${i}"
    python .\cli.py train -cfg .\configs\$GAME\gflownet\config.yml -ovr save_path=$PATH dataset_config.data_augmentation:=False
    ./analyze.ps1 $GAME $PATH -TAILOREDGMM

    Write-Output "${i}/${RUNS}: $GAME (Diversity Sampling: Tuple + Data Augmentation) ..."
    $PATH = ".\experiments\$GAME\DIV_AUG_GFLOW\${i}"
    python .\cli.py train -cfg .\configs\$GAME\gflownet\config.yml -ovr save_path=$PATH
    ./analyze.ps1 $GAME $PATH -TAILOREDGMM

}