#!/bin/bash
#SBATCH --job-name=DIPLOMACY_BENCH.model_v_rules.py
#SBATCH --array=1-20%1

eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

SCRIPT=bench_model_v_rules
SCRIPT_PATH="$(cd $(dirname $SCRIPT.py);pwd)"

PROJ_PATH=$1
MODEL_AI=$2
NON_MODEL_AI=$3

WORKING_DIR=/Tmp/$USER/slurm-$SLURM_JOB_ID/diplomacy

STD_OUT=$PROJ_PATH/results/$SCRIPT.$MODEL_AI.$NON_MODEL_AI.stats.$SLURM_ARRAY_TASK_ID
STD_ERR=$PROJ_PATH/results/$SCRIPT.$MODEL_AI.$NON_MODEL_AI.log.$SLURM_ARRAY_TASK_ID
GAME_DIR=$PROJ_PATH/results/games_$SCRIPT.$MODEL_AI.$NON_MODEL_AI

echo PROJ_PATH=$PROJ_PATH
echo WORKING_DIR=$WORKING_DIR
echo GAME_DIR=$GAME_DIR
echo rsync --ignore-existing -ar $PROJ_PATH/data/containers $WORKING_DIR
echo rsync --ignore-existing -ar $PROJ_PATH/data/data $WORKING_DIR
echo WORKING_DIR=$WORKING_DIR python $SCRIPT_PATH/$SCRIPT.py --model-ai=$MODEL_AI --non-model-ai=$NON_MODEL_AI --games=50
echo python output: $STD_OUT
echo python error: $STD_ERR

mkdir -p $WORKING_DIR
rsync --ignore-existing -ar $PROJ_PATH/data/containers $WORKING_DIR
rsync --ignore-existing -ar $PROJ_PATH/data/data $WORKING_DIR

mkdir -p $GAME_DIR
touch $STD_OUT
touch $STD_ERR

cd $GAME_DIR

pyenv activate diplomacy_bench
module load singularity/3.1.1

WORKING_DIR=$WORKING_DIR python $SCRIPT_PATH/$SCRIPT.py --model-ai=$MODEL_AI --non-model-ai=$NON_MODEL_AI --games=50 >> $STD_OUT 2>> $STD_ERR

kill -9 $(pgrep tensorflow_mode)
