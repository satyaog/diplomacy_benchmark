#!/bin/bash
#SBATCH --job-name=DIPLOMACY_BENCH.model_v_daide.py
#SBATCH --array=1-20%1

eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

SCRIPT=bench_trueskill
SCRIPT_PATH="$(cd $(dirname $SCRIPT.py);pwd)"

PROJ_PATH=$1
NO_PRESS=$2
WORKING_DIR=/Tmp/$USER/slurm-$SLURM_JOB_ID/diplomacy

STD_OUT=$PROJ_PATH/results/$SCRIPT.$NO_PRESS.stats.$SLURM_ARRAY_TASK_ID
STD_ERR=$PROJ_PATH/results/$SCRIPT.$NO_PRESS.log.$SLURM_ARRAY_TASK_ID
GAME_DIR=$PROJ_PATH/results/games_$SCRIPT

echo PROJ_PATH=$PROJ_PATH
echo NO_PRESS=$NO_PRESS
echo WORKING_DIR=$WORKING_DIR
echo GAME_DIR=$GAME_DIR
echo rsync --ignore-existing -ar $PROJ_PATH/data/containers $WORKING_DIR
echo rsync --ignore-existing -ar $PROJ_PATH/data/data $WORKING_DIR
echo WORKING_DIR=$WORKING_DIR python $SCRIPT_PATH/$SCRIPT.py --games=50 --rules=IGNORE_ERRORS,POWER_CHOICE
echo WORKING_DIR=$WORKING_DIR python $SCRIPT_PATH/$SCRIPT.py --games=50 --rules=NO_PRESS,IGNORE_ERRORS,POWER_CHOICE
echo python output: $STD_OUT
echo python error: $STD_ERR

mkdir -p $WORKING_DIR
rsync --ignore-existing -ar $PROJ_PATH/data/containers $WORKING_DIR
rsync --ignore-existing -ar $PROJ_PATH/data/data $WORKING_DIR

mkdir -p $GAME_DIR
touch $STD_OUT
touch $STD_ERR

cd $GAME_DIR

pyenv activate diplomacy_bench_daide
module load singularity/3.1.1

if [ -z "$NO_PRESS" ]; then
  WORKING_DIR=$WORKING_DIR python $SCRIPT_PATH/$SCRIPT.py --games=50 --rules=IGNORE_ERRORS,POWER_CHOICE >> $STD_OUT 2>> $STD_ERR
else
  WORKING_DIR=$WORKING_DIR python $SCRIPT_PATH/$SCRIPT.py --games=50 --rules=NO_PRESS,IGNORE_ERRORS,POWER_CHOICE >> $STD_OUT 2>> $STD_ERR
fi

kill -9 $(pgrep tensorflow_mode)
