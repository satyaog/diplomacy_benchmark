#!/bin/bash
#SBATCH --job-name=DIPLOMACY_BENCH.models.py
#SBATCH --time=3:0:0
#SBATCH --array=1-20%1

eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

SCRIPT=bench_models
SCRIPT_PATH="$(pwd)"

PROJ_PATH=$1
AI_1=$2
AI_2=$3
GAMES=$4
STATS=$5
RULES=$6
WORKING_DIR=/Tmp/${USER}/slurm-${SLURM_JOB_ID}/diplomacy

if [ -z "${RULES}" ]; then
  RULES=NO_PRESS,IGNORE_ERRORS,POWER_CHOICE
fi

STD_OUT=${PROJ_PATH}/results/${SCRIPT}.${AI_1}.${AI_2}.${STATS}.${RULES}.stats.${SLURM_ARRAY_TASK_ID}
STD_ERR=${PROJ_PATH}/results/${SCRIPT}.${AI_1}.${AI_2}.${STATS}.${RULES}.log.${SLURM_ARRAY_TASK_ID}
GAME_DIR=${PROJ_PATH}/results/games_${AI_1}_${AI_2}_${SCRIPT}

echo PROJ_PATH=${PROJ_PATH}
echo AI_1=${AI_1}
echo AI_2=${AI_2}
echo GAMES=${GAMES}
echo STATS=${STATS}
echo RULES=${RULES}
echo WORKING_DIR=${WORKING_DIR}
echo GAME_DIR=${GAME_DIR}
echo rsync --ignore-existing -ar ${PROJ_PATH}/data/containers ${WORKING_DIR}
echo rsync --ignore-existing -ar ${PROJ_PATH}/data/data ${WORKING_DIR}
echo WORKING_DIR=${WORKING_DIR} python ${SCRIPT_PATH}/${SCRIPT}.py --ai-1=${AI_1} --ai-2=${AI_2} --games=${GAMES} --stats=${STATS} --save-dir=${GAME_DIR} --rules=${RULES}
echo python output: ${STD_OUT}
echo python error: ${STD_ERR}

mkdir -p ${WORKING_DIR}
rsync --ignore-existing -ar ${PROJ_PATH}/data/containers ${WORKING_DIR}
rsync --ignore-existing -ar ${PROJ_PATH}/data/data ${WORKING_DIR}

mkdir -p ${GAME_DIR}
touch ${STD_OUT}
touch ${STD_ERR}

cd ${GAME_DIR}

pyenv activate diplomacy_benchmarks
module load singularity/3.1.1

WORKING_DIR=${WORKING_DIR} python ${SCRIPT_PATH}/${SCRIPT}.py --ai-1=${AI_1} --ai-2=${AI_2} --games=${GAMES} --stats=${STATS} --save-dir=${GAME_DIR} --rules=${RULES} >> ${STD_OUT} 2>> ${STD_ERR}

kill -9 $(pgrep tensorflow_mode)
