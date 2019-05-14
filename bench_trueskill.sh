#!/bin/bash
#SBATCH --job-name=DIPLOMACY_BENCH.trueskill.py
#SBATCH --time=3:0:0
#SBATCH --array=1-20%1

eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

SCRIPT=bench_trueskill
SCRIPT_PATH="$(pwd)"

PROJ_PATH=$1
GAMES=$2
STATS=$3
RULES=$4
EXCLUDE_DAIDE=$5
WORKING_DIR=/Tmp/${USER}/slurm-${SLURM_JOB_ID}/diplomacy

if [ -z "${RULES}" ]; then
  RULES=NO_PRESS,IGNORE_ERRORS,POWER_CHOICE
fi

STD_OUT=${PROJ_PATH}/results/${SCRIPT}.${STATS}.${RULES}.${EXCLUDE_DAIDE}.stats.${SLURM_ARRAY_TASK_ID}
STD_ERR=${PROJ_PATH}/results/${SCRIPT}.${STATS}.${RULES}.${EXCLUDE_DAIDE}.log.${SLURM_ARRAY_TASK_ID}
GAME_DIR=${PROJ_PATH}/results/games_${SCRIPT}

echo PROJ_PATH=${PROJ_PATH}
echo GAMES=${GAMES}
echo STATS=${STATS}
echo RULES=${RULES}
echo EXCLUDE_DAIDE=${EXCLUDE_DAIDE}
echo WORKING_DIR=${WORKING_DIR}
echo GAME_DIR=${GAME_DIR}
echo rsync --ignore-existing -ar ${PROJ_PATH}/data/containers ${WORKING_DIR}
echo rsync --ignore-existing -ar ${PROJ_PATH}/data/data ${WORKING_DIR}
echo WORKING_DIR=${WORKING_DIR} python ${SCRIPT_PATH}/${SCRIPT}.py --games=${GAMES} --stats=${STATS} --save-dir=${GAME_DIR} --rules=${RULES} ${EXCLUDE_DAIDE}
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

WORKING_DIR=${WORKING_DIR} python ${SCRIPT_PATH}/${SCRIPT}.py --games=${GAMES} --stats=${STATS} --save-dir=${GAME_DIR} --rules=${RULES} ${EXCLUDE_DAIDE} >> ${STD_OUT} 2>> ${STD_ERR}

kill -9 $(pgrep tensorflow_mode)
