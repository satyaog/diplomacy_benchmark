#!/bin/bash
#SBATCH --job-name=DIPLOMACY_BENCH.daide.py

PROJ_PATH=$1
START=$2
END=$3

for (( i=$START; i<=$END; i++ ))
do
  SLURM_ARRAY_TASK_ID=$i ./bench_model_v_daide.sh $PROJ_PATH
done
