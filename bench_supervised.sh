#!/bin/bash
#SBATCH --job-name=DIPLOMACY_BENCH.supervised_rules.py

PROJ_PATH=$1

for i in {1..20}
do
  SLURM_ARRAY_TASK_ID=$i ./bench_model_v_rules.sh $PROJ_PATH supervised random
  SLURM_ARRAY_TASK_ID=$i ./bench_model_v_rules.sh $PROJ_PATH supervised dumbbot
  SLURM_ARRAY_TASK_ID=$i ./bench_model_v_rules.sh $PROJ_PATH supervised easy
done
