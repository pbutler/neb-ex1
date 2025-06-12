#!/bin/bash

#SBATCH --job-name=testlightning
#SBATCH --output=results/llm-%4j.out
#SBATCH --error=results/llm-%4j.err
#SBATCH --nodes=2
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=8
#SBATCH --mem=0

source .venv/bin/activate
source .envrc
export MLFLOW_EXPERIMENT_NAME=llm-finetune

#srun python3 llm.py
srun ./wrapper.sh llm.py -e 1
