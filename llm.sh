#!/bin/bash

#SBATCH --job-name=testlightning
#SBATCH --output=results/llm-%4j.out
#SBATCH --error=results/llm-%4j.err
#SBATCH --ntasks=2 
#SBATCH --ntasks-per-gpu=1
#SBATCH --mem=0
#SBATCH --cpus-per-gpu=16

source .venv/bin/activate
source .envrc
export MLFLOW_EXPERIMENT_NAME=llm-finetune

#srun python3 llm.py
srun ./wrapper.sh llm.py
#  # Print hello from one node
#  srun bash -c 'echo "Hello from $(hostname)"'
#  
#  # Run a job step with resource allocations taken from SBATCH
#  srun bash -c 'echo "Run nvidia-smi on $(hostname):" && nvidia-smi'
#  
#  # Run a job step with redefined resource allocations
#  srun --cpus-per-task=2 \
#      bash -c 'echo "Number of CPUs available for the job step:" && nproc'
