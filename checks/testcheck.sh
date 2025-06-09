#!/bin/bash

#SBATCH --job-name=testlightning
#SBATCH --output=results/testlightning-%4j.out
#SBATCH --error=results/testlightning-%4j.err
#SBATCH --ntasks=2 
#SBATCH --ntasks-per-gpu=1
#SBATCH --mem=16G

source .venv/bin/activate
source .envrc


srun python3 test.py
#  # Print hello from one node
#  srun bash -c 'echo "Hello from $(hostname)"'
#  
#  # Run a job step with resource allocations taken from SBATCH
#  srun bash -c 'echo "Run nvidia-smi on $(hostname):" && nvidia-smi'
#  
#  # Run a job step with redefined resource allocations
#  srun --cpus-per-task=2 \
#      bash -c 'echo "Number of CPUs available for the job step:" && nproc'
