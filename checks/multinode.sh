#!/bin/bash

#SBATCH --job-name=multinode
#SBATCH --output=results/multinode-%4j.out
#SBATCH --error=results/multinode-%4j.err
#SBATCH --nodes=2                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:1                # number of GPUs per node
#SBATCH --cpus-per-task=16         # number of cores per tasks
#SBATCH --time=01:59:00             # maximum execution time (HH:MM:SS)

######################
### Set environment ###
######################
source .venv/bin/activate
source .envrc
export GPUS_PER_NODE=1
export MLFLOW_SYSTEM_METRICS_NODE_ID=$SLURM_NODEID
######################

######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
######################

export SCRIPT="complete_nlp_example.py"
export SCRIPT_ARGS=" \
    --mixed_precision fp16 \
    --output_dir outputs-mn \
    --with_tracking \
    "
srun ./wrapper.sh $SCRIPT $SCRIPT_ARGS

#export MLFLOW_SYSTEM_METRICS_NODE_ID=$(srun bash -c 'echo $SLURM_NODEID')
#srun bash -c 'echo $MLFLOW_SYSTEM_METRICS_NODE_ID bash'
#export LAUNCHER="accelerate launch \
#    --config_file fsdp_config.yaml \
#    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
#    --num_machines $SLURM_NNODES \
#    --machine_rank $SLURM_NODEID \
#    --rdzv_backend c10d \
#    --main_process_ip $head_node_ip \
#    --main_process_port 29500 \
#    "
#export SCRIPT="complete_nlp_example.py"
#export SCRIPT_ARGS=" \
#    --mixed_precision fp16 \
#    --output_dir outputs-mn \
#    --with_tracking \
#    "
#    
## This step is necessary because accelerate launch does not handle multiline arguments properly
#export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS" 
#srun $CMD
