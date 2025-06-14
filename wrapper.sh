#!/bin/bash

export GPUS_PER_NODE=8
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

export MLFLOW_SYSTEM_METRICS_NODE_ID=node_$SLURM_NODEID

#    --same-network \
#    --multi-gpu \
    #--config_file deepspeed.yaml \

export LAUNCHER="accelerate launch \
    --config_file fsdp2_config.yaml \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --machine_rank $SLURM_PROCID \
    --rdzv_backend static \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    "

echo $LAUNCHER $@
$LAUNCHER $@
