# Nebius Interview ex1-poc

## Instructions

1) setup a 2node x8 H100 GPU cluster
2) `uv sync`
3) set variables in `envrc.example` and change name to `.envrc`
4) submit job via `sbatch llm.sh`
5) run `model-evaluate.py` on before & after models

## Assumptions

1) Went with small instruct model, good for reasoning + function calling
2) Fine-tuned full model, PEFT has advantages (speed, and memory usage for one)
   but I wanted to show off the model distributed
3) No quantization for same reasons and accuracy

## Bugs/Suggestions/Improvements

1) Wierdness with saving the FSDP model, used some suggestions I found on a
   github bug report having to do w/ state dict sharding
2) Not 100% infiniband is working, benchmarks say yes, speed of model? no
3) Git commits are not pretty until the end
4) Rouge score is not the best metric but it's a stand-in for this example

* Tagged what was submitted to the interview under `submission`  tag but I am
  adding om QoL improvements and documentation

## Todo

1) More documentation
