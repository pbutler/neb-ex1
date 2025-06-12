# Nebius Interview ex1-poc

## Instructions
1) setup a 2node x8 H100 GPU cluster
2) set variables in `envrc.example` and change name to `.envrc`
3) submit job via `sbatch llm.sh`
4) run `model-evaluate.py` on before & after models

## Assumptions
1) Went with small instruct model, good for reasoning + function calling
2) Finetuned full model, PEFT has advantages (speed, and memory usage for one)
   but I wanted to show off the model distributed
3) No quantization for same reasons and accuracy

## Bugs/Suggestions/Improvements
1) Wierdness with saving the fsdp model, used some suggestions I found on a 
   github bug report having to do w/ stat dict sharding
2) Not 100% infiniband is working, benchmarks say yes, speed of model? no
3)
