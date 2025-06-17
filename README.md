# Nebius Interview ex1-poc

## Instructions

1. Setup a 2xnode 8xH100 GPU cluster: [instructions](./infrastructure/README.md)
    a. Setup an mlflow managed cluster, for monitoring and recording runs
2. On built cluster, clone this repository.
3. Install [uv](https://github.com/astral-sh/uv)
4. Run `uv sync` in this directory
5. set variables in `envrc.example` and change name to `.envrc`

6. ```bash
    source .envrc
    source .venv/bin/activate
   ```

7. submit jobs via `sbatch llm.sh` (you may choose to change the # of epochs
   to run for by modifying the `-e` option)

    ```
    usage: llm.py [-h] [-q] [--model_name MODEL_NAME] [-s SAVE] [-e EPOCHS] [--dataset_name DATASET_NAME] [--output OUTPUT]

    options:
      -h, --help            show this help message and exit
      -q, --quiet           don't print status messages to stdout
      --model_name MODEL_NAME
      -s SAVE, --save SAVE  samples to save for eval
      -e EPOCHS, --epochs EPOCHS
                            epochs to run for
      --dataset_name DATASET_NAME
      --output OUTPUT       model output dir
    ```

8. Check on runs via the mlflow public endpoint.
9. run `model-evaluate.py` on before & after models.

## Assumptions

1. Went with small instruct model, good for reasoning + function calling
2. Fine-tuned full model, PEFT has advantages (speed, and memory usage for one)
   but I wanted to show off the model distributed
3. No quantization for same reasons and accuracy

## Bugs/Suggestions/Improvements

1. Weirdness with saving the FSDP model, used some suggestions I found on a
   github bug report having to do w/ state dict sharding
2. Not 100% sure infiniband is working, benchmarks say yes, speed of model? no
3. Git commits are not pretty until the end
4. Rouge score is not the best metric but it's a stand-in for this example

* Tagged what was submitted to the interview under `submission`  tag but I am
  adding om QoL improvements and documentation
