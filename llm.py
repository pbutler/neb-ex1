#!/usr/bin/env python3
import importlib.util
import json
import multiprocessing as mp
import os

import datasets as ds
import mlflow.config
import torch
from accelerate import Accelerator
from accelerate import PartialState
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from trl import SFTConfig
from trl import SFTTrainer
#from peft import LoraConfig

# check if flash attention is installed...
if importlib.util.find_spec("flash_attn"):
    attn_method = "flash_attention_2"
else:
    print("Flash Attention not installed, using default attention implementation.")
    attn_method = "sdpa"

mlflow.config.enable_system_metrics_logging()
try: #if running under SLURM TORCHRUN 
    mlflow.config.set_system_metrics_node_id(os.environ["SLURM_NODEID"])

    print("{}".format(os.environ.get("SLURM_NODEID", "NONE")))
    print("Local: {}/{}".format(os.environ["LOCAL_RANK"], os.environ["LOCAL_WORLD_SIZE"]), end=" ")
    print("Global: {}/{}".format(os.environ["RANK"], os.environ["WORLD_SIZE"]), end=" ")
    print("Node: {}/{}".format(os.environ["GROUP_RANK"], os.environ["GROUP_WORLD_SIZE"]))
except: #Pass quietly through if not running under SLURM TORCHRUN # noqa
    pass

mlflow.autolog()
device_string = PartialState().process_index

torch_dtype = torch.bfloat16
max_seq_length = 1024     # Unsloth auto supports RoPE Scaling internally!

#quantization_config = BitsAndBytesConfig(
#        load_in_4bit=True,
#        bnb_4bit_use_double_quant=True,
#        bnb_4bit_quant_type="nf4",
#        bnb_4bit_compute_dtype=torch_dtype,
#        bnb_4bit_quant_storage=quant_storage_dtype,
#    )

def main(args):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="don't print status messages to stdout")
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("-s", "--save", default=100, type=int, help="samples to save for eval")
    parser.add_argument("-e", "--epochs", default=5, type=int, help="epochs to run for")
    parser.add_argument("--dataset_name", default="Salesforce/xlam-function-calling-60k")
    parser.add_argument("--output", default="fsdp_output", help="model output dir")
    options = parser.parse_args()

    accelerator = Accelerator(log_with="mlflow")
    accelerator.init_trackers("llm-finetune", config={})



    tokenizer = AutoTokenizer.from_pretrained(options.model_name, use_fast=True)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def formatting_prompts_func(examples):
        """
        Format prompts from the dataset adds a text column with the full 
        conversation and a prompt column with the user query and tools.

        :param examples: dataset, which is a dictionary with keys 
        'query', 'tools', and 'answers'.
        """
        convos = []
        prompts = []
        
        # Iterate through each item in the batch (examples are structured as lists of values)
        for query, tools, answers in zip(examples['query'], examples['tools'], examples['answers']):
            tool_user = {
                "content": f"You are a helpful assistant with access to the following tools or function calls. Your task is to produce a sequence of tools or function calls necessary to generate response to the user utterance. Use the following tools or function calls as required:\n{tools}",
                "role": "system"
            }
            ques_user = {
                "content": f"{query}",
                "role": "user"
            }
            assistant = {
                "content": f"{answers}",
                "role": "assistant"
            }
            #convos is the fullconversation: prompt + answer
            convos.append([tool_user, ques_user, assistant])
            #prompts is the prompt only: prompt
            prompts.append([tool_user, ques_user])

        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos ]
        prompts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in prompts ]
        #texts = [tokenizer.apply_chat_template(convo, add_generation_prompt=True) for convo in convos]
        return {"text": texts, "prompt": prompts}

    # Load the dataset
    with accelerator.main_process_first():
        full_dataset = ds.load_dataset("Salesforce/xlam-function-calling-60k", split="train")  #, token=hf_token)

    # Format dataset, select only full text columns
    with accelerator.main_process_first():
    # Apply the formatting on dataset
        full_dataset = full_dataset.map(formatting_prompts_func, batched = True).select_columns(["text"])  
        full_dataset = full_dataset.select_columns(["text"])  #remove_columns(["tools", "query", "answers"])

    #save the first options.save samples for eval
    dataset = full_dataset.select(range(options.save, 60000))

    # show an example encoded and decoded text to confirm tokenizer works
    if not options.quiet and accelerator.is_main_process:
        print(json.dumps(dataset[0], indent=2))

        sample_text = dataset[0]["text"]
        encoded_input = tokenizer(sample_text, return_tensors='pt')
        decoded_output = tokenizer.decode(encoded_input['input_ids'][0])

        print("Encoded:", encoded_input)
        print("Decoded:", decoded_output)

    #Model
    print(f"Starting to load the model {options.model_name} into memory")
    model = AutoModelForCausalLM.from_pretrained(
        options.model_name,
        #torch_dtype=quant_storage_dtype,
        attn_implementation = attn_method,
    )
    model.config.use_cache = False 

    # If there is a mismatch between tokenizer vocab size and embedding matrix,
    # throw a warning and then expand the embedding matrix
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print(
            "WARNING: Resizing the embedding matrix to match the tokenizer vocab size."
        )
        model.resize_token_embeddings(len(tokenizer))

    #Training
    args = SFTConfig( #TrainingArguments(
            per_device_train_batch_size = 4,  # Controls the batch size per device
            gradient_accumulation_steps = 4,  # Accumulates gradients to simulate a larger batch
            warmup_steps = 5,
            learning_rate = 2e-4,             # Sets the learning rate for optimization
            num_train_epochs = options.epochs,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            optim = "adamw_torch",
            #optim = "paged_adamw_32bit",
            weight_decay = 0.01,              # Regularization term for preventing overfitting
            lr_scheduler_type = "linear",     # Chooses a linear learning rate decay
            seed = 31415,                        
            output_dir = "outputs",             
            report_to = "mlflow",              # Enables mlflow logging
            logging_steps = 50,                # Sets frequency of logging to W&
            logging_strategy = "steps",       # Logs metrics at each specified step
            save_strategy = "no",               
            load_best_model_at_end = True,    # Loads the best model at the end
            save_only_model = False,          # Saves entire model, not only weights
            log_level="debug",
            dataset_text_field = "text",
            dataset_num_proc = mp.cpu_count(),
            # dataset_kwargs={
            #     "add_special_tokens": False,  # We template with special tokens
            #     "append_concat_token": False, # No need to add additional separator token
            # },
            max_seq_length = max_seq_length,
            packing = False,        # Can make training 5x faster for short sequences.
        )

    # # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    # peft_config = LoraConfig(
    #     lora_alpha=8,
    #     lora_dropout=0.05,
    #     r=16,
    #     bias="none",
    #     target_modules="all-linear",
    #     task_type="CAUSAL_LM",
    #     modules_to_save = ["lm_head", "embed_tokens"] # add if you want to use the Llama 3 instruct template
    # )

    #model = get_peft_model(model, peft_config)

    # Initialize the SFTTrainer with the model, tokenizer, and dataset
    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = dataset,
        #peft_config=peft_config,
        args = args
    )

    accelerator.wait_for_everyone() 
    accelerator.print("Starting training run")
    trainer_stats = trainer.train()  
    print(trainer_stats)

    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")

    accelerator.wait_for_everyone() 

    accelerator.print("Saving model")
    #according to github there is a bug with sharded state dicts
    if accelerator.state.fsdp_plugin is not None:
        accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    accelerator.wait_for_everyone()

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        options.output,
        is_main_process=accelerator.is_main_process,
        #save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
    )

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        accelerator.print("tokens")
        tokenizer.save_pretrained(options.output)

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))
