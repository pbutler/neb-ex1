
import os
import multiprocessing as mp
import json

#from unsloth import FastLanguageModel
#from unsloth.chat_templates import get_chat_template
#from unsloth import unsloth_train
import mlflow
import torch

import huggingface_hub as hfh
from huggingface_hub import login
import datasets as ds
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig
from accelerate import Accelerator, PartialState
from accelerate import utils, tracking
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

mlflow.config.enable_system_metrics_logging()
mlflow.config.set_system_metrics_node_id(os.environ["SLURM_NODEID"])
mlflow.autolog()

device_string = PartialState().process_index

try:
    print("{}".format(os.environ["SLURM_NODEID"]))
    print("Local: {}/{}".format(os.environ["LOCAL_RANK"], os.environ["LOCAL_WORLD_SIZE"]), end=" ")
    print("Global: {}/{}".format(os.environ["RANK"], os.environ["WORLD_SIZE"]), end=" ")
    print("Node: {}/{}".format(os.environ["GROUP_RANK"], os.environ["GROUP_WORLD_SIZE"]))
except:
    pass


torch_dtype = torch.bfloat16
max_seq_length = 1024     # Unsloth auto supports RoPE Scaling internally!


terminators = [
    tokenizer.eos_token_id,
    #tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
    

quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_storage=quant_storage_dtype,
    )

#model = accelerator.prepare(model)
#testprompt(test_prompt, test_answer, tokenizer, model)

with accelerator.main_process_first():
    dataset = ds.load_dataset("Salesforce/xlam-function-calling-60k", split="train")  #, token=hf_token)

# # Initialize the tokenizer with the chat template and mapping
# tokenizer = get_chat_template(
#     tokenizer,
#     chat_template = "llama-3", 
#     mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
#     map_eos_token = True,        # Maps <|im_end|> to <|eot_id|> instead
# )

def formatting_prompts_func(examples):
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
        convos.append([tool_user, ques_user, assistant])
        prompts.append([tool_user, ques_user])

    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos ]
    prompts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in prompts ]
    #texts = [tokenizer.apply_chat_template(convo, add_generation_prompt=True) for convo in convos]
    return {"text": texts, "prompt": prompts}


def main(args):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-q", "--quiet", action="store_false", dest="verbose",
                        help="don't print status messages to stdout")
    parser.add_argument("--version", action="version",
                        version="%(prog)s " + __version__)
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("-s", "--save", default=100, type=int, help="samples to save for eval")
    parser.add_argument("-e", "--epochs", default=5, type=int, help="epochs to run for")
    parser.add_argument("--dataset_name", default="Salesforce/xlam-function-calling-60k")
    parser.add_argument("--output", default="fsdp_output", help="model output dir")
    options = parser.parse_args()

    accelerator = Accelerator(log_with="mlflow")
    accelerator.init_trackers("llm-finetune", config={})

    with accelerator.main_process_first():
    # Apply the formatting on dataset
        full_dataset = full_dataset.map(formatting_prompts_func, batched = True).select_columns(["text"])  
        full_dataset = full_dataset.select_columns(["text"])  #remove_columns(["tools", "query", "answers"])

    #save the first $s$ for eval
    dataset = full_dataset.select(range(options.save, 60000))

    tokenizer = AutoTokenizer.from_pretrained(options.model_name, use_fast=True)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if not options.quiet and accelerator.is_main_process:
        print(json.dumps(dataset[0], indent=2))

        sample_text = dataset[0]["text"]
        encoded_input = tokenizer(sample_text, return_tensors='pt')
        decoded_output = tokenizer.decode(encoded_input['input_ids'][0])

        print("Encoded:", encoded_input)
        print("Decoded:", decoded_output)


    #Model
    print(f"Starting to load the model {model_name} into memory")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=quant_storage_dtype,
        attn_implementation = "flash_attention_2",
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
            report_to = "mlflow",              # Enables Weights & Biases (W&B) logging
            logging_steps = 50,                # Sets frequency of logging to W&
            logging_strategy = "steps",       # Logs metrics at each specified step
            save_strategy = "no",               
            load_best_model_at_end = True,    # Loads the best model at the end
            save_only_model = False,          # Saves entire model, not only weights
            log_level="debug",
            dataset_text_field = "text",
            dataset_num_proc = 12,
            # dataset_kwargs={
            #     "add_special_tokens": False,  # We template with special tokens
            #     "append_concat_token": False, # No need to add additional separator token
            # },
            max_seq_length = max_seq_length,
            packing = False,        # Can make training 5x faster for short sequences.
        )

    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    peft_config = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save = ["lm_head", "embed_tokens"] # add if you want to use the Llama 3 instruct template
    )

    #model = get_peft_model(model, peft_config)

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
