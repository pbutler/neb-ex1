
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
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

mlflow.config.enable_system_metrics_logging()
mlflow.config.set_system_metrics_node_id(os.environ["SLURM_NODEID"])
accelerator = Accelerator()

hf_token = os.getenv("HF_TOKEN")
if hf_token is None:
    raise EnvironmentError("HF_TOKEN is not set!")
hfh.login(hf_token)


max_seq_length = 2048     # Unsloth auto supports RoPE Scaling internally!
dtype = None              # None for auto detection
load_in_4bit = False      # Use 4bit quantization to reduce memory usage. Can be False.

model_name = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = "<|eot_id|>"
tokenizer.pad_token_id = 128009
tokenizer.padding_side = 'left'

#Model
print(f"Starting to load the model {model_name} into memory")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    #max_seq_length = max_seq_length,
    #dtype = dtype,
    #load_in_4bit = load_in_4bit,
)
model.config.pad_token_id = tokenizer.pad_token_id

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

    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts}

with accelerator.main_process_first():
# Apply the formatting on dataset
    dataset = dataset.map(formatting_prompts_func, batched = True,)

#Add the EOS token
def process(row):
    row["query"] = "<user>"+row["query"]+"</user>\n\n"

    tools = []
    for t in json.loads(row["tools"]):
      tools.append(str(t))

    answers = []
    for a in json.loads(row["answers"]):
      answers.append(str(a))

    row["tools"] = "<tools>"+"\n".join(tools)+"</tools>\n\n"
    row["answers"] = "<calls>"+"\n".join(answers)+"</calls>"
    row["text"] = row["query"]+row["tools"]+row["answers"]+tokenizer.eos_token
    return row

dataset = dataset.map(
    process,
    num_proc=mp.cpu_count(),
    load_from_cache_file=False,
)

args = SFTConfig( #TrainingArguments(
        per_device_train_batch_size = 2,  # Controls the batch size per device
        gradient_accumulation_steps = 16,  # Accumulates gradients to simulate a larger batch
        warmup_steps = 5,
        learning_rate = 2e-4,             # Sets the learning rate for optimization
        num_train_epochs = 3,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        #optim = "adamw_8bit",
        weight_decay = 0.01,              # Regularization term for preventing overfitting
        lr_scheduler_type = "linear",     # Chooses a linear learning rate decay
        seed = 31415,                        
        output_dir = "outputs",             
        report_to = "mlflow",              # Enables Weights & Biases (W&B) logging
        logging_steps = 1,                # Sets frequency of logging to W&B
        logging_strategy = "steps",       # Logs metrics at each specified step
        save_strategy = "no",               
        load_best_model_at_end = True,    # Loads the best model at the end
        save_only_model = False,          # Saves entire model, not only weights
        log_level="debug",
        dataset_text_field = "text",
        dataset_num_proc = 2,
        max_seq_length = max_seq_length,
        packing = False,        # Can make training 5x faster for short sequences.
    )


trainer = SFTTrainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = dataset,
    args = args
)

# Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


trainer_stats = trainer.train()  
print(trainer_stats)

# Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# Local saving
model.save_pretrained("demoday-test")
tokenizer.save_pretrained("demoday-test")

