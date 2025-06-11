
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
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

mlflow.config.enable_system_metrics_logging()
mlflow.config.set_system_metrics_node_id(os.environ["SLURM_NODEID"])
accelerator = Accelerator()

device_string = PartialState().process_index

try:
    print("Local: {}/{}".format(os.environ["LOCAL_RANK"], os.environ["LOCAL_WORLD_SIZE"]), end=" ")
    print("Global: {}/{}".format(os.environ["RANK"], os.environ["WORLD_SIZE"]), end=" ")
    print("Node: {}/{}".format(os.environ["GROUP_RANK"], os.environ["GROUP_WORLD_SIZE"]))
except:
    pass
# hf_token = os.getenv("HUF_TOKEN")
# if hf_token is None:
#     raise EnvironmentError("HUF_TOKEN is not set!")
# hfh.login(hf_token)


torch_dtype = torch.bfloat16
quant_storage_dtype = None #torch.bfloat16
model_name = "meta-llama/Llama-3.1-8B-Instruct"
max_seq_length = 1024     # Unsloth auto supports RoPE Scaling internally!
load_in_4bit = False      # Use 4bit quantization to reduce memory usage. Can be False.
test_prompt='<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 9 Jun 2025\n\nYou are a helpful assistant with access to the following tools or function calls. Your task is to produce a sequence of tools or function calls necessary to generate response to the user utterance. Use the following tools or function calls as required:\n[{"name": "live_giveaways_by_type", "description": "Retrieve live giveaways from the GamerPower API based on the specified type.", "parameters": {"type": {"description": "The type of giveaways to retrieve (e.g., game, loot, beta).", "type": "str", "default": "game"}}}]<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhere can I find live giveaways for beta access and games?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
test_answer = '[{"name": "live_giveaways_by_type", "arguments": {"type": "beta"}}, {"name": "live_giveaways_by_type", "arguments":{"type": "game"}}]<|eot_id|>'

def testprompt(prompt, answer, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, do_sample=True, max_new_tokens=128)  #  , do_sample=False, temperature=0.0, max_new_tokens=150)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if accelerator.is_main_process:
        print("*"*10)
        print(prompt)
        print("-"*10)
        print(answer)
        print("#"*10)
        print(outputs)
        print("#"*10)
        print(result)
        print("*"*10)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
#tokenizer.pad_token = "<|eot_id|>"
#tokenizer.pad_token_id = 128009
tokenizer.padding_side = 'right'
#tokenizer.pad_token = tokenizer.eos_token

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

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

#Model
print(f"Starting to load the model {model_name} into memory")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
#    device_map={'':device_string}
    #quantization_config=quantization_config,
    torch_dtype=quant_storage_dtype,
    #attn_implementation = "sdpa",
    attn_implementation = "flash_attention_2",
    #max_seq_length = max_seq_length,
    #dtype = dtype,
    #load_in_4bit = load_in_4bit,
)
# if model.config.pad_token_id is None:
#     model.config.pad_token_id = model.config.eos_token_id
#model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False 

# If there is a mismatch between tokenizer vocab size and embedding matrix,
# throw a warning and then expand the embedding matrix
if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
    print(
        "WARNING: Resizing the embedding matrix to match the tokenizer vocab size."
    )
    model.resize_token_embeddings(len(tokenizer))

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

    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos ]
    #texts = [tokenizer.apply_chat_template(convo, add_generation_prompt=True) for convo in convos]
    return {"text": texts}

with accelerator.main_process_first():
# Apply the formatting on dataset
    dataset = dataset.map(formatting_prompts_func, batched = True).select_columns(["text"])  #remove_columns(["tools", "query", "answers"])

full_dataset = dataset
dataset = full_dataset #.select(range(15000))
#dataset = full_dataset[:15000]

#Add the EOS token
# def process(row):
#     row["query"] = "<user>"+row["query"]+"</user>\n\n"
# 
#     tools = []
#     for t in json.loads(row["tools"]):
#       tools.append(str(t))
# 
#     answers = []
#     for a in json.loads(row["answers"]):
#       answers.append(str(a))
# 
#     row["tools"] = "<tools>"+"\n".join(tools)+"</tools>\n\n"
#     row["answers"] = "<calls>"+"\n".join(answers)+"</calls>"
#     row["text"] = row["query"]+row["tools"]+row["answers"]+tokenizer.eos_token
#     return row

#with accelerator.main_process_first():
#    dataset = dataset.map(
#        process,
#        num_proc=64,
#    )

if accelerator.is_main_process:
    print(json.dumps(dataset[0], indent=2))

    sample_text = dataset[0]["text"]
    encoded_input = tokenizer(sample_text, return_tensors='pt')
    decoded_output = tokenizer.decode(encoded_input['input_ids'][0])

    print("Encoded:", encoded_input)
    print("Decoded:", decoded_output)


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

def last_format(row):
    return row["text"] + row["answers"] + "<|eot_id|><|end_of_text|>"

args = SFTConfig( #TrainingArguments(
        per_device_train_batch_size = 4,  # Controls the batch size per device
        gradient_accumulation_steps = 4,  # Accumulates gradients to simulate a larger batch
        warmup_steps = 5,
        learning_rate = 2e-4,             # Sets the learning rate for optimization
        num_train_epochs = 5,
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

#model, tokenizer, dataset = accelerator.prepare(model, tokenizer, dataset)


trainer = SFTTrainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = dataset,
    #formatting_func=last_format,
    #peft_config=peft_config,
    args = args
)

# Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

accelerator.wait_for_everyone() 

trainer_stats = trainer.train()  
print(trainer_stats)

# Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
#print(f"Peak reserved memory = {used_memory} GB.")
#print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
#print(f"Peak reserved memory % of max memory = {used_percentage} %.")
#print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

accelerator.wait_for_everyone() 
if accelerator.state.fsdp_plugin is not None:
    accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(
      "fsdp_output",
      is_main_process=accelerator.is_main_process,
      #save_function=accelerator.save,
      state_dict=accelerator.get_state_dict(model),
)

accelerator.wait_for_everyone()

if accelerator.is_main_process:
    accelerator.print("tokens")
    tokenizer.save_pretrained("fsdp_output")


#if accelerator.is_main_process:
#    print("saving model")
#    model.save_pretrained("demoday-test")
#    print("saving tokenizer")
#    #if accelerator.is_main_process:
#    # Local saving

#testprompt(test_prompt, test_answer, tokenizer, model)
accelerator.wait_for_everyone()

#prompt = "<user>Check if the numbers 8 and 1233 are powers of two.</user>\n\n<tools>"

# print("*"*80)
# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
# #    torch_dtype=torch.float16,
#     device_map="auto",
# )
# outputs = pipe(prompt, max_new_tokens=120, do_sample=True)



# unwrapped_model = accelerator.unwrap_model(model)
# unwrapped_model.save_pretrained(
#     "demoday-test",
#     is_main_process=accelerator.is_main_process,
#     save_function=accelerator.save,
# )
accelerator.end_training()
print("done")
