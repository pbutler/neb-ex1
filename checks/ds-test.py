import os
import huggingface_hub as hfh
from datasets import load_dataset

hf_token = os.getenv("HF_TOKEN")
if hf_token is None:
    raise EnvironmentError("HF_TOKEN is not set!")
hfh.login(hf_token)

# Loading the dataset
dataset = load_dataset("Salesforce/xlam-function-calling-60k", split="train", token=hf_token)

