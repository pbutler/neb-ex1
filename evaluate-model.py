#!/usr/bin/env python

import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import datasets as ds
from rouge_score import rouge_scorer



def main(args):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="don't print status messages to stdout")
    parser.add_argument("model_name")
    parser.add_argument("-n", "--samples", default=100, type=int)
    parser.add_argument("--dataset_name", default="Salesforce/xlam-function-calling-60k")
    options = parser.parse_args()

    def formatting_prompts_func(examples):
        convos = []
        prompts = []

        # Iterate through each item in the batch (examples are structured as lists of values)
        for query, tools, answers in zip(examples['query'], examples['tools'], examples['answers']):
            tool_user = {
                "content": f"You are a helpful assistant with access to the following tools or function calls. Your task is toproduce a sequence of tools or function calls necessary to generate response to the user utterance. Use the following tools or function calls as required:\n{tools}",
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

        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        prompts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in prompts]
        #texts = [tokenizer.apply_chat_template(convo, add_generation_prompt=True) for convo in convos]
        return {"text": texts, "prompt": prompts}

    tokenizer = AutoTokenizer.from_pretrained(options.model_name)
    full_dataset = ds.load_dataset(options.dataset_name, split="train")  #, token=hf_token)
    dataset = full_dataset.select(range(options.samples))
    dataset = dataset.map(formatting_prompts_func, batched = True)

    model = AutoModelForCausalLM.from_pretrained(options.model_name).to("cuda")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

    scores = []
    for i in range(len(dataset)):
        if not options.quiet:
            print(dataset[i]["prompt"])

        outputs = pipe(dataset[i]["prompt"], max_new_tokens=200, do_sample=True, temperature=1.0)
        output = outputs[0]["generated_text"]
        ideal = dataset[i]["text"][:-1]
        if not options.quiet:
            print("*"*40)
            print(output)
            print("*"*40)
            print(ideal)
            print("*"*40)

        score = scorer.score(ideal, output)["rougeL"].fmeasure
        scores += [score]

    print(sum(scores) / len(scores))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))
