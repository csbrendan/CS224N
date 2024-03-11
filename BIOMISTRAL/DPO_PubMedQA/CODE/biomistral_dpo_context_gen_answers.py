from datasets import load_dataset
import random
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
import numpy as np
import re

# the saved model directory
saved_model_dir = 'data/biomistral-7b-dpo-pqa-context-lora'

tokenizer = AutoTokenizer.from_pretrained(saved_model_dir)

# the quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
)

# load trained model with the quantization configuration
model = AutoModelForCausalLM.from_pretrained(
    saved_model_dir,
    quantization_config=quantization_config,
    device_map="auto",
)

print("######### MODEL & TOKENIZER LOADED ########")
test_dataset = load_dataset("pubmed_qa", "pqa_labeled") #pqa_labeled

# dataset to a DataFrame
df = pd.DataFrame(test_dataset['train'])

df_hold_out = df[-500:]
#df_hold_out = df[-30:]

# print some data
print("########  TOP 5 of DATAFRAME from PUBMEDQA ##########")
print(df_hold_out.head()) 

df_hold_out['messages'] = df_hold_out.apply(lambda row: [
    {"role": "system", "content": "You are a helpful assistant that answers medical questions."},
    {"role": "user", "content": row['question']},
    {"role": "assistant", "content": row['long_answer']}
], axis=1)

test_dataset = df_hold_out[['messages']]  
test_dataset = test_dataset.to_dict(orient='records')


print(f"Total number of examples in test_dataset: {len(test_dataset)}")
for i, sample in enumerate(test_dataset[:5]):
    print(f"Sample {i}: {sample}")

#################

# inference on the selected questions
results = []
for question_data in test_dataset:
    question = question_data['messages'][1]['content']
    actual_answer = question_data['messages'][2]['content']

    # prep messages 
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers medical questions."},
        {"role": "user", "content": question},
    ]

    input_ids = tokenizer.apply_chat_template(messages, truncation=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.2, # less random plez
        top_k=20, #was 50, going with 10-30 to decrase randomness
        top_p=0.95 # hi top_p 95-99 selects words from a smaller part of dist, hence less randomness
    )

    full_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    # extract the assistant's response as we dont want to muddy the json
    # look for the last occurrence of the question in the full_text and extract everything that follows
    pattern = re.escape(question)  # escape the question to ensure it's treated as literal text in the regex
    match = re.search(pattern, full_text)
    if match:
        start_index = match.end()  # get the end index of the match
        predicted_answer = full_text[start_index:].strip()  # extract txt following the question

        # split into sections / tags
        sections = re.split(r"\n?<\|(system|user|assistant)\|>\n?", predicted_answer)

        # keep assistants txt
        if len(sections) >= 3:
            assistant_text = sections[2]
            predicted_answer = assistant_text
        else:
            print("Warning: Unexpected format in predicted_answer")
    else:
        print("Falling back to full answer text since ")
        predicted_answer = full_text  # if the question isnt found

    result = {
        "question": question,
        "predicted_answer": predicted_answer,
        "actual_answer": actual_answer
    }
    results.append(result)


with open("dpo_inference_pqa_context_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Inference results saved to dpo_inference_pqa_context_results.json")


