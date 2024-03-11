from datasets import load_dataset
import pandas as pd
import numpy as np

# Load the PubMedQA dataset
dataset = load_dataset("pubmed_qa", "pqa_artificial")
print("Length of the 'train' split:", len(dataset['train']))

df = pd.DataFrame(dataset['train'])

df = df.head(n=5000)
print("**** LEN OF DF SHOULD BE 5000")
print(len(df))

# append context to beginning of Question
df['question'] = df.apply(lambda row: ' '.join(row['context']['contexts']) + ' ' + row['question'], axis=1)

total_rows = len(df)
split_0_count = int(total_rows * 0.9)  # 90% for training
split_1_count = int(total_rows * 0.05)  # 5% for validation
split_2_count = total_rows - split_0_count - split_1_count  # Remaining 5% for testing

split_values = np.concatenate([
    np.zeros(split_0_count),  # training
    np.ones(split_1_count),  # validation
    np.full(split_2_count, 2)  # testing
])

df['split'] = split_values
df['split'] = df['split'].astype(int)

# change to fit the format used by apply_chat_template()
df['messages'] = df.apply(lambda row: [
    {"role": "system", "content": "You are a helpful assistant that answers medical questions."},
    {"role": "user", "content": row['question']},
    {"role": "assistant", "content": row['long_answer']}
], axis=1)

train_dataset = df[df['split'] == 0][['messages']]
eval_dataset = df[df['split'] == 1][['messages']]
test_dataset = df[df['split'] == 2][['messages']]

# Limit the training set, so to save half for test set in generation script
#print("Length of train_dataset before slicing:", len(train_dataset))
#train_dataset = train_dataset[:500]
#print("Length of train_dataset after slicing:", len(train_dataset))

train_dataset = train_dataset.to_dict(orient='records')
eval_dataset = eval_dataset.to_dict(orient='records')
test_dataset = test_dataset.to_dict(orient='records')

raw_datasets = {
    "train": train_dataset,
    "validation": eval_dataset,
    "test": test_dataset
}

from transformers import AutoTokenizer

model_id = "BioMistral/BioMistral-7B"

tokenizer = AutoTokenizer.from_pretrained(model_id)

# set pad_token_id equal to the eos_token_id if not set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# set reasonable default for models without max length
if tokenizer.model_max_length > 100_000:
    tokenizer.model_max_length = 2048

DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

import re
import random
from multiprocessing import cpu_count

def apply_chat_template(example, tokenizer):
    messages = example["messages"]
    example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)
    return example

column_names = list(raw_datasets["train"][0].keys())
raw_datasets = {
    split: [apply_chat_template(example, tokenizer) for example in raw_datasets[split]]
    for split in raw_datasets
}

# create the splits
train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["validation"]

print("################ LEN TRAIN_DATASET #################")
print(len(train_dataset))

for index in random.sample(range(len(train_dataset)), 3):
    print(f"Sample {index} of the processed training set:\n\n{train_dataset[index]['text']}")

from transformers import BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
)
device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None

model_kwargs = dict(
    torch_dtype="auto",
    use_cache=False,
    device_map=device_map,
    quantization_config=quantization_config,
)

from trl import SFTTrainer
from peft import LoraConfig
from transformers import TrainingArguments

output_dir = 'data/biomistral-7b-sft-pqa-context-lora'

training_args = TrainingArguments(
    fp16=True,
    do_eval=True,
    evaluation_strategy="epoch",
    gradient_accumulation_steps=128,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=2.0e-05,
    log_level="info",
    logging_steps=5,
    logging_strategy="steps",
    lr_scheduler_type="cosine",
    max_steps=-1,
    num_train_epochs=1,
    output_dir=output_dir,
    overwrite_output_dir=True,
    per_device_eval_batch_size=8,
    per_device_train_batch_size=4,
    save_strategy="no",
    save_total_limit=None,
    seed=42,
)

peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

trainer = SFTTrainer(
    model=model_id,
    model_init_kwargs=model_kwargs,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    tokenizer=tokenizer,
    packing=True,
    peft_config=peft_config,
    max_seq_length=tokenizer.model_max_length,
)

train_result = trainer.train()

# Save the trained model
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

metrics = train_result.metrics
train_samples = len(train_dataset)
metrics["train_samples"] = train_samples
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()


