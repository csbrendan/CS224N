import json
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer
import pandas as pd
from peft import PeftConfig
from peft import PeftModel
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
import random
import torch
from transformers import AutoTokenizer
import re
from multiprocessing import cpu_count
from trl import DPOTrainer
from peft import LoraConfig
from transformers import TrainingArguments

# MY DATASET - load my dpo dataset containing chosen/rejected (actual_answer/predicted_answer)
with open('sft_pqa_context_inference_results.json', 'r') as file:
    json_data = json.load(file)
df = pd.DataFrame(json_data)
# map fields to match the DPO format
df_mapped = df.rename(columns={'actual_answer': 'chosen', 'predicted_answer': 'rejected'})

dataset = Dataset.from_pandas(df_mapped)  # convert to Hug Face Dataset

train_test_split = dataset.train_test_split(test_size=0.1)
dataset_dict = DatasetDict(train=train_test_split['train'], test=train_test_split['test'])
raw_datasets = DatasetDict(dataset_dict)

example = raw_datasets["train"][0]
print(example.keys())
print(example["question"])
print(example["rejected"])
print(example["chosen"])

#exit()

## tokenizer
model_id = 'data/biomistral-7b-sft-pqa-context-lora'

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
# trunc: left to ensure i dont lose labels in final turn
tokenizer.truncation_side = "left"

# default for models without max length
if tokenizer.model_max_length > 100_000:
    tokenizer.model_max_length = 2048

DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

def apply_chat_template(example, tokenizer, assistant_prefix="<|assistant|>\n"):
    def _strip_prefix(s, pattern):
        return re.sub(f"^{re.escape(pattern)}", "", s)

    prompt_messages = [{"role": "user", "content": example["question"]}]
    prompt_messages.insert(0, {"role": "system", "content": "You are a helpful assistant that answers medical questions."})
    chosen_messages = [{"role": "assistant", "content": example["chosen"]}]
    rejected_messages = [{"role": "assistant", "content": example["rejected"]}]

    example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
    example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
    example["text_prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    example["text_chosen"] = _strip_prefix(example["text_chosen"], assistant_prefix)
    example["text_rejected"] = _strip_prefix(example["text_rejected"], assistant_prefix)

    return example


column_names = list(raw_datasets["train"].features)

raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=cpu_count(),
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
)

# change column names to what TRL expects, text_chosen -> chosen and text_rejected -> rejected
for split in ["train", "test"]:
    raw_datasets[split] = raw_datasets[split].rename_columns(
        {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
    )

# few random samples from training set:
for index in random.sample(range(len(raw_datasets["train"])), 3):
    print(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
    print(f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['chosen']}")
    print(f"Rejected sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['rejected']}")


# Load SFT model
peft_config = PeftConfig.from_pretrained(model_id)
print("Adapter weights model repo:", model_id)
print("Base model weights model repo:", peft_config.base_model_name_or_path)

# specify how to quantize 
quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
)
device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None

# load the base biomistral in 4-bit
model_kwargs = dict(
    torch_dtype="auto",
    use_cache=False,  # false since we use grad checkpointing
    device_map=device_map,
    quantization_config=quantization_config,
)
base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, **model_kwargs)

## load base model + SFT adapter weights
model = PeftModel.from_pretrained(base_model, model_id)

####### Define DPOTrainer
output_dir = 'data/biomistral-7b-dpo-pqa-context-lora'

# based on config
training_args = TrainingArguments(
    bf16=True,
    do_eval=True,
    evaluation_strategy="steps",
    eval_steps=100,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant":False},
    learning_rate=5.0e-6,
    log_level="info",
    logging_steps=10,
    lr_scheduler_type="cosine",
    num_train_epochs=1,
    optim="paged_adamw_32bit",
    output_dir=output_dir,  
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=1,
    seed=42,
    warmup_ratio=0.1,
)

peft_config = LoraConfig(
        r=128,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",  "up_proj",  "down_proj"],
)

trainer = DPOTrainer(
        model,
        ref_model=None,
        model_init_kwargs=None,
        ref_model_init_kwargs=None,
        args=training_args,
        beta=0.01,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"],
        tokenizer=tokenizer,
        max_length=1024,
        max_prompt_length=512,
        peft_config=peft_config,
        loss_type="sigmoid",
    )

##### TRAIN
train_result = trainer.train()

metrics = train_result.metrics
max_train_samples = len(raw_datasets["train"])
metrics["train_samples"] = min(max_train_samples, len(raw_datasets["train"]))
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

trainer.save_state()

### SAVING the model
output_dir = 'data/biomistral-7b-dpo-pqa-context-lora'
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
