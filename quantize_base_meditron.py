from ludwig.api import LudwigModel
import yaml
from datasets import load_dataset
import pandas as pd
import numpy as np; np.random.seed(123)
import os

# HF tokey
os.environ["HF_AUTH_TOKEN"] = "HF_TOKEN"
os.environ["HUGGING_FACE_HUB_TOKEN"] = "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" #if you need clear token w.o. restart session empty this field
assert os.environ["HUGGING_FACE_HUB_TOKEN"]

# we only need 1 example for dummy finetune
dataset = load_dataset("pubmed_qa", "pqa_labeled")
df = pd.DataFrame(dataset)

df = pd.json_normalize(df['train'])

# If you want to concatenate contexts into a single string (optional)
df['context'] = df['context.contexts'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
df['context.reasoning_required_pred'] = df['context.reasoning_required_pred'].apply(lambda x: ''.join(x) if isinstance(x, list) else x)
df['context.reasoning_free_pred'] = df['context.reasoning_free_pred'].apply(lambda x: ''.join(x) if isinstance(x, list) else x)

# Calculate the number of rows for each split
total_rows = len(df)
split_0_count = int(total_rows * 0.9)  # 90% for training
split_1_count = int(total_rows * 0.05)  # 5% for validation
split_2_count = total_rows - split_0_count - split_1_count  # Remaining 5% for testing

split_values = np.concatenate([
    np.zeros(split_0_count),  # Training
    np.ones(split_1_count),  # Validation
    np.full(split_2_count, 2)  # Testing
])

np.random.shuffle(split_values)

df['split'] = split_values
df['split'] = df['split'].astype(int)

# Limit the DataFrame to the first 1000 rows
df = df.head(n=3)



# Your model configuration, focusing on the quantization and adapter settings
config_str = """
model_type: llm
base_model: epfl-llm/meditron-7b

input_features:
  - name: combined_input  # Assuming 'combined_input' combines question and context
    type: text

output_features:
  - name: long_answer
    type: text

prompt:
  template: >-
    Below is a medical question that requires an informed answer. Use the provided context to generate a detailed and relevant response.

    ### Context: {context}

    ### Question: {question}

    ### Detailed Answer:

generation:
  temperature: 0.1
  max_new_tokens: 512

adapter:
  type: lora

quantization:
  bits: 4

preprocessing:
  global_max_sequence_length: 512

trainer:
  type: finetune
  epochs: 1
  batch_size: 1
  eval_batch_size: 2
  gradient_accumulation_steps: 16
  learning_rate: 0.0004
  learning_rate_scheduler:
    warmup_fraction: 0.03
"""

# Load the configuration
qlora_config = yaml.safe_load(config_str)

# Initialize the model with the configuration
model = LudwigModel(qlora_config)

#dummy train
model.train(dataset=df)

# Save the quantized model
model.save("./quantized_baseline_model")
