import logging
import os
import torch
import yaml
from ludwig.api import LudwigModel
import getpass
#import locale; locale.getpreferredencoding = lambda: "UTF-8"
import locale
import numpy as np; np.random.seed(123)
import pandas as pd
from datasets import load_dataset

'''
#maybe need this only if OOM
def clear_cache():
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
'''
encoding = locale.getpreferredencoding()
encoding = "utf-8"

# HF tokey
os.environ["HF_AUTH_TOKEN"] = "HF_TOKEN"
os.environ["HUGGING_FACE_HUB_TOKEN"] = "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" #if you need clear token w.o. restart session empty this field
assert os.environ["HUGGING_FACE_HUB_TOKEN"]

#df = pd.read_json("https://raw.githubusercontent.com/sahil280114/codealpaca/master/data/code_alpaca_20k.json")

#im using pubmed_qa instead:
dataset = load_dataset("pubmed_qa", "pqa_labeled")
df = pd.DataFrame(dataset)

# Selecting the columns you need
#df = df[['pubid', 'question', 'context', 'long_answer', 'final_decision']]

df = pd.json_normalize(df['train'])

# If you want to concatenate contexts into a single string (optional)
df['context'] = df['context.contexts'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')

df['context.reasoning_required_pred'] = df['context.reasoning_required_pred'].apply(lambda x: ''.join(x) if isinstance(x, list) else x)
df['context.reasoning_free_pred'] = df['context.reasoning_free_pred'].apply(lambda x: ''.join(x) if isinstance(x, list) else x)

print(df['context'].head())

print(df.columns)  # Now, this should print the actual column names from the dictionaries
print(df.head())  # And this will print the top 5 rows with the unpacked columns


for index in range(5):  # Adjust the range for the number of examples you want to see
    print(f"Example {index}:")
    for column in df.columns:
        print(f"{column}: {df.loc[index, column]}")
    print("\n" + "-"*50 + "\n")

# IF U JUST WANT TO SEE DATA UNCOMMENT
#import sys
#sys.exit()


# Calculate the number of rows for each split
total_rows = len(df)
split_0_count = int(total_rows * 0.9)  # 90% for training
split_1_count = int(total_rows * 0.05)  # 5% for validation
split_2_count = total_rows - split_0_count - split_1_count  # Remaining 5% for testing

# Create an array with split values based on the counts
split_values = np.concatenate([
    np.zeros(split_0_count),  # Training
    np.ones(split_1_count),  # Validation
    np.full(split_2_count, 2)  # Testing
])

# Shuffle the array to ensure randomness
np.random.shuffle(split_values)

# Add the 'split' column to the DataFrame
df['split'] = split_values
df['split'] = df['split'].astype(int)

# Limit the DataFrame to the first 1000 rows
df = df.head(n=1000)

print(f"Total number of examples in the dataset: {df.shape[0]}")


#pd.set_option('display.max_colwidth', None)
print(df[['question', 'long_answer']].head(2))


#######################
#
# FINE TUNE MEDITRON
#
######################



qlora_fine_tuning_config = yaml.safe_load(
"""
model_type: llm
base_model: epfl-llm/meditron-7b

input_features:
  - name: question
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
  split:
    type: random
    probabilities:
    - 0.8
    - 0.1
    - 0.1

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
)



model = LudwigModel(config=qlora_fine_tuning_config, logging_level=logging.INFO)
results = model.train(dataset=df)
model.save("./medi_ft_model")

###############################
#
# MAKE SOME PREDICTIONS
#
##############################



test_examples = pd.DataFrame([
      {
            "question": "Create an array of length 5 which contains all even numbers between 1 and 10.",
            "context": "this is a linear algebra question",
      },
      {
            "question": "What is the capital of France.",
            "context": "France, in Western Europe, encompasses medieval cities, alpine villages and Mediterranean beaches.",
      },
      {
            "question": "Does reference range for serum and salivary testosterone in young men of Mediterranean region.",
            "context": "The interassay variability found in the measurement of testosterone (T) levels warrants the need for laboratories to validate their methods to establish trustworthy cut-off points for diagnosis of male hypogonadism. The aims of this study were to validate measurement of total T (TT) at our laboratory in order to obtain reference ranges for TT, calculated free T (CFT), calculated bioavailable T (CBT), and salivary T (ST) in healthy young men from the Mediterranean region, and to evaluate the potential clinical value of ST by establishing its correlation with serum T. An observational, cross-sectional study with sequential sampling. Men aged 18-30 years with body mass index (BMI)<30. Chronic diseases, hepatic insufficiency or use of drugs altering circulating T levels. Main outcome measures TT (chemiluminescent immunoassay UniCell DXI 800 [Access T Beckman Coulter]), CFT and CBT (Vermeulen's formula), and ST (radioimmunoassay for serum TT modified for saliva [Coat-A-Count, Siemens]). Descriptive statistical analyses and correlation by Spearman's rho (SPSS 19.0 Inc., Chicago) were used. One hundred and twenty-one subjects aged 24±3.6 years with BMI 24±2.5 kg/m(2) were enrolled. Hormone study: TT, 19±5.5 nmol/L (reference range [rr.] 9.7-33.3); CFT, 0.38 nmol/L (rr. 0.22-0.79); CBT, 9.7 nmol/L (rr. 4.9-19.2); and ST, 0.35 nmol/L (rr. 0.19-0.68). Correlation between ST and CFT was 0.46.",
      },
])


# PREDICT !
predictions = model.predict(test_examples)[0]

try:
    print(predictions.keys())
    print("long_answer_predictions:", predictions['long_answer_predictions'].iloc[0])
    print("long_answer_probabilities:", predictions['long_answer_probabilities'].iloc[0])
    print("long_answer_response:", predictions['long_answer_response'].iloc[0])
    print("long_answer_probability:", predictions['long_answer_probability'].iloc[0])
except:
    pass


for input_with_prediction in zip(test_examples['question'], test_examples['context'], predictions['long_answer_response']):
    print(f"Question: {input_with_prediction[0]}")
    #print(f"Context: {input_with_prediction[1]}")
    print(f"Generated Long Answer: {input_with_prediction[2][0]}")  # Assuming predictions are lists and you want the first element
    print("\n\n")

