import pandas as pd
from ludwig.api import LudwigModel
import json
from datasets import load_dataset
import numpy as np

# Load your dataset
# If your dataset is in a JSON or CSV file:
# df = pd.read_json('path_to_your_dataset.json')
# OR for CSV: df = pd.read_csv('path_to_your_dataset.csv')

# If you're using a dataset from Hugging Face or similar:
dataset = load_dataset("pubmed_qa", "pqa_labeled")
df = pd.json_normalize(dataset['train'])  # Adjust as needed based on your dataset structure

# Create a DataFrame with 'question' and 'context' columns
#test_examples = df[['question', 'context.contexts']].copy()
#test_examples['context'] = test_examples['context.contexts'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
#test_examples = test_examples.drop(columns=['context.contexts'])

# Preprocess context columns
df['context'] = df['context.contexts'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
df['context.reasoning_required_pred'] = df['context.reasoning_required_pred'].apply(lambda x: ''.join(x) if isinstance(x, list) else x)
df['context.reasoning_free_pred'] = df['context.reasoning_free_pred'].apply(lambda x: ''.join(x) if isinstance(x, list) else x)

# Create a test split manually (10% of the data)  COMMENT OUT TO  TEST
#test_df = df.sample(frac=0.1, random_state=42)  # Adjust frac as needed for the percentage

test_df = df.head(n=50)

# Load the model
model = LudwigModel.load("./medi_ft_model")

# Make predictions
predictions = model.predict(test_df[['question', 'context']])[0]

# Collect predictions and actual answers
results = []
for input_with_prediction in zip(test_df['question'], test_df['context'], predictions['long_answer_response'], test_df['long_answer']):
    results.append({
        "Question": input_with_prediction[0],
        "Context": input_with_prediction[1],
        "Generated Long Answer": input_with_prediction[2][0],  # Assuming predictions are lists and you want the first element
        "Actual Long Answer": input_with_prediction[3]  # Append the actual long answer from the dataset
    })



# Save results to a JSON file
with open('predictions_from_saved_model.json', 'w') as f:
    json.dump(results, f, indent=4)

