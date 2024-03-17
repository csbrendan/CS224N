from rouge_score import rouge_scorer
import json

def truncate_to_length(text, length):
    words = text.split()[:length]
    return ' '.join(words)

def compute_metrics(predictions):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    results = []
    total_scores = {'rouge1': {'f1': 0, 'precision': 0, 'recall': 0}, 
                    'rouge2': {'f1': 0, 'precision': 0, 'recall': 0}, 
                    'rougeL': {'f1': 0, 'precision': 0, 'recall': 0}}
    valid_entries = 0

    for entry in predictions:
        if "question" not in entry or "predicted_answer" not in entry or "actual_answer" not in entry:
            continue

        valid_entries += 1

        actual_answer = entry["actual_answer"]
        predicted_answer = entry["predicted_answer"]
        rouge_scores = scorer.score(predicted_answer, actual_answer)

        for key in total_scores.keys():
            total_scores[key]['f1'] += rouge_scores[key].fmeasure
            total_scores[key]['precision'] += rouge_scores[key].precision
            total_scores[key]['recall'] += rouge_scores[key].recall

    avg_scores = {}
    for key, value in total_scores.items():
        avg_scores[key] = {metric: val / valid_entries for metric, val in value.items()}

    return avg_scores

# Load predictions from JSON file
with open("dpo_inference_sum_results.json", "r") as file:
    predictions = json.load(file)

# Compute metrics
average_scores = compute_metrics(predictions)

# Output average scores
print("Average Scores:")
for rouge_key, metrics in average_scores.items():
    print(f"\n{rouge_key}:")
    for metric, score in metrics.items():
        print(f"{metric.capitalize()}: {score:.4f}")

# Optionally, save average results to a JSON file
with open('dpo_sum_metrics.json', 'w') as f:
    json.dump(average_scores, f, indent=4)


#with open("dpo_pqa_context_benchmark.json", "r") as file:
#with open('dpo_metrics.json', 'w') as f:
  