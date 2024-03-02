from rouge_score import rouge_scorer
import json

def compute_metrics(predictions):

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    results = []

    for entry in predictions:
        question = entry["Question"]
        generated_answer = entry["Generated Long Answer"]
        actual_answer = entry["Actual Long Answer"]

        rouge_scores = scorer.score(generated_answer, actual_answer)

        # Compute precision, recall, and F1
        true_positives = len(set(generated_answer.split()) & set(actual_answer.split()))
        if len(generated_answer.split()) == 0:
            precision = 0.0
        else:
            precision = true_positives / len(generated_answer.split())
        if len(actual_answer.split()) == 0:
            recall = 0.0
        else:
            recall = true_positives / len(actual_answer.split())
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        result = {
            "Question": question,
            "Generated Long Answer": generated_answer,
            "Actual Long Answer": actual_answer,
            "ROUGE-1": rouge_scores["rouge1"].fmeasure,
            "ROUGE-2": rouge_scores["rouge2"].fmeasure,
            "ROUGE-L": rouge_scores["rougeL"].fmeasure,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        }
        results.append(result)
    return results

# Load predictions from JSON file
with open("predictions_from_base_model.json", "r") as file:
    predictions = json.load(file)

# Compute metrics
metrics = compute_metrics(predictions)

# Save results to a JSON file
with open('evaluation_ft_scores.json', 'w') as f:
    json.dump(metrics, f, indent=4)
