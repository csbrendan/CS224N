import numpy as np
import matplotlib.pyplot as plt

# Average scores for experiment 1
scores_base = {
    "Average ROUGE-1": 0.11033143815827309,
    "Average ROUGE-2": 0.031082459212422264,
    "Average ROUGE-L": 0.0835457530153052,
    "Average Precision": 0.03815396563776175,
    "Average Recall": 0.28732987223622863,
    "Average F1": 0.0648821300914947
}



# Average scores for experiment 2
scores_ft = {
    "Average ROUGE-1": 0.35011973445834654,
    "Average ROUGE-2": 0.12938462044076895,
    "Average ROUGE-L": 0.26643062436313547,
    "Average Precision": 0.39897320813908393,
    "Average Recall": 0.1995087544903754,
    "Average F1": 0.2513314564689761
}

# Extract metric names and scores for both experiments
metrics = list(scores_base.keys())
values_exp1 = list(scores_base.values())
values_exp2 = list(scores_ft.values())

# Set the width of the bars
bar_width = 0.35

# Set the position of the bars on the x-axis
r1 = np.arange(len(metrics))
r2 = [x + bar_width for x in r1]

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(r1, values_exp1, color='skyblue', width=bar_width, edgecolor='grey', label='Base Model')
plt.bar(r2, values_exp2, color='orange', width=bar_width, edgecolor='grey', label='SFT Model')

# Adding labels and title
plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.title('Comparison of Average Scores for Evaluation Metrics')
plt.xticks([r + bar_width/2 for r in range(len(metrics))], metrics, rotation=45, ha='right')  # Set x-axis labels
plt.legend()

# Show plot
plt.tight_layout()
plt.show()
