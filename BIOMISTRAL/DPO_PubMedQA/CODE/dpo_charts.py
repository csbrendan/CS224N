import json
import matplotlib.pyplot as plt

file_path = '/Users/brendanmurphy/Desktop/CS224N/PROJECT/CODE_AZURE/biomistral-7b-dpo-pqa-context-lora/biomistral-7b-dpo-pqa-context-lora/trainer_state.json'

with open(file_path, 'r') as f:
    data = json.load(f)

log_history = data['log_history']

loss_values = [entry['loss'] for entry in log_history if 'loss' in entry]
epochs = [entry['epoch'] for entry in log_history if 'loss' in entry]

plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_values, linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('DPO Training Loss')
plt.legend()
#plt.grid(True)
plt.grid(axis='y')
plt.show()

reward_values = [entry['rewards/margins'] for entry in log_history if 'rewards/margins' in entry]
epochs = [entry['epoch'] for entry in log_history if 'rewards/margins' in entry]

plt.figure(figsize=(10, 6))
plt.plot(epochs, reward_values, linestyle='-', color='r')
plt.xlabel('Epoch')
plt.ylabel('Rewards/Margins')
plt.title('DPO Rewards/Margins')
plt.legend()
#plt.grid(True)
plt.grid(axis='y')
plt.show()