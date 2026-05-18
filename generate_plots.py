import os
import json
import matplotlib.pyplot as plt
import numpy as np

experiments = {
    "LFM Base (Causal)": "./results_lfm/checkpoint-2634/trainer_state.json",
    "RoBERTa Baseline": "./results_roberta/checkpoint-2634/trainer_state.json",
    "LFM Bypass (Causal Hack)": "./results_lfm_bypass/checkpoint-2000/trainer_state.json",
    "LFM Unmasked (2 Layers)": "./results_lfm_unmasked/checkpoint-2634/trainer_state.json",
    "LFM Unmasked (4 Layers)": "./results_lfm_unmasked_4layers/checkpoint-2634/trainer_state.json",
    "LFM Unmasked (8 Layers + LoRA)": "./results_lfm_unmasked_lora_8layers/checkpoint-2634/trainer_state.json",
    "LFM Unmasked (12 Layers + LoRA Opt)": "./results_lfm_unmasked_lora_optimized/checkpoint-3512/trainer_state.json"
}

test_results_map = {
    "LFM Base (Causal)": "lfm_test_results.json",
    "RoBERTa Baseline": "roberta_test_results.json",
    "LFM Bypass (Causal Hack)": "lfm_bypass_test_results.json",
    "LFM Unmasked (2 Layers)": "lfm_unmasked_test_results.json", # Wait, did it get overwritten? We'll see
    "LFM Unmasked (4 Layers)": "lfm_unmasked_test_results.json", # Wait, let's verify
    "LFM Unmasked (8 Layers + LoRA)": "results_lfm_unmasked_lora_8layers/checkpoint-2634/trainer_state.json", # We can get final eval metrics if test_results missing
    "LFM Unmasked (12 Layers + LoRA Opt)": "lfm_unmasked_lora_optimized_test_results.json"
}

# We will extract history from trainer states
all_data = {}

for name, path in experiments.items():
    if not os.path.exists(path):
        print(f"Warning: {path} not found for {name}")
        continue
        
    with open(path, 'r') as f:
        state = json.load(f)
        
    history = state.get("log_history", [])
    
    train_steps = []
    train_epochs = []
    train_loss = []
    
    eval_epochs = []
    eval_loss = []
    eval_f1 = []
    eval_precision = []
    eval_recall = []
    eval_accuracy = []
    
    for log in history:
        epoch = log.get("epoch", 0)
        if "loss" in log and "eval_loss" not in log:
            # Training log
            train_steps.append(log.get("step", 0))
            train_epochs.append(epoch)
            train_loss.append(log["loss"])
        elif "eval_loss" in log:
            # Evaluation log
            eval_epochs.append(epoch)
            eval_loss.append(log["eval_loss"])
            eval_f1.append(log.get("eval_f1", log.get("eval_f1_score", 0)))
            eval_precision.append(log.get("eval_precision", 0))
            eval_recall.append(log.get("eval_recall", 0))
            eval_accuracy.append(log.get("eval_accuracy", 0))
            
    all_data[name] = {
        "train_epochs": train_epochs,
        "train_loss": train_loss,
        "eval_epochs": eval_epochs,
        "eval_loss": eval_loss,
        "eval_f1": eval_f1,
        "eval_precision": eval_precision,
        "eval_recall": eval_recall,
        "eval_accuracy": eval_accuracy
    }

# Custom style for plot
plt.style.use('seaborn-v0_8-whitegrid')
os.makedirs("plots", exist_ok=True)

def plot_metric(metric_key, title, ylabel, filename, is_train=False):
    plt.figure(figsize=(10, 6))
    
    for name, data in all_data.items():
        epochs = data["train_epochs"] if is_train else data["eval_epochs"]
        values = data[metric_key]
        
        if not values:
            continue
            
        if is_train:
            # Smooth training loss for better visualization
            if len(values) > 10:
                window = 5
                smoothed_values = np.convolve(values, np.ones(window)/window, mode='valid')
                smoothed_epochs = epochs[window-1:]
                plt.plot(smoothed_epochs, smoothed_values, label=name, alpha=0.8, linewidth=2)
            else:
                plt.plot(epochs, values, label=name, alpha=0.8, linewidth=2)
        else:
            plt.plot(epochs, values, marker='o', label=name, alpha=0.9, linewidth=2.5)
            
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(loc='best', frameon=True, facecolor='white', edgecolor='#dddddd')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", filename), dpi=300)
    plt.close()
    print(f"Saved line plot {filename}")

def plot_metric_bar(metric_key, title, ylabel, filename):
    plt.figure(figsize=(12, 7))
    
    names = []
    final_values = []
    
    # Gather final checkpoint data for each experiment
    for name, data in all_data.items():
        values = data[metric_key]
        if values:
            names.append(name)
            final_values.append(values[-1]) # Last checkpoint value
            
    # Sorting by values for clearer comparison
    sorted_pairs = sorted(zip(final_values, names))
    final_values, names = zip(*sorted_pairs)
    
    # Distinct pleasant color map
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(names)))
    
    bars = plt.barh(names, final_values, color=colors, edgecolor='none', height=0.6)
    
    # Add numerical values inside/next to the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2., 
                 f'{width:.4f}', 
                 va='center', ha='left', fontsize=11, fontweight='bold', color='#333333')
                 
    plt.title(title, fontsize=15, fontweight='bold', pad=20)
    plt.xlabel(ylabel, fontsize=12, fontweight='bold')
    plt.xlim(0, max(final_values) * 1.12) # Extend limit to make room for text
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join("plots", filename), dpi=300)
    plt.close()
    print(f"Saved bar plot {filename}")

# Generate plots
plot_metric("train_loss", "Training Loss Comparison", "Loss", "train_loss.png", is_train=True)
plot_metric("eval_loss", "Validation Loss Comparison", "Loss", "eval_loss.png")

# Use Bar Graphs for final F1, Precision and Recall
plot_metric_bar("eval_f1", "Final Validation F1 Score Comparison", "F1 Score", "eval_f1.png")
plot_metric_bar("eval_precision", "Final Validation Precision Comparison", "Precision", "eval_precision.png")
plot_metric_bar("eval_recall", "Final Validation Recall Comparison", "Recall", "eval_recall.png")

print("Plot generation completed successfully.")
