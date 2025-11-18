import matplotlib.pyplot as plt
import numpy as np
import os

def plot_train_test_metrics(results, out_dir):
    """
    results = {
        'DecisionTree': {
            'train_accuracy': ...,
            'test_accuracy': ...,
            'train_f1': ...,
            'test_f1': ...
        },
        'RandomForest': { ... }
    }
    """
    models = list(results.keys())

    train_acc = [results[m]["train_accuracy"] for m in models]
    test_acc  = [results[m]["test_accuracy"]  for m in models]

    train_f1 = [results[m]["train_f1"] for m in models]
    test_f1  = [results[m]["test_f1"]  for m in models]

    plt.figure(figsize=(10, 6))

    # Accuracy
    plt.plot(models, train_acc, marker="o", label="Train Accuracy", linewidth=3)
    plt.plot(models, test_acc, marker="o", label="Test Accuracy", linewidth=3)

    # F1
    plt.plot(models, train_f1, marker="s", label="Train Macro F1", linewidth=3)
    plt.plot(models, test_f1, marker="s", label="Test Macro F1", linewidth=3)

    plt.title("Baseline Model Train vs Test Performance (Clip-level)", fontsize=16)
    plt.ylabel("Score")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=12)

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "train_test_baseline_plot.png"), dpi=200, bbox_inches='tight')
    plt.close()
