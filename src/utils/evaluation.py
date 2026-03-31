

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, roc_curve
from statsmodels.stats.contingency_tables import mcnemar
from scipy.optimize import brentq
from scipy.interpolate import interp1d


# Calculate confidence intervals using bootstrapping
def calculate_confidence_intervals(y_true, y_pred, n_iterations=1000, seed=42):
    np.random.seed(seed)
    n_size = len(y_true)
    y_true_arr = np.array(y_true)
    preds_arr = np.array(y_pred)
    
    accuracies, f1_scores = [], []
    
    for _ in range(n_iterations):
        indices = np.random.randint(0, n_size, n_size)
        y_test_boot = y_true_arr[indices]
        preds_boot = preds_arr[indices]
        
        accuracies.append(accuracy_score(y_test_boot, preds_boot))
        f1_scores.append(f1_score(y_test_boot, preds_boot))
        
    acc_ci = (np.percentile(accuracies, 2.5), np.percentile(accuracies, 97.5))
    f1_ci = (np.percentile(f1_scores, 2.5), np.percentile(f1_scores, 97.5))
    
    print("\n[1] BOOTSTRAPPING (95% Confidence Intervals)")
    print(f"Accuracy: {np.mean(accuracies):.4f} [{acc_ci[0]:.4f}, {acc_ci[1]:.4f}]")
    print(f"F1-Score: {np.mean(f1_scores):.4f} [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]")
    
    return acc_ci, f1_ci

# Perform McNemars test for mathematical proof of significant difference between two classifiers
def perform_mcnemar_test(y_true, preds_ensemble, preds_baseline):

    n_size = len(y_true)
    y_true_arr = np.array(y_true)
    table = [[0, 0], [0, 0]]
    
    for i in range(n_size):
        base_correct = (preds_baseline[i] == y_true_arr[i])
        ens_correct = (preds_ensemble[i] == y_true_arr[i])
        
        if base_correct and ens_correct: table[0][0] += 1
        elif base_correct and not ens_correct: table[0][1] += 1
        elif not base_correct and ens_correct: table[1][0] += 1
        else: table[1][1] += 1

    result = mcnemar(table, exact=False, correction=True)
    
    print("\n[2] MCNEMAR'S TEST (Baseline vs. Ensemble)")
    print(f"Statistic: {result.statistic:.4f} | P-value: {result.pvalue:.4e}")
    return result.pvalue

# calculate EER and plot histogram of predicted probabilities for both classes
def calculate_and_plot_eer(y_true, y_probs, save_path=None):

    # 1. Calculate EER
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    
    print("\n[3] EQUAL ERROR RATE (EER)")
    print(f"EER: {eer:.4f} ({eer*100:.2f}%)")
    
    # 2. Plot Histogram
    if save_path:
        plt.figure(figsize=(10, 6))
        sns.histplot(y_probs[np.array(y_true) == 0], color='blue', label='Humans (True 0)', kde=True, stat="density", linewidth=0, alpha=0.5, bins=50)
        sns.histplot(y_probs[np.array(y_true) == 1], color='red', label='Bots (True 1)', kde=True, stat="density", linewidth=0, alpha=0.5, bins=50)
        plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Boundary (0.5)')
        plt.title('Prediction Probability Distribution (Ensemble)')
        plt.xlabel('Predicted Probability of being a Bot')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n[4] VISUALIZATION: Histogram saved to {save_path}")
        plt.close()
        
    return eer

