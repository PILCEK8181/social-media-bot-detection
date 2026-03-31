
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix, roc_curve
from utils.save_metrics import save_metrics
import random

import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from utils.evaluation import (
    calculate_confidence_intervals,
    perform_mcnemar_test,
    calculate_and_plot_eer,
)

TEMP_DIR = './temp'
MODELS_DIR = './models'
RESULTS_DIR = './results'
SEED = random.randint(1, 10000)

def main():
    
    print("=" * 70)
    print("Final Meta-Classifier (Logistic Regression Stacking)")
    print("=" * 70)
    print("Seed:", SEED)
    print("Loading prediction files...")

    
    # --- 01 RF ----
    df_rf = pd.read_csv(os.path.join(TEMP_DIR, 'predictions/preds_rf.csv'))
    
    # --- 02 LSTM ----
    df_lstm = pd.read_csv(os.path.join(TEMP_DIR, 'predictions/preds_lstm.csv'))

    #---- 03 Roberta ----
    # df_rob = pd.read_csv(os.path.join(TEMP_DIR, 'predictions/preds_roberta_weighted.csv'))
    # df_rob = pd.read_csv(os.path.join(TEMP_DIR, 'predictions/preds_roberta.csv'))
    df_rob = pd.read_csv(os.path.join(TEMP_DIR, 'predictions/preds_roberta_oversample.csv'))

    
    # convert user_id to str and remove 'u' prefix to ensure proper merging
    df_rob['user_id'] = df_rob['user_id'].astype(str).str.replace('u', '', regex=False)
    df_rf['user_id'] = df_rf['user_id'].astype(str).str.replace('u', '', regex=False)
    df_lstm['user_id'] = df_lstm['user_id'].astype(str).str.replace('u', '', regex=False)

    # merge trough user_id
    df = df_rob.merge(df_lstm[['user_id', 'prob_lstm']], on='user_id')
    df = df.merge(df_rf[['user_id', 'prob_rf']], on='user_id')
    
    # val // test
    df_val = df[df['split'] == 'val']
    df_test = df[df['split'] == 'test']
    
    # FINAL FEATURES going into Meta-Classifier
    # use only RF and Roberta probabilities as features for the meta-classifier, since LSTM performed poorly and may add noise
    # its possible to change those features or include LSTM
    features = ['prob_rf', 'prob_roberta']
    
    X_val = df_val[features]
    y_val = df_val['label']
    
    X_test = df_test[features]
    y_test = df_test['label']
    
    print(f"  Training Meta-Classifier on {len(X_val)} validation samples...")
    print(f"  Evaluating on {len(X_test)} test samples...")
    
    # Train Logistic Regression as Meta-Classifier
    meta_model = LogisticRegression(class_weight=None, random_state=SEED)
    meta_model.fit(X_val, y_val)
    
    # Final predictions on the test set
    preds = meta_model.predict(X_test)
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL ENSEMBLE RESULTS (Test Set)")
    print("=" * 70)
    print(f"Accuracy:  {accuracy_score(y_test, preds):.4f}")
    print(f"F1 Score:  {f1_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test, preds):.4f}")
    print(f"Recall:    {recall_score(y_test, preds):.4f}")
    print(f"MCC:       {matthews_corrcoef(y_test, preds):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))
    
    # weights of the meta-classifier (feature importance)
    print("\n" + "=" * 70)
    print("MODEL FEATURE IMPORTANCE (Logistic Regression Weights)")
    print("=" * 70)
    weights = meta_model.coef_[0]
    for name, weight in zip(features, weights):
        print(f"  {name:15}: {weight:.4f}")
    
    # save model for live mode usage
    model_path = os.path.join(MODELS_DIR, 'meta_classifier_lr.pkl')
    joblib.dump(meta_model, model_path)
    print(f"\nMeta-Model saved successfully to {model_path}")

    # weight importance visualization
    weights = meta_model.coef_[0]
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=features, y=weights, palette='viridis')
    plt.title('LR weights for Meta-Classifier Features')
    plt.ylabel('weight')
    plt.xlabel('Expert models')
    
    plot_path = os.path.join(RESULTS_DIR, '00model_weights.png')
    plt.savefig(plot_path)
    print(f"Visualization saved to {plot_path}")

    save_metrics(
        filename="00_meta_classifier.py",
        seed=SEED,
        acc=accuracy_score(y_test, preds),
        prec=precision_score(y_test, preds),
        recall=recall_score(y_test, preds),
        f1=f1_score(y_test, preds),
        mcc=matthews_corrcoef(y_test, preds),
        note="final - top 15 features, NONE class weights, oversampled roberta"
    )

    print("\n" + "=" * 70)
    print("STARTING THESIS EVALUATION SUITE")
    print("=" * 70)
    
    probs_ensemble = meta_model.predict_proba(X_test)[:, 1]
    preds_rf_binary = (np.array(X_test['prob_rf']) >= 0.5).astype(int)
    hist_path = os.path.join(RESULTS_DIR, '01_prob_histogram.png')
    
    # 1. Bootstrapping
    calculate_confidence_intervals(y_test, preds)
    
    # 2. McNemar's Test
    perform_mcnemar_test(y_test, preds, preds_rf_binary)
    
    # 3. EER & Histogram
    calculate_and_plot_eer(y_test, probs_ensemble, save_path=hist_path)
    
    # 4. Robustness TODO
    
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()