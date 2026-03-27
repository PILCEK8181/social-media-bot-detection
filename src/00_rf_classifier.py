import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix
from utils.save_metrics import save_metrics
import random

import joblib
import matplotlib.pyplot as plt
import seaborn as sns

TEMP_DIR = './temp'
MODELS_DIR = './models'
RESULTS_DIR = './results'
SEED = random.randint(1, 10000)

def main():
    print("=" * 70)
    print("Final Meta-Classifier (Random Forest Stacking)")
    print("=" * 70)
    print("Seed:", SEED)
    print("Loading prediction files...")

    
    # --- 01 RF ----
    df_rf = pd.read_csv(os.path.join(TEMP_DIR, 'predictions/preds_rf.csv'))
    
    # --- 02 LSTM ----
    df_lstm = pd.read_csv(os.path.join(TEMP_DIR, 'predictions/preds_lstm.csv'))

    #---- 03 Roberta ----
    #df_rob = pd.read_csv(os.path.join(TEMP_DIR, 'predictions/preds_roberta_weighted.csv'))
    # df_rob = pd.read_csv(os.path.join(TEMP_DIR, 'predictions/preds_roberta.csv'))
    df_rob = pd.read_csv(os.path.join(TEMP_DIR, 'predictions/preds_roberta_oversample.csv'))

    
    # conver user_id to str and remove 'u' prefix to ensure proper merging
    df_rob['user_id'] = df_rob['user_id'].astype(str).str.replace('u', '', regex=False)
    df_rf['user_id'] = df_rf['user_id'].astype(str).str.replace('u', '', regex=False)
    df_lstm['user_id'] = df_lstm['user_id'].astype(str).str.replace('u', '', regex=False)

    # merge trough user_id
    df = df_rob.merge(df_lstm[['user_id', 'prob_lstm']], on='user_id')
    df = df.merge(df_rf[['user_id', 'prob_rf']], on='user_id')
    
    # val // test
    df_val = df[df['split'] == 'val']
    df_test = df[df['split'] == 'test']
    
    features = ['prob_rf', 'prob_roberta']
    
    X_val = df_val[features]
    y_val = df_val['label']
    
    X_test = df_test[features]
    y_test = df_test['label']
    
    print(f"  Training Meta-Classifier on {len(X_val)} validation samples...")
    print(f"  Evaluating on {len(X_test)} test samples...")
    
    # Train Random Forest as Meta-Classifier
    meta_model = RandomForestClassifier(class_weight='balanced', random_state=SEED)
    meta_model.fit(X_val, y_val)
    
    # Final predictions on the test set
    preds = meta_model.predict(X_test)
    
    # evaluation
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
    
    # Feature importance from Random Forest
    print("\n" + "=" * 70)
    print("MODEL FEATURE IMPORTANCE (Random Forest)")
    print("=" * 70)
    importances = meta_model.feature_importances_
    for name, importance in zip(features, importances):
        print(f"  {name:15}: {importance:.4f}")
    

    # save model for live mode usage
    model_path = os.path.join(MODELS_DIR, 'meta_classifier_rf.pkl')
    joblib.dump(meta_model, model_path)
    print(f"\nMeta-Model saved successfully to {model_path}")

    # feature importance visualization
    importances = meta_model.feature_importances_
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=features, y=importances, palette='viridis')
    plt.title('Random Forest Feature Importances for Meta-Classifier')
    plt.ylabel('Importance')
    plt.xlabel('Expert models')
    
  
    plot_path = os.path.join(RESULTS_DIR, '00model_rf_importances.png')
    plt.savefig(plot_path)
    print(f"Visualization saved to {plot_path}")

    save_metrics(
        filename="00_rf_classifier.py",
        seed=SEED,
        acc=accuracy_score(y_test, preds),
        prec=precision_score(y_test, preds),
        recall=recall_score(y_test, preds),
        f1=f1_score(y_test, preds),
        mcc=matthews_corrcoef(y_test, preds),
        note="Meta-Classifier Random Forest // text,feature - balanced None, weighted roberta"
    )

if __name__ == "__main__":
    main()
