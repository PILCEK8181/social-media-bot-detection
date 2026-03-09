
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix, classification_report

TEMP_DIR = './temp'
MODELS_DIR = './models'
RESULTS_DIR = './results'
SEED = 16

def main():
    print("=" * 70)
    print("Final Meta-Classifier (Logistic Regression Stacking)")
    print("=" * 70)
    
    print("Seed:", SEED)

    print("Loading prediction files...")

    
    # --- 01 RF ----
    df_rf = pd.read_csv(os.path.join(TEMP_DIR, 'preds_rf.csv'))
    
    # --- 02 LSTM ----
    df_lstm = pd.read_csv(os.path.join(TEMP_DIR, 'preds_lstm.csv'))

    #---- 03 Roberta ----
    #df_rob = pd.read_csv(os.path.join(TEMP_DIR, 'preds_roberta_weighted.csv'))
    #df_rob = pd.read_csv(os.path.join(TEMP_DIR, 'preds_roberta.csv'))
    df_rob = pd.read_csv(os.path.join(TEMP_DIR, 'preds_roberta_oversample.csv'))

    
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
    
    # FINAL FEATURES going into Meta-Classifier
    features = ['prob_roberta', 'prob_rf']
    
    X_val = df_val[features]
    y_val = df_val['label']
    
    X_test = df_test[features]
    y_test = df_test['label']
    
    print(f"  Training Meta-Classifier on {len(X_val)} validation samples...")
    print(f"  Evaluating on {len(X_test)} test samples...")
    
    # Train Logistic Regression as Meta-Classifier, using balanced class weights to handle any imbalance in the validation set
    meta_model = LogisticRegression(class_weight='balanced', random_state=SEED)
    meta_model.fit(X_val, y_val)
    
    # Final predictions on the test set
    preds = meta_model.predict(X_test)
    
    #EVAL
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
    

    import joblib
    import matplotlib.pyplot as plt
    import seaborn as sns


    # save model for live mode usage
    model_path = os.path.join(MODELS_DIR, '00XDmeta_classifier_lr.pkl')
    joblib.dump(meta_model, model_path)
    print(f"\nMeta-Model saved successfully to {model_path}")

    # weight importance visualization
    weights = meta_model.coef_[0]
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=features, y=weights, palette='viridis')
    plt.title('LR weights for Meta-Classifier Features')
    plt.ylabel('weight')
    plt.xlabel('Expert models')
    
  
    plot_path = os.path.join(RESULTS_DIR, '00_XDmodel_weights.png')
    plt.savefig(plot_path)
    print(f"Visualization saved to {plot_path}")

if __name__ == "__main__":
    main()