import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings
from pathlib import Path
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, matthews_corrcoef, roc_auc_score, 
                             confusion_matrix, classification_report)

from utils.save_metrics import save_metrics

warnings.filterwarnings('ignore')

# Config
DATA_DIR = './data/twibot22/'
USER_FILE = os.path.join(DATA_DIR, 'user.json')
LABEL_FILE = os.path.join(DATA_DIR, 'label.csv')
SPLIT_FILE = os.path.join(DATA_DIR, 'split.csv')
MODELS_DIR = './models'
RESULTS_DIR = './results'
TEMP_DIR = './temp'

REFERENCE_DATE = pd.Timestamp('2022-12-31').tz_localize('UTC')

SEED = random.randint(1, 10000)
random.seed(SEED)
np.random.seed(SEED)

print(f"Device: CPU")
print(f"Seed: {SEED}")
print("=" * 70)


def load_labels():
    print("Loading labels...")
    df_labels = pd.read_csv(LABEL_FILE)
    label_map = dict(zip(df_labels['id'].astype(str), 
                         (df_labels['label'] == 'bot').astype(int)))
    print(f"  Loaded {len(label_map)} labels")
    return label_map


def load_splits():
    print("Loading splits...")
    df_split = pd.read_csv(SPLIT_FILE)
    split_map = dict(zip(df_split['id'].astype(str), df_split['split']))
    print(f"  Loaded {len(split_map)} splits")
    return split_map


def load_raw_users():
    print("Loading user data...")
    with open(USER_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    print(f"  Loaded {len(raw_data)} raw users")
    return raw_data


def extract_user_features(entry):

    metrics = entry.get('public_metrics', {})
    
    # Account age
    age_days = 1
    try:
        dt = pd.to_datetime(entry.get('created_at'))
        if dt.tz is None: 
            dt = dt.tz_localize('UTC')
        age = (REFERENCE_DATE - dt).days
        age_days = max(age, 1)
    except: 
        age_days = 1
    
    followers = int(metrics.get('followers_count', 0))
    following = int(metrics.get('following_count', 0))
    tweets = int(metrics.get('tweet_count', 0))
    listed = int(metrics.get('listed_count', 0))
    
    username = entry.get('username', '')
    name = entry.get('name', '')
    description = entry.get('description', '') or ''
    location = entry.get('location', '') or ''
    url = entry.get('url', '') or ''
    img_url = entry.get('profile_image_url', '')
    
    # Engagement ratios
    follower_following_ratio = followers / max(following, 1)
    tweets_per_day = max(tweets / age_days, 0.001)
    followers_per_tweet = followers / max(tweets, 1)
    listed_followers_ratio = listed / max(followers, 1)
    
    # Text metrics
    username_len = len(username)
    name_len = len(name)
    desc_len = len(description)
    
    # Character counts in text
    name_digits = sum(1 for c in name if c.isdigit())
    name_special = sum(1 for c in name if not c.isalnum() and not c.isspace())
    username_digits = sum(1 for c in username if c.isdigit())
    username_special = sum(1 for c in username if not c.isalnum() and not c.isspace())
    
    # Description features
    has_mention = 1 if '@' in description else 0
    has_hashtag = 1 if '#' in description else 0
    has_url_in_desc = 1 if 'http' in description.lower() else 0
    has_location = 1 if location and len(location) > 0 else 0
    has_url_field = 1 if url and len(url) > 0 else 0
    
    features = {
        'id': entry.get('id', ''),
        # Raw counts
        'followers_count': followers,
        'following_count': following,
        'tweet_count': tweets,
        'listed_count': listed,
        # Ratios (engineered)
        'follower_following_ratio': follower_following_ratio,
        'tweets_per_day': tweets_per_day,
        'followers_per_tweet': followers_per_tweet,
        'listed_followers_ratio': listed_followers_ratio,
        # Profile text metrics
        'username_length': username_len,
        'name_length': name_len,
        'description_length': desc_len,
        'name_digit_count': name_digits,
        'name_special_char_count': name_special,
        'username_digit_count': username_digits,
        'username_special_char_count': username_special,
        # Description features
        'has_mention': has_mention,
        'has_hashtag': has_hashtag,
        'has_url_in_description': has_url_in_desc,
        'has_location': has_location,
        'has_url_field': has_url_field,
        # Account flags
        'verified': 1 if entry.get('verified') is True else 0,
        'protected': 1 if entry.get('protected') is True else 0,
        'default_profile_image': 1 if 'default_profile_images' in img_url else 0,
        'account_age_days': age_days,
    }
    return features

# Process data
def load_data():
    label_map = load_labels()
    split_map = load_splits()
    raw_data = load_raw_users()
    
    print("Processing users...")
    data = []
    for i, entry in enumerate(raw_data):
        if (i + 1) % 50000 == 0:
            print(f"  Processed {i+1}/{len(raw_data)}...")
        
        user_id = str(entry.get('id', ''))
        if user_id not in label_map or user_id not in split_map:
            continue
        
        features = extract_user_features(entry)
        features['id'] = user_id
        features['label'] = label_map[user_id]
        features['split'] = split_map[user_id]
        data.append(features)
    
    df = pd.DataFrame(data)
    print(f"  Total: {len(df)} records")
    return df


def prepare_features(df):
    print("\nPreparing features...")
    
    # Feature engineering: raw counts (log), ratios, text metrics, flags
    raw_counts = ['followers_count', 'following_count', 'tweet_count', 'listed_count']
    for col in raw_counts:
        df[f'log_{col}'] = np.log1p(df[col])
    
    log_features = [f'log_{col}' for col in raw_counts]
    ratios = ['follower_following_ratio', 'tweets_per_day', 'followers_per_tweet', 'listed_followers_ratio']
    
    text_metrics = ['username_length', 'name_length', 'description_length', 'name_digit_count', 
                    'name_special_char_count', 'username_digit_count', 'username_special_char_count']
    
    flags = ['has_mention', 'has_hashtag', 'has_url_in_description', 'has_location', 'has_url_field',
             'verified', 'protected', 'default_profile_image', 'account_age_days']
    
    top15 = ['log_followers_count', 'log_tweet_count', 'follower_following_ratio', 'description_length', 
             'log_listed_count', 'tweets_per_day', 'log_following_count', 'account_age_days', 'listed_followers_ratio', 
             'log_following_count', 'followers_per_tweet', 'name_length', 'verified', 'has_url_field', 'name_special_char_count']
    
    top10 = ['log_followers_count', 'log_tweet_count', 'follower_following_ratio', 'description_length', 'log_listed_count', 
             'tweets_per_day', 'log_following_count', 'account_age_days', 'listed_followers_ratio', 'followers_per_tweet']

    # Final selected features for the model (can be changed to top10, all, or custom)
    feature_cols_final = top15
    
    print(f"  Total features: {len(feature_cols_final)}")
    
    return df, feature_cols_final


def split_data(df, feature_cols_final):
    print("\nSplitting data...")
    
    X_train = df[df['split'] == 'train'][feature_cols_final].copy()
    y_train = df[df['split'] == 'train']['label']
    
    X_valid = df[df['split'] == 'val'][feature_cols_final].copy()
    y_valid = df[df['split'] == 'val']['label']
    
    X_test = df[df['split'] == 'test'][feature_cols_final].copy()
    y_test = df[df['split'] == 'test']['label']

    X_train = pd.DataFrame(X_train, columns=feature_cols_final)
    X_valid = pd.DataFrame(X_valid, columns=feature_cols_final)
    X_test = pd.DataFrame(X_test, columns=feature_cols_final)
    
    print(f"  Train: {len(X_train)}, Valid: {len(X_valid)}, Test: {len(X_test)}")
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def train_model(X_train, y_train):
    print("\nTraining Random Forest...")
    
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5,
                                      min_samples_leaf=2, class_weight='balanced', 
                                      random_state=SEED, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    return rf_model


def evaluate(model, X, y, name):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    metrics = {
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred, zero_division=0),
        'Recall': recall_score(y, y_pred, zero_division=0),
        'F1': f1_score(y, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y, y_pred),
    }
    
    print(f"\n{name}:")
    for metric, value in metrics.items():
        print(f"  {metric:12s}: {value:.4f}")
    
    return metrics, y_pred, y_proba


def save_model(model, path):
    import joblib
    joblib.dump(model, path)
    print(f"Model saved to: {path}")


def plot_results(test_metrics, valid_metrics, train_metrics, feature_cols_final, 
                 rf_model, y_test, y_test_pred, y_test_proba):
    print("\nGenerating plots...")
    
    # Feature importance ranking
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("Top 15 Most Important Features:")
    for i in range(min(15, len(indices))):
        idx = indices[i]
        print(f"  {i+1:2d}. {feature_cols_final[idx]:35s} {importances[idx]:.4f}")
    
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Feature importance
    ax1 = fig.add_subplot(gs[0, 0])
    names = [feature_cols_final[i] for i in indices]
    sns.barplot(x=importances[indices], y=names, ax=ax1, palette="viridis")
    ax1.set_title("Feature Importance", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Importance Weight")
    
    # Confusion matrix
    ax2 = fig.add_subplot(gs[0, 1])
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=['Human', 'Bot'], yticklabels=['Human', 'Bot'])
    ax2.set_title(f"Confusion Matrix (Test Acc: {test_metrics['Accuracy']:.1%})", 
                  fontsize=12, fontweight='bold')
    ax2.set_ylabel("True Label")
    ax2.set_xlabel("Predicted Label")
    
    # Metrics comparison
    ax3 = fig.add_subplot(gs[1, 0])
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'MCC']
    train_vals = [train_metrics[m] for m in metrics_names]
    valid_vals = [valid_metrics[m] for m in metrics_names]
    test_vals = [test_metrics[m] for m in metrics_names]
    
    x = np.arange(len(metrics_names))
    width = 0.25
    ax3.bar(x - width, train_vals, width, label='Train', alpha=0.8)
    ax3.bar(x, valid_vals, width, label='Valid', alpha=0.8)
    ax3.bar(x + width, test_vals, width, label='Test', alpha=0.8)
    ax3.set_ylabel('Score')
    ax3.set_title('Metrics Comparison', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics_names, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Metrics heatmap
    ax4 = fig.add_subplot(gs[1, 1])
    metrics_comparison = pd.DataFrame({
        'Train': train_vals,
        'Valid': valid_vals,
        'Test': test_vals
    }, index=metrics_names)
    sns.heatmap(metrics_comparison, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax4, 
                cbar_kws={'label': 'Score'}, vmin=0, vmax=1)
    ax4.set_title('Metrics Heatmap', fontsize=12, fontweight='bold')
    
    # Prediction confidence distribution
    ax5 = fig.add_subplot(gs[1, 2])
    plot_df = pd.DataFrame({
        'Label': y_test.map({0: 'Human', 1: 'Bot'}),
        'Probability': y_test_proba
    })
    sns.boxplot(x='Label', y='Probability', data=plot_df, ax=ax5, palette="Set2")
    ax5.set_title('Prediction Confidence Distribution', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Bot Probability')
    ax5.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    png_path = os.path.join(RESULTS_DIR, '01_rf_model_evaluation.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Combined evaluation plots saved to: {png_path}")
    plt.close()

# Save predictions for meta-classifier
def save_predictions(df, X_valid, y_valid, X_test, y_test, rf_model):
    print("\nSaving predictions...")
    
    val_uids = df[df['split'] == 'val']['id'].values
    test_uids = df[df['split'] == 'test']['id'].values
    
    val_probs = rf_model.predict_proba(X_valid)[:, 1]
    test_probs = rf_model.predict_proba(X_test)[:, 1]
    
    df_val = pd.DataFrame({'user_id': val_uids, 'prob_rf': val_probs, 'split': 'val', 'label': y_valid.values})
    df_test = pd.DataFrame({'user_id': test_uids, 'prob_rf': test_probs, 'split': 'test', 'label': y_test.values})
    
    Path(TEMP_DIR, 'predictions').mkdir(parents=True, exist_ok=True)
    pd.concat([df_val, df_test]).to_csv(os.path.join(TEMP_DIR, 'predictions', 'preds_rf.csv'), index=False)
    print(f"Predictions saved to: {os.path.join(TEMP_DIR, 'predictions', 'preds_rf.csv')}")


def main():
    print("\n" + "=" * 70)
    print("Random Forest Bot Detection - User Features")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Load and prepare data
    df = load_data()
    print(f"\nDataset splits:\n{df['split'].value_counts()}\nClass distribution:\n{df['label'].value_counts()}")
    
    df, feature_cols_final = prepare_features(df)
    X_train, y_train, X_valid, y_valid, X_test, y_test = split_data(df, feature_cols_final)
    
    # Train model
    rf_model = train_model(X_train, y_train)
    
    # Evaluate
    train_metrics, y_train_pred, y_train_proba = evaluate(rf_model, X_train, y_train, "TRAIN")
    valid_metrics, y_valid_pred, y_valid_proba = evaluate(rf_model, X_valid, y_valid, "VALIDATION")
    test_metrics, y_test_pred, y_test_proba = evaluate(rf_model, X_test, y_test, "TEST")
    
    # Save model
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    save_model(rf_model, os.path.join(MODELS_DIR, '01_rf.joblib'))
    
    # Visualize results
    plot_results(test_metrics, valid_metrics, train_metrics, feature_cols_final, 
                 rf_model, y_test, y_test_pred, y_test_proba)
    
    # Save predictions
    save_predictions(df, X_valid, y_valid, X_test, y_test, rf_model)
    
    # Save metrics
    save_metrics(
        filename="01_rf.py",
        seed=SEED,
        acc=test_metrics['Accuracy'],
        prec=test_metrics['Precision'],
        recall=test_metrics['Recall'],
        f1=test_metrics['F1'],
        mcc=test_metrics['MCC'],
        note="balanced todo final BASE - top 15 features"
    )
    
    print("\n" + "=" * 70)
    print("Training completed")
    print("=" * 70)


if __name__ == '__main__':
    main()