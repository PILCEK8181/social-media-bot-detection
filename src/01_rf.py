import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, matthews_corrcoef, roc_auc_score, 
                             confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import re
import joblib
import warnings
warnings.filterwarnings('ignore')
from utils.save_metrics import save_metrics
import random

#TODO todays date?
#TODO refactor
# Account age reference point
REFERENCE_DATE = pd.Timestamp('2022-12-31').tz_localize('UTC')
# Paths
USER_FILE = './data/twibot22/user.json'
LABEL_FILE = './data/twibot22/label.csv'
SPLIT_FILE = './data/twibot22/split.csv'
SEED = random.randint(1, 10000)

def load_data():
    print("Loading labels...")
    df_labels = pd.read_csv(LABEL_FILE)
    label_map = dict(zip(df_labels['id'].astype(str), 
                        (df_labels['label'] == 'bot').astype(int)))
    print(f"Loaded {len(label_map)} labels")
    
    print("Loading splits...")
    df_split = pd.read_csv(SPLIT_FILE)
    split_map = dict(zip(df_split['id'].astype(str), df_split['split']))
    print(f"Loaded {len(split_map)} splits")
    
    print("Loading user data...")
    with open(USER_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    data = []
    for i, entry in enumerate(raw_data):
        if (i + 1) % 50000 == 0:
            print(f"  Processing {i+1}/{len(raw_data)}...")
        
        user_id = str(entry.get('id', ''))
        if user_id not in label_map or user_id not in split_map:
            continue
        
        metrics = entry.get('public_metrics', {})
        
        # Account age
        age_days = 0
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
            'id': user_id,
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
            # Target
            'label': label_map[user_id],
            'split': split_map[user_id]
        }
        data.append(features)
    
    df = pd.DataFrame(data)
    print(f"Total: {len(df)} records")
    return df

df = load_data()
print(f"\nDataset splits:\n{df['split'].value_counts()}\nClass distribution:\n{df['label'].value_counts()}")



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

feature_cols_final = log_features + ratios + text_metrics + flags
print(f"Total features: {len(feature_cols_final)}")

X_train = df[df['split'] == 'train'][feature_cols_final].copy()
y_train = df[df['split'] == 'train']['label']
X_valid = df[df['split'] == 'val'][feature_cols_final].copy()
y_valid = df[df['split'] == 'val']['label']
X_test = df[df['split'] == 'test'][feature_cols_final].copy()
y_test = df[df['split'] == 'test']['label']

# Z-score
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=feature_cols_final)
X_valid = pd.DataFrame(X_valid, columns=feature_cols_final)
X_test = pd.DataFrame(X_test, columns=feature_cols_final)

print("Z-score normalization applied")

print(f"Train: {len(X_train)}, Valid: {len(X_valid)}, Test: {len(X_test)}")

#todo
print("\nTraining RandomForest...")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5,
                                  min_samples_leaf=2, class_weight='balanced', 
                                  random_state=SEED, n_jobs=-1)
rf_model.fit(X_train, y_train)

def evaluate(X, y, name):
    y_pred = rf_model.predict(X)
    y_proba = rf_model.predict_proba(X)[:, 1]
    
    metrics = {
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred, zero_division=0),
        'Recall': recall_score(y, y_pred, zero_division=0),
        'F1': f1_score(y, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y, y_pred),
        #'ROC-AUC': roc_auc_score(y, y_proba)
    }
    
    print(f"\n {name}:")
    for metric, value in metrics.items():
        print(f"  {metric:12s}: {value:.4f}")
    
    return metrics, y_pred, y_proba

train_metrics, y_train_pred, y_train_proba = evaluate(X_train, y_train, "TRAIN")
valid_metrics, y_valid_pred, y_valid_proba = evaluate(X_valid, y_valid, "VALIDATION")
test_metrics, y_test_pred, y_test_proba = evaluate(X_test, y_test, "TEST")

# save model
joblib.dump(rf_model, './models/01_rf.joblib')
print("\nModel saved")


######################################## --- EVALUATION & PLOTTING--- ########################################


# Feature importance ranking
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
print("\nTop 15 Most Important Features:")
for i in range(min(15, len(indices))):
    idx = indices[i]
    print(f"  {i+1:2d}. {feature_cols_final[idx]:35s} {importances[idx]:.4f}")


sns.set_style("whitegrid")
fig = plt.figure(figsize=(20, 12))
# 2 x 3 grid
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Feature importance
ax1 = fig.add_subplot(gs[0, 0])
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
names = [feature_cols_final[i] for i in indices]
sns.barplot(x=importances[indices], y=names, ax=ax1, palette="viridis")
ax1.set_title("Feature Importance", fontsize=12, fontweight='bold')
ax1.set_xlabel("Importance Weight")

# Confussion matrix
ax2 = fig.add_subplot(gs[0, 1])
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=['Human', 'Bot'], yticklabels=['Human', 'Bot'])
ax2.set_title(f"Confusion Matrix (Test Acc: {test_metrics['Accuracy']:.1%})", 
              fontsize=12, fontweight='bold')
ax2.set_ylabel("True Label")
ax2.set_xlabel("Predicted Label")

# Metrics comp
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

# metrics heatmap
ax4 = fig.add_subplot(gs[1, 1])
metrics_comparison = pd.DataFrame({
    'Train': train_vals,
    'Valid': valid_vals,
    'Test': test_vals
}, index=metrics_names)
sns.heatmap(metrics_comparison, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax4, 
            cbar_kws={'label': 'Score'}, vmin=0, vmax=1)
ax4.set_title('Metrics Heatmap', fontsize=12, fontweight='bold')

# prediction confidence distribution
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
plt.savefig('./results/TEST01_rf_model_evaluation.png', dpi=300, bbox_inches='tight')
print("Plots saved to ../results/TEST01_rf_model_evaluation.png")
plt.show()


# Extract user IDs for validation and test sets
val_uids = df[df['split'] == 'val']['id'].values
test_uids = df[df['split'] == 'test']['id'].values

val_probs = rf_model.predict_proba(X_valid)[:, 1] # bot prob
test_probs = rf_model.predict_proba(X_test)[:, 1]

df_val = pd.DataFrame({'user_id': val_uids, 'prob_rf': val_probs, 'split': 'val', 'label': y_valid.values})
df_test = pd.DataFrame({'user_id': test_uids, 'prob_rf': test_probs, 'split': 'test', 'label': y_test.values})
pd.concat([df_val, df_test]).to_csv('./temp/preds_rf.csv', index=False)

save_metrics(
        filename="01_rf.py",
        seed=SEED,
        acc=test_metrics['Accuracy'],
        prec=test_metrics['Precision'],
        recall=test_metrics['Recall'],
        f1=test_metrics['F1'],
        mcc=test_metrics['MCC'],
        note="Random Forest // feature"
    )