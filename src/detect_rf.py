"""
Offline Bot Detection - Random Forest ONLY
Uses strictly profile metadata (15 features).

USAGE:
    python bot_detector_rf_only.py <username>

EXAMPLE:
    python bot_detector_rf_only.py Charles_leclerc
"""

import os
import sys
import json
import joblib
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict

# Suppress warnings
warnings.filterwarnings('ignore')

# Paths
DEMO_DIR = './demo'
MODELS_DIR = './models'
OUTPUT_DIR = './output'

# Model path
RF_MODEL_PATH = os.path.join(MODELS_DIR, '01_rf.joblib')

# Top 15 features by Gini importance (must match training order in 01_rf.py)
TOP_15_FEATURES = [
    'log_followers_count', 'log_tweet_count', 'follower_following_ratio', 'description_length',
    'log_listed_count', 'tweets_per_day', 'log_following_count', 'account_age_days',
    'listed_followers_ratio', 'followers_per_tweet', 'name_length', 'verified',
    'username_length', 'has_url_field', 'name_special_char_count'
]

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ============================================================================
# Data Loading & Preprocessing
# ============================================================================

def load_profile_data(username: str) -> Dict:
    
    profile_path = os.path.join(DEMO_DIR, f'profile_{username}.json')
    
    if not os.path.exists(profile_path):
        raise FileNotFoundError(f"Data file not found:\n  - {profile_path}")
    
    with open(profile_path, 'r') as f:
        profile = json.load(f)
    
    metrics = profile.get('public_metrics', {})
    
    profile_data = {
        'username': str(profile.get('username', '')).strip(),
        'display_name': str(profile.get('name', '')).strip(),
        'bio': str(profile.get('description', '')).strip(),
        'followers_count': int(metrics.get('followers_count', 0)),
        'following_count': int(metrics.get('following_count', 0)),
        'tweet_count': int(metrics.get('tweet_count', 0)),
        'listed_count': int(metrics.get('listed_count', 0)),
        'verified': bool(profile.get('verified', False)),
        'creation_date': pd.to_datetime(profile.get('created_at'), errors='coerce'),
    }
        
    return profile_data

# Extracted features based on 01_rf.py top 15 Gini importance features
def extract_rf_features(profile_data: Dict) -> Dict:

    
    followers = profile_data['followers_count']
    following = profile_data['following_count']
    tweets = profile_data['tweet_count']
    listed = profile_data['listed_count']
    
    # Account age
    creation_date = profile_data['creation_date']
    if pd.isna(creation_date):
        creation_date = pd.Timestamp.now(tz='UTC')
    elif creation_date.tz is None:
        creation_date = creation_date.tz_localize('UTC')
    reference_date = pd.Timestamp.now(tz='UTC')
    account_age_days = max((reference_date - creation_date).days, 1)
    
    name = profile_data['display_name']
    username = profile_data['username']
    description = profile_data['bio']
    
    features = {
        'log_followers_count': np.log1p(followers),
        'log_tweet_count': np.log1p(tweets),
        'follower_following_ratio': followers / max(following, 1),
        'description_length': len(description),
        'log_listed_count': np.log1p(listed),
        'tweets_per_day': max(tweets / account_age_days, 0.001),
        'log_following_count': np.log1p(following),
        'account_age_days': account_age_days,
        'listed_followers_ratio': listed / max(followers, 1),
        'followers_per_tweet': followers / max(tweets, 1),
        'name_length': len(name),
        'verified': 1 if profile_data['verified'] else 0,
        'username_length': len(username),
        'has_url_field': 1 if 'http' in description.lower() else 0,
        'name_special_char_count': sum(1 for c in name if not c.isalnum() and not c.isspace()),
    }
    return features

# ============================================================================
# Predictions
# ============================================================================

def predict_rf(rf_model, features: Dict) -> float:
    X_raw = np.array([features[col] for col in TOP_15_FEATURES]).reshape(1, -1)
    return float(rf_model.predict_proba(X_raw)[0, 1])

# ============================================================================
# Main Pipeline
# ============================================================================

def detect_bot(username: str) -> Dict:
    print(f"\n{'='*70}\nBOT DETECTION (RF ONLY) FOR USER: @{username}\n{'='*70}")
    
    # 1. Load Data
    print("\nLoading profile data...")
    profile_data = load_profile_data(username)
    print(f"  User: {profile_data['display_name']} (@{username}) | Followers: {profile_data['followers_count']:,}")
    
    # 2. Extract Features
    print("Extracting Random Forest features...")
    rf_features = extract_rf_features(profile_data)
    
    # 3. Load Models & Predict
    print("Loading RF model and predicting...")
    
    if not os.path.exists(RF_MODEL_PATH):
        raise FileNotFoundError(f"Random Forest model missing at: {RF_MODEL_PATH}")
        
    rf_model = joblib.load(RF_MODEL_PATH)
    
    rf_prob = predict_rf(rf_model, rf_features)
    
    # Simple threshold at 0.5 since we don't have a meta-classifier
    result_class = "BOT" if rf_prob >= 0.5 else "HUMAN"
    
    print(f"\n{'='*70}\nFINAL DECISION\n{'='*70}")
    print(f"  Classification: {result_class}")
    print(f"  Confidence:     {rf_prob:.4f} (Bot Probability)\n{'='*70}\n")
    
    return {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'username': username,
        'display_name': profile_data['display_name'],
        'model': 'Random Forest Only',
        'prediction': result_class,
        'bot_probability': rf_prob
    }

def save_results(results: Dict, username: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(OUTPUT_DIR, f"rf_only_{username}_{timestamp}.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"JSON results saved to: {out_file}")

def main():
    parser = argparse.ArgumentParser(description='Offline Bot Detection - RF ONLY')
    parser.add_argument('username', help='Twitter username to analyze')
    args = parser.parse_args()
    
    try:
        results = detect_bot(args.username)
        save_results(results, args.username)
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()