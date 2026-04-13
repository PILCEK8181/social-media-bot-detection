"""
Offline Bot Detection - Ensemble Model
Combines RF (metadata) and RoBERTa (text embeddings) models

===============================================================================
USAGE:
    python bot_detector.py <username> [--mode demo|live]

EXAMPLE:
    python bot_detector.py Charles_leclerc              # uses pre-scraped demo data
    python bot_detector.py Charles_leclerc --mode live   # scrapes fresh data first

REQUIREMENTS:
    - Demo data files in ./demo/profile_<username>.json and ./tweets_<username>.json
    - Trained models in ./models/:
        * 01_rf.joblib (Random Forest)
        * 03roberta_oversample.pth (RoBERTa classifier)
        * meta_classifier_lr.pkl (Logistic Regression ensemble combiner)

INPUT DATA FORMAT:
    profile_<username>.json:
        - username, name, description, public_metrics (followers_count, following_count,
          tweet_count, listed_count), verified, created_at, location, profile_image_url

    tweets_<username>.json:
        - Date, Text, Likes, Retweets, Replies, Author

OUTPUT:
    - ./output/bot_detection_<username>_<timestamp>.o (formatted report)
    - ./output/bot_detection_<username>_<timestamp>.json (structured data)

MODEL DESCRIPTIONS:
    1. Random Forest: Profile metadata classifier
       - Uses top 15 Gini importance features
       - Learns from account characteristics and behavior patterns
       - Outputs bot probability

    2. RoBERTa: Text content classifier
       - Analyzes user bio and recent tweets with transformer embeddings
       - Bio: 768-dim embedding from profile description
       - Tweets: 768-dim average embedding from up to 20 tweets
       - Outputs bot probability

    3. Meta-Classifier: Logistic Regression ensemble
       - Combines RF and RoBERTa probabilities optimally
       - Provides final bot classification and confidence score

OUTPUT INTERPRETATION:
    - Classification: "BOT" or "HUMAN"
    - Confidence: Probability of predicted class (0.0-1.0)
    - Expert Models: Individual predictions showing how models agree/disagree

===============================================================================
"""

import os
import sys
import json
import joblib
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from detect_rf import TOP_15_FEATURES, predict_rf
from detect_roberta import BotDetectionModel, generate_embeddings as generate_roberta_embeddings

# Suppress warnings
warnings.filterwarnings('ignore')

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
DEMO_DIR = './demo'
MODELS_DIR = './models'
OUTPUT_DIR = './output'
TEMP_DIR = './temp'

# Model paths
RF_MODEL_PATH = os.path.join(MODELS_DIR, '01_rf.joblib')
ROBERTA_MODEL_PATH = os.path.join(MODELS_DIR, '03roberta_oversample.pth')
META_CLASSIFIER_PATH = os.path.join(MODELS_DIR, 'meta_classifier_lr.pkl')

# Make output directory if it doesn't exist
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ============================================================================
# Data Loading & Preprocessing
# ============================================================================

# Load the demo profile and tweets data 
def load_profile_data(username: str) -> Tuple[Dict, List[Dict]]:
    
    profile_path = os.path.join(DEMO_DIR, f'profile_{username}.json')
    tweets_path = os.path.join(DEMO_DIR, f'tweets_{username}.json')
    
    if not os.path.exists(profile_path) or not os.path.exists(tweets_path):
        raise FileNotFoundError(
            f"Data files not found for user '{username}'.\n"
            f"Expected:\n  - {profile_path}\n  - {tweets_path}"
        )
    
    # Load profile
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
        'creation_date': pd.to_datetime(profile.get('created_at')),
    }
    
    # Load tweets
    with open(tweets_path, 'r') as f:
        tweets_data = json.load(f)
    
    # Sort by date
    tweets_data.sort(key=lambda t: t.get('Date', ''))
    
    return profile_data, tweets_data

# Extracted features based on 01_rf.py top 15 Gini importance features
def extract_rf_features(profile_data: Dict) -> Dict:
    
    followers = profile_data['followers_count']
    following = profile_data['following_count']
    tweets = profile_data['tweet_count']
    listed = profile_data['listed_count']
    
    # Account age (fixed reference date matching TwiBot-22 dataset)
    creation_date = profile_data['creation_date']
    if creation_date.tz is None:
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
# Model Predictions
# ============================================================================

def predict_roberta(roberta_model, bio_embedding: np.ndarray, tweets_embedding: np.ndarray) -> float:
    
    # Combine embeddings
    combined = np.concatenate([bio_embedding, tweets_embedding])  # (1536,)
    X = torch.from_numpy(combined).float().unsqueeze(0)  # (1, 1536)
    X = X.to(DEVICE)
    
    with torch.no_grad():
        logits = roberta_model(X)  # (1, 2)
        probs = torch.softmax(logits, dim=1)
        prob = probs[0, 1].item()  # Probability of class 1 (bot)
    
    return float(prob)

# Combine predictions from RF and RoBERTa using meta-classifier
def combine_predictions(rf_prob: float, roberta_prob: float, meta_classifier) -> Tuple[int, float]:
    
    X = np.array([[roberta_prob, rf_prob]]).astype(np.float32)
    prediction = meta_classifier.predict(X)[0]
    probability = meta_classifier.predict_proba(X)[0, 1]
    
    return int(prediction), float(probability)


# ============================================================================
# Main Detection Pipeline
# ============================================================================

def detect_bot(username: str) -> Dict:
    print("\n" + "="*70)
    print(f"BOT DETECTION FOR USER: @{username}")
    print("="*70)
    
    try:
        # Load data
        print("\nLoading profile and tweet data...")
        profile_data, tweets_data = load_profile_data(username)
        print(f"  User: {profile_data['display_name']} (@{username})")
        print(f"  Followers: {profile_data['followers_count']:,}")
        print(f"  Tweets: {profile_data['tweet_count']:,}")
        print(f"  Tweets in demo data: {len(tweets_data)}")
        
        # Extract features
        print("\nExtracting Random Forest features...")
        rf_features = extract_rf_features(profile_data)
        
        print("Extracting RoBERTa features...")
        bio = profile_data['bio']
        tweets = [str(t.get('Text', '')) for t in tweets_data[:20]]
        
        print("Generating RoBERTa embeddings...")
        bio_emb, tweets_emb = generate_roberta_embeddings(bio, tweets)
        
        # Load models
        print("Loading models...")
        rf_model = joblib.load(RF_MODEL_PATH)
        print(f"RF model loaded")
        
        # Load RoBERTa model
        roberta_checkpoint = torch.load(ROBERTA_MODEL_PATH, map_location=DEVICE)
        if isinstance(roberta_checkpoint, dict) and 'model_state_dict' in roberta_checkpoint:
            # Checkpoint saved with metadata
            roberta_state_dict = roberta_checkpoint['model_state_dict']
        else:
            # Direct state dict
            roberta_state_dict = roberta_checkpoint
        
        roberta_model = BotDetectionModel(input_dim=1536)
        roberta_model.load_state_dict(roberta_state_dict)
        roberta_model = roberta_model.to(DEVICE)
        roberta_model.eval()
        print(f"RoBERTa model loaded")
        
        meta_classifier = joblib.load(META_CLASSIFIER_PATH)
        print(f"Meta-Classifier loaded")
        
        # Get predictions
        print("\n" + "="*70)
        print("PREDICTIONS")
        print("="*70)
        
        rf_prob = predict_rf(rf_model, rf_features)
        print(f"\n  Random Forest:      {rf_prob:.4f} (Bot probability)")
        
        roberta_prob = predict_roberta(roberta_model, bio_emb, tweets_emb)
        print(f"  RoBERTa:            {roberta_prob:.4f} (Bot probability)")
        
        final_pred, final_prob = combine_predictions(rf_prob, roberta_prob, meta_classifier)
        
        print("\n" + "="*70)
        print("FINAL DECISION")
        print("="*70)
        
        result_class = "BOT" if final_pred == 1 else "HUMAN"
        
        print(f"  Classification:     {result_class}")
        print(f"  Confidence:         {final_prob:.4f}")
        print("="*70 + "\n")
        
        # Prepare results
        results = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'username': username,
            'display_name': profile_data['display_name'],
            'followers': profile_data['followers_count'],
            'following': profile_data['following_count'],
            'tweets': profile_data['tweet_count'],
            'verified': profile_data['verified'],
            'expert_models': {
                'random_forest': float(rf_prob),
                'roberta': float(roberta_prob),
            },
            'ensemble': {
                'prediction': result_class,
                'confidence': float(final_prob),
                'raw_probability': float(final_prob)
            }
        }
        
        return results
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise


def save_results(results: Dict, username: str):   
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"bot_detection_{username}_{timestamp}.o")
    
    # Write results as formatted text
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("BOT DETECTION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"User: @{results['username']}\n")
        f.write(f"Display Name: {results['display_name']}\n")
        f.write(f"Followers: {results['followers']:,}\n")
        f.write(f"Following: {results['following']:,}\n")
        f.write(f"Tweets: {results['tweets']:,}\n")
        f.write(f"Verified: {results['verified']}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("EXPERT MODEL PREDICTIONS (Bot Probability)\n")
        f.write("-"*70 + "\n")
        f.write(f"Random Forest:  {results['expert_models']['random_forest']:.4f}\n")
        f.write(f"RoBERTa:        {results['expert_models']['roberta']:.4f}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("ENSEMBLE DECISION\n")
        f.write("-"*70 + "\n")
        f.write(f"Classification: {results['ensemble']['prediction']}\n")
        f.write(f"Confidence:     {results['ensemble']['confidence']:.4f}\n\n")
        
        f.write(f"Generated: {results['timestamp']}\n")
    
    # Also save as JSON for easier parsing
    json_output = output_file.replace('.o', '.json')
    with open(json_output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    print(f"JSON results saved to: {json_output}")
    
    return output_file


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Bot Detection using Ensemble Model')
    parser.add_argument('username', help='Twitter username to analyze (without @)')
    parser.add_argument('--mode', choices=['demo', 'live'], default='demo',
                        help='demo = use existing data in ./demo, live = scrape fresh data first')
    args = parser.parse_args()
    
    try:
        if args.mode == 'live':
            from scrape import scrape_user
            scrape_user(args.username, output_dir=DEMO_DIR)
        
        results = detect_bot(args.username)
        results['mode'] = args.mode
        save_results(results, args.username)
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()