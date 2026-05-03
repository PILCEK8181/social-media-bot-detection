"""
Bot Detection Ensemble - Demo/Testing Script

Author: xzacekp00 Patrik Žáček
Institution: BUT FIT (Brno University of Technology, Faculty of Information Technology)
Type: Bachelor's Thesis 2026
Topic: Detection of Fake Accounts on Social Media Networks

Description:
    Offline bot detection using the complete ensemble model. Combines predictions
    from Random Forest (metadata), RoBERTa (content embeddings), and stacks them
    with Logistic Regression for final bot/human classification.

USAGE:
    python bot_detector.py --target <username> [--mode demo|live] [--threshold 0.5] [--verbose]

EXAMPLE:
    python bot_detector.py --target Charles_leclerc              # uses pre-scraped demo data
    python bot_detector.py --target Charles_leclerc --mode live   # scrapes fresh data (not stored)
    python bot_detector.py --target @suspect_user --threshold 0.7 --verbose  # with custom threshold and debug logging

CONFIGURABLE PARAMETERS:
    --target: String - Target username (e.g., @elonmusk) or path to a batch file (required)
    --mode: String - "live" or "demo" (default: demo)
              * demo: Uses pre-scraped data from ./demo/ directory
              * live: Scrapes fresh data and performs analysis WITHOUT storing scraped data
    --threshold: Float - Probability threshold for binary classification (default: 0.5)
    --verbose: Flag - Enables detailed logging of the feature extraction process
              * In live mode, shows debug info about temporary directory cleanup

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
    - ./output/bot_detection_<username>_<timestamp>.json (structured JSON data)

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

OUTPUT SPECIFICATION:
    Console Output:
        - Human-readable summary printed to stdout
        - Classification label (BOT or HUMAN)
        - Confidence score (probability 0.0-1.0)
        - Individual modality scores (Metadata, Text, Temporal)
    
    Structured Output (JSON):
        - username: Twitter/X username
        - prediction: Binary label based on the decision threshold
        - probability: Aggregate likelihood of the account being a bot (P ∈ [0, 1])
        - modality_scores: Raw confidence scores from individual sub-models
          * metadata: Random Forest probability
          * text: RoBERTa probability
        - timestamp: ISO 8601 timestamp of analysis
        - threshold: Probability threshold used for classification

===============================================================================
"""

import os
import sys
import json
import joblib
import argparse
import warnings
import tempfile
import shutil
import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from detect_rf import predict_rf
from detect_roberta import BotDetectionModel, generate_embeddings as generate_roberta_embeddings

# Suppress warnings
warnings.filterwarnings('ignore')

# Suppress all transformers and HF Hub messages
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

# Set logging to error level for verbose libraries
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('transformers.modeling_utils').setLevel(logging.ERROR)
logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.ERROR)
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
logging.getLogger('filelock').setLevel(logging.ERROR)

# Suppress all root logger output
logging.getLogger().setLevel(logging.ERROR)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

DEMO_DIR = PROJECT_ROOT / 'demo'
MODELS_DIR = PROJECT_ROOT / 'models'
OUTPUT_DIR = PROJECT_ROOT / 'output'
TEMP_DIR = PROJECT_ROOT / 'temp'


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
def load_profile_data(username: str, data_dir: str = None) -> Tuple[Dict, List[Dict]]:
    
    if data_dir is None:
        data_dir = DEMO_DIR
    
    profile_path = os.path.join(data_dir, f'profile_{username}.json')
    tweets_path = os.path.join(data_dir, f'tweets_{username}.json')
    
    if not os.path.exists(profile_path) or not os.path.exists(tweets_path):
        raise FileNotFoundError(
            f"Data files not found for user '{username}'.\n"
            f"Expected:\n  - {profile_path}\n  - {tweets_path}"
        )
    
    # Load profile
    with open(profile_path, 'r', encoding='utf-8') as f:
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
    with open(tweets_path, 'r', encoding='utf-8') as f:
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
    
    # Top 15 features based on Gini importance from 01_rf.py
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

def detect_bot(username: str, threshold: float = 0.5, verbose: bool = False, data_dir: str = None) -> Dict:
    if verbose:
        print("[INFO] Detailed logging enabled")
    
    print("\n" + "="*70)
    print(f"REPORT for @{username}")
    print("="*70)
    
    try:
        # Load data
        print("\n[INFO] Fetching timeline for @{username}...".format(username=username))
        profile_data, tweets_data = load_profile_data(username, data_dir=data_dir)
        if verbose:
            print(f"  [DEBUG] User: {profile_data['display_name']} (@{username})")
            print(f"  [DEBUG] Followers: {profile_data['followers_count']:,}")
            print(f"  [DEBUG] Tweets: {profile_data['tweet_count']:,}")
            print(f"  [DEBUG] Tweets in demo data: {len(tweets_data)}")
        else:
            print(f"[INFO] Analyzed {len(tweets_data)} tweets.")
        
        # Extract features
        if verbose:
            print("\n[DEBUG] Extracting Random Forest features...")
        rf_features = extract_rf_features(profile_data)
        
        if verbose:
            print("[DEBUG] Extracting RoBERTa features...")
        bio = profile_data['bio']
        tweets = [str(t.get('Text', '')) for t in tweets_data[:20]]
        
        if verbose:
            print("[DEBUG] Generating RoBERTa embeddings...")
        bio_emb, tweets_emb = generate_roberta_embeddings(bio, tweets)
        
        # Load models
        if verbose:
            print("\n[DEBUG] Loading models...")
        rf_model = joblib.load(RF_MODEL_PATH)
        if verbose:
            print(f"[DEBUG] RF model loaded")
        
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
        if verbose:
            print(f"[DEBUG] RoBERTa model loaded")
        
        meta_classifier = joblib.load(META_CLASSIFIER_PATH)
        if verbose:
            print(f"[DEBUG] Meta-Classifier loaded")
        
        # Get predictions
        if verbose:
            print("\n[DEBUG] Extracting features...")
        
        rf_prob = predict_rf(rf_model, rf_features)
        roberta_prob = predict_roberta(roberta_model, bio_emb, tweets_emb)
        final_pred, final_prob = combine_predictions(rf_prob, roberta_prob, meta_classifier)
        
        # Apply threshold
        prediction_label = 1 if final_prob >= threshold else 0
        result_class = "BOT" if prediction_label == 1 else "HUMAN"
        
        # Console Output - Human-readable summary
        print("\n" + "-"*70)
        print(f"Verdict: {result_class}")
        print(f"Confidence: {final_prob:.2f}")
        print(f"Breakdown: Metadata({rf_prob:.2f}) | Text({roberta_prob:.2f})")
        print("-"*70)
        print(f"[INFO] Result saved to /output/bot_detection_{username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        print("-"*70 + "\n")
        
        # Prepare results - Structured Output (JSON)
        results = {
            'username': username,
            'prediction': result_class,
            'probability': float(final_prob),
            'threshold': threshold,
            'modality_scores': {
                'metadata': float(rf_prob),
                'text': float(roberta_prob),
            },
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'display_name': profile_data['display_name'],
            'followers': profile_data['followers_count'],
            'following': profile_data['following_count'],
            'tweets': profile_data['tweet_count'],
            'verified': profile_data['verified'],
        }
        
        return results
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise


def save_results(results: Dict, username: str):   
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"bot_detection_{username}_{timestamp}.json")
    
    # Save structured JSON output
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    #print(f"[INFO] Result saved to {os.path.basename(output_file)}")
    return output_file


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Bot Detection using Ensemble Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
EXAMPLES:
  %(prog)s --target Charles_leclerc
  %(prog)s --target @suspect_user --mode live --threshold 0.7
  %(prog)s --target @suspect_user --threshold 0.7 --verbose
        ''')
    
    parser.add_argument('--target', required=True, 
                        help='Target username (e.g., @elonmusk) or path to a batch file')
    parser.add_argument('--mode', choices=['demo', 'live'], default='demo',
                        help='demo = use existing data in ./demo, live = scrape fresh data first (default: demo)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold for binary classification (default: 0.5)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enables detailed logging of the feature extraction process')
    
    args = parser.parse_args()
    
    # Clean up username (remove @ if present)
    username = args.target.lstrip('@')
    
    # Set up data directory for live mode
    temp_dir = None
    data_dir = DEMO_DIR
    
    try:
        if args.mode == 'live':
            from scrape import scrape_user
            # Create temporary directory for live scrape
            temp_dir = tempfile.mkdtemp(prefix='bot_detection_live_')
            print(f"[INFO] Using temporary directory: {temp_dir}")
            scrape_user(username, output_dir=temp_dir)
            data_dir = temp_dir
        
        results = detect_bot(username, threshold=args.threshold, verbose=args.verbose, data_dir=data_dir)
        save_results(results, username)
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        sys.exit(1)
    finally:
        # Clean up temporary directory if it was created for live mode (so it doesnt violate the "no data storage" requirement)
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            if args.verbose:
                print(f"[DEBUG] Cleaned up temporary directory: {temp_dir}")


if __name__ == '__main__':
    main()