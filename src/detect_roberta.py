"""
Offline Bot Detection - RoBERTa ONLY
Uses strictly text embeddings from Tweets and Profile Bio.

USAGE:
    python bot_detector_roberta_only.py <username>

EXAMPLE:
    python bot_detector_roberta_only.py Charles_leclerc
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple

# Suppress warnings
warnings.filterwarnings('ignore')

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Device: {DEVICE}")

# Paths
DEMO_DIR = './demo'
MODELS_DIR = './models'
OUTPUT_DIR = './output'

# Model paths
ROBERTA_MODEL_PATH = os.path.join(MODELS_DIR, '03roberta_oversample.pth')

# RoBERTa config
ROBERTA_MODEL_NAME = 'roberta-base'
ROBERTA_MAX_LENGTH = 128
EMBEDDING_DIM = 768
MAX_TWEETS = 20

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ============================================================================
# MLP classifier definition (must match architecture used in training)
# ============================================================================

class BotDetectionModel(nn.Module):
    def __init__(self, input_dim: int = 1536): # 768 (Tweets) + 768 (Bio)
        super(BotDetectionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(0.2)
        
        self.output = nn.Linear(128, 2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        
        x = self.output(x)
        return x

# ============================================================================
# Data Loading & Text Extraction
# ============================================================================

def load_text_data(username: str) -> Tuple[str, List[str], str]:
    profile_path = os.path.join(DEMO_DIR, f'profile_{username}.json')
    tweets_path = os.path.join(DEMO_DIR, f'tweets_{username}.json')
    
    if not os.path.exists(profile_path):
        raise FileNotFoundError(f"Profile file not found: {profile_path}")
        
    # Extract Bio
    with open(profile_path, 'r') as f:
        profile = json.load(f)
    display_name = str(profile.get('name', '')).strip()
    bio = str(profile.get('description', '')).strip()
    
    # Extract Tweets (up to MAX_TWEETS)
    tweets_list = []
    if os.path.exists(tweets_path):
        with open(tweets_path, 'r') as f:
            tweets_data = json.load(f)
        valid_tweets = [str(t.get('Text', '')).strip() for t in tweets_data if str(t.get('Text', '')).strip()]
        tweets_list = valid_tweets[:MAX_TWEETS]
        
    return bio, tweets_list, display_name

# ============================================================================
# Embedding Generation
# ============================================================================

def generate_embeddings(bio: str, tweets: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    from transformers import RobertaTokenizer, RobertaModel
    
    tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
    model = RobertaModel.from_pretrained(ROBERTA_MODEL_NAME).to(DEVICE)
    model.eval()
    
    with torch.no_grad():
        # 1. BIO EMBEDDING
        if bio and bio.lower() != 'none': #TODO
            bio_enc = tokenizer(
                bio, 
                max_length=ROBERTA_MAX_LENGTH, 
                padding='max_length', 
                truncation=True, 
                return_tensors='pt'
            )
            bio_out = model(
                bio_enc['input_ids'].to(DEVICE), 
                attention_mask=bio_enc['attention_mask'].to(DEVICE)
            )
            bio_embedding = bio_out.last_hidden_state[0, 0, :].cpu().numpy() # [CLS]
        else:
            bio_embedding = np.zeros(EMBEDDING_DIM, dtype=np.float32)
            
        # 2. TWEETS EMBEDDING (Averaged)
        if tweets:
            tweet_enc = tokenizer(
                tweets, 
                max_length=ROBERTA_MAX_LENGTH, 
                padding='longest', 
                truncation=True, 
                return_tensors='pt'
            )
            tweet_out = model(
                tweet_enc['input_ids'].to(DEVICE), 
                attention_mask=tweet_enc['attention_mask'].to(DEVICE)
            )
            # Extract [CLS] tokens for all tweets
            cls_embeddings = tweet_out.last_hidden_state[:, 0, :].cpu().numpy()
            # Average them
            tweets_embedding = np.mean(cls_embeddings, axis=0)
        else:
            tweets_embedding = np.zeros(EMBEDDING_DIM, dtype=np.float32)
            
    return bio_embedding.astype(np.float32), tweets_embedding.astype(np.float32)

# ============================================================================
# Main Pipeline
# ============================================================================

def detect_bot(username: str) -> Dict:
    print(f"\n{'='*70}\nBOT DETECTION (RoBERTa ONLY) FOR USER: @{username}\n{'='*70}")
    
    # 1. Load Data
    print("\nLoading text data...")
    bio, tweets, display_name = load_text_data(username)
    print(f"  User: {display_name} (@{username})")
    print(f"  Bio length: {len(bio)} chars")
    print(f"  Tweets loaded: {len(tweets)}")
    
    # 2. Generate Embeddings
    print("Generating RoBERTa embeddings (this might take a few seconds)...")
    bio_emb, tweets_emb = generate_embeddings(bio, tweets)
    
    # 3. Load Custom Classification Model
    print("Loading trained classification model...")
    if not os.path.exists(ROBERTA_MODEL_PATH):
        raise FileNotFoundError(f"Model missing at: {ROBERTA_MODEL_PATH}")
        
    clf_model = BotDetectionModel(input_dim=1536).to(DEVICE)
    checkpoint = torch.load(ROBERTA_MODEL_PATH, map_location=DEVICE, weights_only=False)
    
    # Handle saved dictionary format
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        clf_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        clf_model.load_state_dict(checkpoint)
        
    clf_model.eval()
    
    # 4. Prediction
    print("Calculating bot probability...")
    
    # CRITICAL FIX: Order MUST be [Tweets, Bio] as per the training script!
    combined_features = np.concatenate([tweets_emb, bio_emb])
    
    X = torch.from_numpy(combined_features).float().unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits = clf_model(X)
        probs = torch.softmax(logits, dim=1)
        bot_prob = probs[0, 1].item()
    
    # Result
    result_class = "BOT" if bot_prob >= 0.5 else "HUMAN"
    
    print(f"\n{'='*70}\nFINAL DECISION\n{'='*70}")
    print(f"  Classification: {result_class}")
    print(f"  Confidence:     {bot_prob:.4f} (Bot Probability)\n{'='*70}\n")
    
    return {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'username': username,
        'display_name': display_name,
        'model': 'RoBERTa Text Only',
        'prediction': result_class,
        'bot_probability': float(bot_prob),
        'metrics': {
            'tweets_analyzed': len(tweets),
            'has_bio': len(bio) > 0
        }
    }

def save_results(results: Dict, username: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(OUTPUT_DIR, f"roberta_only_{username}_{timestamp}.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"JSON results saved to: {out_file}")

def main():
    parser = argparse.ArgumentParser(description='Offline Bot Detection - RoBERTa ONLY')
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