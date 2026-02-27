import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
import ijson
import gc
import argparse
from datetime import datetime

from transformers import RobertaTokenizer, RobertaModel

# COnfig
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 256  # Based on GPU memory, can be adjusted

DATA_DIR = './data/twibot22'
OUTPUT_DIR = './temp'

MAX_LENGTH = 128  # Padding set to 128 // faster computation // should be enough for most tweets
MAX_TWEETS_PER_USER = 20  # maximum 20 tweets for user for computational efficiency
CHECKPOINT_INTERVAL = 100000  # Checkpoint every 100k users to detect issues early

EMBEDDING_DIM = 768  # RoBERTa-base output dimension
MODEL_NAME = 'roberta-base'

print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0) 
    print(f"GPU: {gpu_name}")

print(f"Using model: {MODEL_NAME}")
print(f"batch_size={BATCH_SIZE}, max_tweets_per_user={MAX_TWEETS_PER_USER}")

def create_output_dir():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")


def load_labels_and_splits() -> Tuple[Dict[str, int], Dict[str, str]]:

    print("\nLoading labels and splits...")
    
    label_file = os.path.join(DATA_DIR, 'label.csv')
    split_file = os.path.join(DATA_DIR, 'split.csv')
    
    # Load labels
    df_labels = pd.read_csv(label_file)
    label_map = {}
    for _, row in df_labels.iterrows():
        user_id = str(row['id'])
        label = 1 if row['label'] == 'bot' else 0
        label_map[user_id] = label
    
    # Load splits
    df_split = pd.read_csv(split_file)
    split_map = {}
    for _, row in df_split.iterrows():
        user_id = str(row['id'])
        split_map[user_id] = row['split']
    
    print(f"Loaded {len(label_map)} users with labels")
    print(f"Loaded {len(split_map)} users with splits")
    print(f"Label distribution: {pd.Series(label_map.values()).value_counts().to_dict()}")
    
    # label_map: dict of user_id -> label (0=human, 1=bot)
    # split_map: dict of user_id -> split (train/val/test)
    
    return label_map, split_map


def load_tokenizer_and_model():

    print("\n\Loading RoBERTa model and tokenizer...")

    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    model = RobertaModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    
    print(f"Loaded {MODEL_NAME}")
    print(f"Model moved to {DEVICE}")
    
    return tokenizer, model


#Extract tweets for users in label_map.
# user_tweets: dict of user_id -> list of tweet texts
def extract_tweets_from_files(label_map: Dict[str, int]) -> Dict[str, List[str]]:
    print("\nExtracting tweets from JSON files...")
    
    user_tweets = {user_id: [] for user_id in label_map.keys()}
    total_tweets = 0
    saved_tweets = 0
    
    # Process each tweet file // tweet_i.json, i=0..8
    for i in range(9):
        tweet_file = os.path.join(DATA_DIR, f'tweet_{i}.json')
        if not os.path.exists(tweet_file):
            print(f"Warning: {tweet_file} not found, skipping...")
            continue
        
        print(f"\nProcessing tweet_{i}.json...")
        file_size = os.path.getsize(tweet_file) / (1024**3)
        print(f"File size: {file_size:.2f} GB")
        
        try:
            with open(tweet_file, 'rb') as f:
                #  ijson for streaming parsing to handle large files efficiently
                parser = ijson.items(f, 'item')
                
                # Process each tweet in the file
                for tweet in parser:
                    total_tweets += 1
                    
                    # Get author_id and tweet text
                    author_id = tweet.get('author_id', '')
                    # The author_id is an integer in the JSON, add 'u' prefix to match label format
                    author_id_str = f'u{author_id}' if author_id else ''
                    text = tweet.get('text', '')
                    
                    # Save if user is in label map and text is not empty
                    if author_id_str in user_tweets and text.strip():
                        if len(user_tweets[author_id_str]) < MAX_TWEETS_PER_USER:
                            user_tweets[author_id_str].append(text)
                            saved_tweets += 1
                    
                    # Progress indicator
                    if total_tweets % 2000000 == 0:
                        print(f"Processed {total_tweets:,} tweets (saved: {saved_tweets:,})")
            
            print(f"tweet_{i}.json finished (saved {saved_tweets:,} so far)")
            
        except Exception as e:
            print(f"Error processing {tweet_file}: {e}")
            continue
    
    # statistics
    users_with_tweets = sum(1 for tweets in user_tweets.values() if len(tweets) > 0)
    avg_tweets = saved_tweets / users_with_tweets if users_with_tweets > 0 else 0
    
    print(f"\n Total tweets processed: {total_tweets:,}")
    print(f" Tweets saved: {saved_tweets:,}")
    print(f" Users with tweets: {users_with_tweets:,} / {len(user_tweets):,}")
    print(f" Average tweets per user: {avg_tweets:.1f}")
    
    return user_tweets

#Generate RoBERTa embeddings for a list of tweets.
@torch.no_grad()
def generate_embeddings(tweets: List[str],tokenizer,model,batch_size: int = BATCH_SIZE,max_length: int = MAX_LENGTH) -> np.ndarray:
    """
    Aggregates embeddings by averaging.
    
    Dynamic padding: padding='longest' -> longest tweet in bathc
    
    Returns:
        - embeddings: (768,) array - average embedding for all tweets
    """
    # If user has no tweets, return zero vector
    if not tweets:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    
    all_embeddings = []
    
    # Process tweets in batches
    for i in range(0, len(tweets), batch_size):
        batch_tweets = tweets[i:i+batch_size]
        
        # Tokenize 
        encodings = tokenizer(
            batch_tweets,
            max_length=max_length,
            padding='longest',  # dynamic padding to longest tweet in batch
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encodings['input_ids'].to(DEVICE)
        attention_mask = encodings['attention_mask'].to(DEVICE)
        
        # Get embeddings (using [CLS] token as sequence representation)
        outputs = model(input_ids, attention_mask=attention_mask)
        batch_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        all_embeddings.append(batch_embeddings.cpu().numpy())
    
    # Concatenate and average
    all_embeddings = np.vstack(all_embeddings)
    mean_embedding = np.mean(all_embeddings, axis=0)
    
    return mean_embedding.astype(np.float32)


def preprocess_embeddings(user_tweets: Dict[str, List[str]],label_map: Dict[str, int],split_map: Dict[str, str],tokenizer,model,checkpoint_dir: str = None ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Generate embeddings for all users with checkpointing.
    
    Returns:
        - embeddings: (num_users, 768) array
        - labels: (num_users,) array
        - user_ids: (num_users,) array
        - splits: (num_users,) array
    """
    print("\nGenerating embeddings...")
    if checkpoint_dir:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    user_ids = []
    all_embeddings = []
    all_labels = []
    all_splits = []
    
    users_processed = 0
    users_skipped = 0
    
    # Process each user
    for user_id in tqdm(label_map.keys(), desc="Generating embeddings"):
        tweets = user_tweets.get(user_id, [])
        
        # Skip users without tweets TODO TODO
        # if not tweets:
        #     users_skipped += 1
        #     continue
        
        # Generate embedding
        embedding = generate_embeddings(tweets, tokenizer, model)
        
        # Store
        user_ids.append(user_id)
        all_embeddings.append(embedding)
        all_labels.append(label_map[user_id])
        all_splits.append(split_map.get(user_id, 'unknown'))
        
        users_processed += 1
        
        # Checkpoint every CHECKPOINT_INTERVAL users
        if checkpoint_dir and users_processed % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{users_processed}.pt')
            checkpoint_data = {
                'user_ids': user_ids,
                'embeddings': np.array(all_embeddings),
                'labels': np.array(all_labels, dtype=np.int64),
                'splits': np.array(all_splits),
                'users_processed': users_processed,
                'timestamp': datetime.now().isoformat()
            }
            torch.save(checkpoint_data, checkpoint_path, pickle_protocol=4)
            print(f"  Checkpoint saved: {users_processed} users processed")
    
    # Convert to numpy arrays
    embeddings = np.array(all_embeddings)
    labels = np.array(all_labels, dtype=np.int64)
    splits = np.array(all_splits)
    
    print(f"\nProcessed {users_processed} users")
    # print(f"Skipped {users_skipped} users (no tweets)")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Splits distribution: {dict(zip(*np.unique(splits, return_counts=True)))}")
    
    # Clean up to free memory
    del all_embeddings
    del all_labels
    del all_splits
    gc.collect()

    return embeddings, labels, user_ids, splits


def save_embeddings(embeddings: np.ndarray,labels: np.ndarray,user_ids: List[str],splits: np.ndarray,output_filename: str = 'roberta_embeddings.pt'):
    
    print("\nSaving embeddings...")
    
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # Save as PyTorch file for easy loading
    data = {
        'embeddings': torch.from_numpy(embeddings).float() if len(embeddings) > 0 else torch.zeros(0, EMBEDDING_DIM),
        'labels': torch.from_numpy(labels).long(),
        'user_ids': user_ids,
        'splits': splits,
        'model_name': MODEL_NAME,
        'embedding_dim': EMBEDDING_DIM,
        'timestamp': datetime.now().isoformat()
    }
    
    # pickle 4 for large files support
    torch.save(data, output_path, pickle_protocol=4)
    print(f"Saved to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / (1024**3):.2f} GB")
    
    # Also save metadata as CSV
    metadata_path = os.path.join(OUTPUT_DIR, output_filename.replace('.pt', '_metadata.csv'))
    metadata_df = pd.DataFrame({
        'user_id': user_ids,
        'label': labels,
        'split': splits
    })
    metadata_df.to_csv(metadata_path, index=False)
    print(f"Metadata saved to {metadata_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Preprocess')
    parser.add_argument('--output', type=str, default='roberta_embeddings.pt',
                       help='Output filename (default: roberta_embeddings.pt)')
    parser.add_argument('--enable-checkpoint', action='store_true', default=True)
    args = parser.parse_args()
    
    print("=" * 70)
    print("RoBERTa Embedding Preprocessor - TwiBot-22 (v2.0 Optimized)")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Device: {DEVICE}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Max token length: {MAX_LENGTH}")
    print(f"  Max tweets/user: {MAX_TWEETS_PER_USER}")
    print(f"  Checkpointing: {'ENABLED' if args.enable_checkpoint else 'disabled'} (every {CHECKPOINT_INTERVAL} users)")
    print("=" * 70)
    
    # Create output directory
    create_output_dir()
    
    # Load labels and splits
    label_map, split_map = load_labels_and_splits()
    
    # Load model and tokenizer
    tokenizer, model = load_tokenizer_and_model()
    
    # Extract tweets
    user_tweets = extract_tweets_from_files(label_map)
    
    # Generate embeddings
    checkpoint_dir = os.path.join(OUTPUT_DIR, 'checkpoints') if args.enable_checkpoint else None
    embeddings, labels, user_ids, splits = preprocess_embeddings(
        user_tweets, label_map, split_map, tokenizer, model,
        checkpoint_dir=checkpoint_dir
    )
    
    # Clean up to free memory
    del user_tweets
    gc.collect()

    # Save embeddings
    save_embeddings(embeddings, labels, user_ids, splits, args.output)
    
    print("\n" + "=" * 60)
    print("✓ Preprocessing completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
