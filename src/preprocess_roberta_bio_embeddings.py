import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
import gc
import argparse
from datetime import datetime

from transformers import RobertaTokenizer, RobertaModel

# Config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 256  # v2.0: dynamický padding efektivita
DATA_DIR = './data/twibot22'
OUTPUT_DIR = './temp'

MAX_LENGTH = 128  # bios are usually short, 128 tokens should be sufficient
CHECKPOINT_INTERVAL = 100000  # Ccchekpointing

EMBEDDING_DIM = 768  # RoBERTa-base output dimension
MODEL_NAME = 'roberta-base'

print(f"Device: {DEVICE}")
print(f"Using model: {MODEL_NAME}")
print(f"Bio Embeddings v2.0: dynamic_padding, batch_size={BATCH_SIZE}, max_length={MAX_LENGTH}")


def create_output_dir():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")


def load_labels_and_splits() -> Tuple[Dict[str, int], Dict[str, str]]:
    print("\nLoading labels and splits...")
    
    label_file = os.path.join(DATA_DIR, 'label.csv')
    split_file = os.path.join(DATA_DIR, 'split.csv')
    
    # load labels
    df_labels = pd.read_csv(label_file)
    label_map = {}
    for _, row in df_labels.iterrows():
        user_id = str(row['id'])
        label = 1 if row['label'] == 'bot' else 0
        label_map[user_id] = label
    
    # load splits
    df_split = pd.read_csv(split_file)
    split_map = {}
    for _, row in df_split.iterrows():
        user_id = str(row['id'])
        split_map[user_id] = row['split']
    
    print(f"Loaded {len(label_map)} users with labels")
    print(f"Label distribution: {pd.Series(label_map.values()).value_counts().to_dict()}")
    
    return label_map, split_map


def load_tokenizer_and_model():
    print("\nLoading RoBERTa model and tokenizer...")
    
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    model = RobertaModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    
    print(f"Loaded {MODEL_NAME}")
    return tokenizer, model


def extract_bios_from_user_file(label_map: Dict[str, int]) -> Dict[str, str]:

    print("\nExtracting BIOs from user.json...")
    
    user_file = os.path.join(DATA_DIR, 'user.json')
    user_bios = {user_id: '' for user_id in label_map.keys()}
    
    total_users = 0
    users_with_bio = 0
    
    try:
        with open(user_file, 'r', encoding='utf-8') as f:
            # no need for streaming, user file is relatively small
            users_data = json.load(f)
        
        # process each user
        for user in users_data:
            total_users += 1
            
            user_id_str = user.get('id', '')  
            bio = user.get('description', '')
            
            # saves bio if user is in label_map and bio is not empty
            if user_id_str in user_bios:
                if bio and bio.strip():
                    user_bios[user_id_str] = bio.strip()
                    users_with_bio += 1
            
            # Progress print
            if total_users % 100000 == 0:
                print(f"Processed {total_users:,} users (with bio: {users_with_bio:,})")
        
        print(f"Total users processed: {total_users:,}")
        print(f"Users with BIO: {users_with_bio:,} / {len(label_map):,}")
        print(f"Percentage with BIO: {100*users_with_bio/len(label_map):.1f}%")
        
    except Exception as e:
        print(f"Error reading user.json: {e}")
        return user_bios
    
    return user_bios


@torch.no_grad()
def generate_embedding(bio_text: str,tokenizer,model,max_length: int = MAX_LENGTH) -> np.ndarray:

    if not bio_text or not bio_text.strip():
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    
    # tokenize
    encodings = tokenizer(
        bio_text,
        max_length=max_length,
        padding=True,  
        truncation=True,
        return_tensors='pt'
    )
    
    # 
    input_ids = encodings['input_ids'].to(DEVICE)
    attention_mask = encodings['attention_mask'].to(DEVICE)
    
    # Get embedding 
    outputs = model(input_ids, attention_mask=attention_mask)
    embedding = outputs.last_hidden_state[0, 0, :]  # [CLS] token
    
    return embedding.cpu().numpy().astype(np.float32)


def preprocess_embeddings(user_bios: Dict[str, str],label_map: Dict[str, int],split_map: Dict[str, str],tokenizer,model,checkpoint_dir: str = None) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:

    print("\nGenerating embeddings...")
    if checkpoint_dir:
        print(f"  Checkpointing enabled: every {CHECKPOINT_INTERVAL} users")
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    user_ids = []
    all_embeddings = []
    all_labels = []
    all_splits = []
    
    users_processed = 0
    users_skipped = 0
    
    # process each user
    for user_id in tqdm(label_map.keys(), desc="Generating bio embeddings"):
        bio = user_bios.get(user_id, '')
        
        # skip if no bio
        if not bio or not bio.strip():
            users_skipped += 1
            continue
        
        embedding = generate_embedding(bio, tokenizer, model)
        
        # save
        user_ids.append(user_id)
        all_embeddings.append(embedding)
        all_labels.append(label_map[user_id])
        all_splits.append(split_map.get(user_id, 'unknown'))
        
        users_processed += 1
        
        # checpoints
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
            torch.save(checkpoint_data, checkpoint_path)
            print(f"  Checkpoint saved: {users_processed} users processed")
    
    # convert to numpy arrays
    embeddings = np.array(all_embeddings)
    labels = np.array(all_labels, dtype=np.int64)
    splits = np.array(all_splits)
    
    print(f"\nProcessed {users_processed} users")
    print(f"Skipped {users_skipped} users (no bio)")
    print(f"Embeddings shape: {embeddings.shape}")
    if len(labels) > 0:
        print(f"Labels distribution: {np.bincount(labels)}")
    else:
        print(f"Labels distribution: (no users with bio)")
    print(f"Splits distribution: {dict(zip(*np.unique(splits, return_counts=True)))}")
    
    return embeddings, labels, user_ids, splits

# save embeddings in pytorch format 
def save_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    user_ids: List[str],
    splits: np.ndarray,
    output_filename: str = 'roberta_bio_embeddings.pt'
):
    print("\nSaving embeddings...")
    
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    #  PyTorch dict
    data = {
        'embeddings': torch.from_numpy(embeddings).float() if len(embeddings) > 0 else torch.zeros(0, EMBEDDING_DIM),
        'labels': torch.from_numpy(labels).long(),
        'user_ids': user_ids,
        'splits': splits,
        'model_name': MODEL_NAME,
        'embedding_dim': EMBEDDING_DIM,
        'data_type': 'bio',
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(data, output_path)
    print(f"Saved to {output_path}")
    if os.path.exists(output_path):
        print(f"File size: {os.path.getsize(output_path) / (1024**3):.2f} GB")
    
    # metadata csv
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
    parser = argparse.ArgumentParser(description='bio')
    parser.add_argument('--output', type=str, default='roberta_bio_embeddings.pt',
                       help='Output filename (default: roberta_bio_embeddings.pt)')
    parser.add_argument('--enable-checkpoint', action='store_true',
                       help='Enable checkpointing (v2.0 feature)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("RoBERTa Bio Embedding Preprocessor - TwiBot-22 (v2.0)")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Device: {DEVICE}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nOptimizations (v2.0):")
    print(f"  Batch processing: dynamic (bio by bio)")
    print(f"  Max token length: {MAX_LENGTH}")
    print(f"  Checkpointing: {'ENABLED' if args.enable_checkpoint else 'disabled'} (every {CHECKPOINT_INTERVAL} users)")
    print("=" * 70)
    
    # ouptut dir
    create_output_dir()
    
    # labels and splits
    label_map, split_map = load_labels_and_splits()
    
    # model and tokenizer
    tokenizer, model = load_tokenizer_and_model()
    
    # Extract BIOs
    user_bios = extract_bios_from_user_file(label_map)
    
    # embeddings
    checkpoint_dir = os.path.join(OUTPUT_DIR, 'bio_checkpoints') if args.enable_checkpoint else None
    embeddings, labels, user_ids, splits = preprocess_embeddings(
        user_bios, label_map, split_map, tokenizer, model,
        checkpoint_dir=checkpoint_dir
    )
    
    # save embeddings
    save_embeddings(embeddings, labels, user_ids, splits, args.output)
    
    print("\n" + "=" * 70)
    print("✓ Bio Embedding Preprocessing completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
