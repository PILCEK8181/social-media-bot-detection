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
BATCH_SIZE = 256  # 
DATA_DIR = './data/twibot22'
OUTPUT_DIR = './temp'

MAX_LENGTH = 128  # Bios are usually short
EMBEDDING_DIM = 768  # RoBERTa-base output dimension
MODEL_NAME = 'roberta-base'

print(f"Device: {DEVICE}")
print(f"Using model: {MODEL_NAME}")
print(f"Bio Embeddings, Batched processing, fixed Zero Imputation, batch_size={BATCH_SIZE}")


def load_labels_and_splits() -> Tuple[Dict[str, int], Dict[str, str]]:
    print("\nLoading labels and splits...")
    label_file = os.path.join(DATA_DIR, 'label.csv')
    split_file = os.path.join(DATA_DIR, 'split.csv')
    
    df_labels = pd.read_csv(label_file)
    label_map = {str(row['id']): (1 if row['label'] == 'bot' else 0) for _, row in df_labels.iterrows()}
    
    df_split = pd.read_csv(split_file)
    split_map = {str(row['id']): row['split'] for _, row in df_split.iterrows()}
    
    print(f"Loaded {len(label_map)} users with labels")
    return label_map, split_map


def load_tokenizer_and_model():
    print("\nLoading RoBERTa model and tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    model = RobertaModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    return tokenizer, model


def extract_bios_from_user_file(label_map: Dict[str, int]) -> Dict[str, str]:
    print("\nExtracting BIOs from user.json...")
    user_file = os.path.join(DATA_DIR, 'user.json')
    user_bios = {user_id: '' for user_id in label_map.keys()}
    
    users_with_bio = 0
    try:
        with open(user_file, 'r', encoding='utf-8') as f:
            users_data = json.load(f)
        
        for user in users_data:
            user_id_str = str(user.get('id', ''))  
            bio = user.get('description', '')
            
            if user_id_str in user_bios:
                if bio and str(bio).strip() and str(bio).strip() != 'None':
                    user_bios[user_id_str] = str(bio).strip()
                    users_with_bio += 1
                    
        print(f"Users with valid BIO: {users_with_bio:,} / {len(label_map):,}")
        
    except Exception as e:
        print(f"Error reading user.json: {e}")
    
    return user_bios


@torch.no_grad()
def generate_batch_embeddings(bio_texts: List[str], tokenizer, model, max_length: int = MAX_LENGTH) -> np.ndarray:
    # Tokenize and generate embeddings for a batch of bios
    encodings = tokenizer(
        bio_texts,
        max_length=max_length,
        padding='longest',  # dynamic padding
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encodings['input_ids'].to(DEVICE)
    attention_mask = encodings['attention_mask'].to(DEVICE)
    
    outputs = model(input_ids, attention_mask=attention_mask)
    embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token 
    
    return embeddings.cpu().numpy()


def preprocess_embeddings(user_bios: Dict[str, str], label_map: Dict[str, int], split_map: Dict[str, str], tokenizer, model) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    print("\nGenerating embeddings (Batched + Zero Imputation)...")
    
    all_user_ids = list(label_map.keys())
    
    all_embeddings = []
    all_labels = []
    all_splits = []
    
    current_batch_texts = []
    current_batch_indices = []
    
    # Prepare final embeddings array with zeros (for users without bios)
    final_embeddings = np.zeros((len(all_user_ids), EMBEDDING_DIM), dtype=np.float32)
    
    missing_bios = 0
    
    for idx, user_id in enumerate(tqdm(all_user_ids, desc="Processing users")):
        all_labels.append(label_map[user_id])
        all_splits.append(split_map.get(user_id, 'unknown'))
        
        bio = user_bios.get(user_id, '')
        
        # No bio -> leave embedding as zero vector which is already initialized in final_embeddings
        if not bio:
            missing_bios += 1
            continue
            
        # Else add to current batch
        current_batch_texts.append(bio)
        current_batch_indices.append(idx)
        
        # Send batch for embedding generation when batch size is reached
        if len(current_batch_texts) == BATCH_SIZE:
            batch_emb = generate_batch_embeddings(current_batch_texts, tokenizer, model)
            for j, original_idx in enumerate(current_batch_indices):
                final_embeddings[original_idx] = batch_emb[j]
            
            # Clear
            current_batch_texts = []
            current_batch_indices = []
            
    # process any remaining bios in the last batch
    if current_batch_texts:
        batch_emb = generate_batch_embeddings(current_batch_texts, tokenizer, model)
        for j, original_idx in enumerate(current_batch_indices):
            final_embeddings[original_idx] = batch_emb[j]

    labels = np.array(all_labels, dtype=np.int64)
    splits = np.array(all_splits)
    
    print(f"\nTotal users processed: {len(all_user_ids)}")
    print(f"Users with ZERO embeddings (no bio): {missing_bios}")
    print(f"Final Embeddings shape: {final_embeddings.shape}")
    
    return final_embeddings, labels, all_user_ids, splits


def save_embeddings(embeddings: np.ndarray, labels: np.ndarray, user_ids: List[str], splits: np.ndarray, output_filename: str):
    print("\nSaving embeddings...")
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    data = {
        'embeddings': torch.from_numpy(embeddings).float(),
        'labels': torch.from_numpy(labels).long(),
        'user_ids': user_ids,
        'splits': splits,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(data, output_path)
    print(f"Saved formatted data to {output_path}")

def main():
    # easier than function
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    label_map, split_map = load_labels_and_splits()
    tokenizer, model = load_tokenizer_and_model()
    user_bios = extract_bios_from_user_file(label_map)
    
    embeddings, labels, user_ids, splits = preprocess_embeddings(user_bios, label_map, split_map, tokenizer, model)
    
    save_embeddings(embeddings, labels, user_ids, splits, 'roberta_bio_embeddings.pt')

if __name__ == '__main__':
    main()