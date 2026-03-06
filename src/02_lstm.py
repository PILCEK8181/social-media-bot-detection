"""
LSTM Bot Detection - Inter-Arrival Times (IAT)
Využívá data z processed_timestamps.csv a datum založení účtu z user.json.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, matthews_corrcoef
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import random
import time

SEED = int(time.time()) # unique seed acorfing to the time
print(f"seed: {SEED}")

# --- fixed seed ---
# SEED = 16
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(SEED)
#     torch.backends.cudnn.deterministic = True
# -----------------------------

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = './data/twibot22'
TEMP_DIR = './temp'
MODELS_DIR = './models'

# hyperparams
MAX_SEQ_LEN = 200    # Max 200 tweets per user
BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 1e-3 # todo try same as roberta

print(f"Device: {DEVICE}")
print("=" * 70)


def load_and_prepare_iat_data():
    """loads labels creatio times and iat's"""
    print("\nData Processing & IAT Calculation...")
    
    # 1. labels and splits
    df_labels = pd.read_csv(os.path.join(DATA_DIR, 'label.csv'))
    df_split = pd.read_csv(os.path.join(DATA_DIR, 'split.csv'))
    
    label_map = {str(row['id']).replace('u', ''): (1 if row['label'] == 'bot' else 0) for _, row in df_labels.iterrows()}
    split_map = {str(row['id']).replace('u', ''): row['split'] for _, row in df_split.iterrows()}
    
    # 2. account creation times
    print("  Loading account creation times from user.json...")
    user_creation = {}
    with open(os.path.join(DATA_DIR, 'user.json'), 'r', encoding='utf-8') as f:
        users = json.load(f)
        for u in tqdm(users, desc="  Parsing users"):
            uid = str(u.get('id', '')).replace('u', '')
            created_at = u.get('created_at')
            if uid in label_map and created_at:
                # faster
                user_creation[uid] = pd.to_datetime(created_at, format='mixed', utc=True)

    # 3. tweet times
    print("  Loading tweet timestamps (this might take a minute)...")
    df_tweets = pd.read_csv(os.path.join(TEMP_DIR, 'processed_timestamps.csv'))
    df_tweets['user_id'] = df_tweets['user_id'].astype(str)
    # Zrychlený převod času
    df_tweets['timestamp'] = pd.to_datetime(df_tweets['timestamp'], format='mixed', utc=True)
    
    # 4. group tweets by user
    print("  Grouping and sorting sequences...")
    grouped = df_tweets.groupby('user_id')['timestamp'].apply(list).to_dict()
    del df_tweets # Uvolnění RAM
    
    # 5. calc iat and create tensor
    all_user_ids = list(label_map.keys())
    
    # matrxi full of -1
    X = np.full((len(all_user_ids), MAX_SEQ_LEN), -1.0, dtype=np.float32)
    y = np.zeros(len(all_user_ids), dtype=np.float32)
    splits = []
    
    missing_creation = 0
    
    for idx, uid in enumerate(tqdm(all_user_ids, desc="  Calculating IATs")):
        y[idx] = label_map[uid]
        splits.append(split_map.get(uid, 'unknown'))
        
        # build timeline
        timestamps = []
        if uid in user_creation:
            timestamps.append(user_creation[uid])
        else:
            missing_creation += 1
            
        if uid in grouped:
            timestamps.extend(grouped[uid])
            
        if len(timestamps) > 1:
            # sort chronologically
            timestamps = sorted(timestamps)
            
            # diff in seconds
            diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() for i in range(1, len(timestamps))]
            
            diffs = diffs[:MAX_SEQ_LEN]
            
            # Log transf: log(1 + x) // todo del?
            diffs_log = np.log1p(np.maximum(diffs, 0)) 
            
            # load into matrix rest stays -1
            X[idx, :len(diffs_log)] = diffs_log
            
    print(f" Users without creation date (should be ~0): {missing_creation}")
    print(f" IAT Tensor Shape: {X.shape}")
    
    # splits
    train_mask = np.array([s == 'train' for s in splits])
    val_mask = np.array([s == 'val' for s in splits])
    test_mask = np.array([s == 'test' for s in splits])
    
    # torch format
    X_tensor = torch.tensor(X).unsqueeze(-1)
    y_tensor = torch.tensor(y)
    
    train_loader = DataLoader(TensorDataset(X_tensor[train_mask], y_tensor[train_mask]), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_tensor[val_mask], y_tensor[val_mask]), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_tensor[test_mask], y_tensor[test_mask]), batch_size=BATCH_SIZE, shuffle=False)
    
    # weights
    num_bots = y[train_mask].sum()
    num_humans = len(y[train_mask]) - num_bots
    pos_weight = torch.tensor([num_humans / (num_bots + 1e-5)]).to(DEVICE)
    print(f" Calculated pos_weight for BCE: {pos_weight.item():.4f}")
    
    return train_loader, val_loader, test_loader, pos_weight


class IAT_LSTM_Model(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(IAT_LSTM_Model, self).__init__()

        #input
        # batch_first=True ->  (batch, seq_len, features)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(32, 1) # bin out
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, 1)
        
        output, (h_n, c_n) = self.lstm(x)
        
        # last state (LSTMs last layer)
        last_hidden = h_n[-1, :, :] # shape: (batch_size, hidden_size)
        
        x = self.fc1(last_hidden)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out(x)
        
        return x.squeeze() # shape: (batch_size)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        
        # Gradient clipping 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item()
            
            # sigmoid converts logits to probs
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).int()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    
    return total_loss / len(loader), acc, f1, all_preds, all_labels


def main():
    print("\n" + "=" * 70)
    print("LSTM Bot Detection Training - Inter-Arrival Times")
    print("=" * 70)
    
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    
    # 1. load data and weights
    train_loader, val_loader, test_loader, pos_weight = load_and_prepare_iat_data()
    
    # 2. model init
    model = IAT_LSTM_Model(hidden_size=64, num_layers=2).to(DEVICE)
    print("\nModel Architecture initialized:")
    print("  LSTM (in: 1, hidden: 64, layers: 2) -> Linear(32) -> Output(1)")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    print("\nTraining Model...")
    best_f1 = 0
    best_model_state = None
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        
        scheduler.step(val_f1)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = model.state_dict().copy()
            
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

    if best_model_state:
        model.load_state_dict(best_model_state)
        
    print("\nEvaluating on Test Set...")
    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(model, test_loader, criterion, DEVICE)
    
    print("\n" + "=" * 70)
    print("TEST SET RESULTS (LSTM - IAT)")
    print("=" * 70)
    print(f"Accuracy: {test_acc:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    print(f"Precision: {precision_score(test_labels, test_preds, zero_division=0):.4f}")
    print(f"Recall: {recall_score(test_labels, test_preds, zero_division=0):.4f}")
    print(f"MCC: {matthews_corrcoef(test_labels, test_preds):.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_labels, test_preds)
    print(cm)
    
    model_path = os.path.join(MODELS_DIR, '02_lstm_iat_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\n✓ Model saved to {model_path}")

if __name__ == '__main__':
    main()