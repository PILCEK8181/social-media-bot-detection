"""
LSTM Temporal Branch - Tweet Pattern Analysis

Author: xzacekp00 Patrik Žáček
Institution: BUT FIT (Brno University of Technology, Faculty of Information Technology)
Type: Bachelor's Thesis 2026
Topic: Detection of Fake Accounts on Social Media Networks

Description:
    BiLSTM classifier for bot detection based on tweet temporal patterns (inter-arrival times).
    Analyzes timing patterns of user tweets to identify bot-like behavior. Uses class-weighted
    loss to handle dataset imbalance. Part of the ensemble model combining metadata, content,
    and temporal analysis.

Features:
    - Bidirectional LSTM with 2 layers and 128 hidden units
    - Packed sequence processing for variable-length timelines
    - Class-weighted cross-entropy loss
    - Early stopping with patience=7
    - Output: Bot probability predictions

"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score, matthews_corrcoef
from pathlib import Path
import random
from utils.save_metrics import save_metrics

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = './data/twibot22'
TEMP_DIR = './temp'
MODELS_DIR = './models'

MAX_SEQ_LEN = 200
BATCH_SIZE = 256
EPOCHS = 40
LEARNING_RATE = 5e-4
HIDDEN_SIZE = 128
NUM_LAYERS = 2
PATIENCE = 7

SEED = random.randint(1, 10000)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

print(f"Device: {DEVICE}")
print(f"Seed: {SEED}")
print("=" * 70)


# Load timestamps and prepare IAT sequences
def load_and_prepare_iat_data():
    print("\nLoading data...")
    
    # labels and splits
    df_labels = pd.read_csv(os.path.join(DATA_DIR, 'label.csv'))
    df_split = pd.read_csv(os.path.join(DATA_DIR, 'split.csv'))
    
    label_map = {str(row['id']).replace('u', ''): (1 if row['label'] == 'bot' else 0) for _, row in df_labels.iterrows()}
    split_map = {str(row['id']).replace('u', ''): row['split'] for _, row in df_split.iterrows()}
    
    # account creation times
    print("  Loading account creation times...")
    user_creation = {}
    with open(os.path.join(DATA_DIR, 'user.json'), 'r', encoding='utf-8') as f:
        users = json.load(f)
        for u in users:
            uid = str(u.get('id', '')).replace('u', '')
            created_at = u.get('created_at')
            if uid in label_map and created_at:
                user_creation[uid] = pd.to_datetime(created_at, format='mixed', utc=True)

    # tweet times
    print("  Loading tweet timestamps...")
    df_tweets = pd.read_csv(os.path.join(TEMP_DIR, 'processed_timestamps.csv'))
    df_tweets['user_id'] = df_tweets['user_id'].astype(str)
    df_tweets['timestamp'] = pd.to_datetime(df_tweets['timestamp'], format='mixed', utc=True)
    
    # group tweets by user
    grouped = df_tweets.groupby('user_id')['timestamp'].apply(list).to_dict()
    del df_tweets
    
    # calc IAT sequences
    print("  Calculating IATs...")
    all_user_ids = list(label_map.keys())
    
    # Initialize tensors
    X = np.zeros((len(all_user_ids), MAX_SEQ_LEN), dtype=np.float32)
    lengths = np.ones(len(all_user_ids), dtype=np.int64)
    y = np.zeros(len(all_user_ids), dtype=np.float32)
    splits = []
    
    missing_creation = 0
    
    # Loop through users and build IAT sequences
    for idx, uid in enumerate(all_user_ids):
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
            timestamps = sorted(timestamps)
            diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() for i in range(1, len(timestamps))]
            diffs = diffs[:MAX_SEQ_LEN]
            
            # Log transform: log(1 + x)
            diffs_log = np.log1p(np.maximum(diffs, 0)) 
            
            X[idx, :len(diffs_log)] = diffs_log
            lengths[idx] = len(diffs_log)
            
    print(f"IAT tensor shape: {X.shape}")
    print(f"Users without creation date: {missing_creation}")
    
    train_mask = np.array([s == 'train' for s in splits])
    val_mask = np.array([s == 'val' for s in splits])
    test_mask = np.array([s == 'test' for s in splits])
    
    # Z-score normalize using training set statistics
    train_vals = []
    for i in np.where(train_mask)[0]:
        if lengths[i] > 1 or X[i, 0] != 0:
            train_vals.append(X[i, :lengths[i]])
    train_vals = np.concatenate(train_vals) if train_vals else np.array([0.0])
    
    iat_mean = train_vals.mean()
    iat_std = train_vals.std() + 1e-8
    
    for i in range(len(X)):
        if lengths[i] > 0:
            X[i, :lengths[i]] = (X[i, :lengths[i]] - iat_mean) / iat_std
    
    X_tensor = torch.tensor(X).unsqueeze(-1)
    lengths_tensor = torch.tensor(lengths)
    y_tensor = torch.tensor(y)
    
    train_loader = DataLoader(
        TensorDataset(X_tensor[train_mask], lengths_tensor[train_mask], y_tensor[train_mask]),
        batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(X_tensor[val_mask], lengths_tensor[val_mask], y_tensor[val_mask]),
        batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(
        TensorDataset(X_tensor[test_mask], lengths_tensor[test_mask], y_tensor[test_mask]),
        batch_size=BATCH_SIZE, shuffle=False)
    
    # class weights
    num_bots = y[train_mask].sum()
    num_humans = len(y[train_mask]) - num_bots
    pos_weight = torch.tensor([num_humans / (num_bots + 1e-5)]).to(DEVICE)
    
    print(f"Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
    print(f"pos_weight: {pos_weight.item():.4f}")
    
    return train_loader, val_loader, test_loader, pos_weight, all_user_ids, val_mask, test_mask, y

# BILSTM with attention for IAT sequences
class IAT_LSTM_Model(nn.Module):
    
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, bidirectional=True):
        super(IAT_LSTM_Model, self).__init__()
        
        self.num_directions = 2 if bidirectional else 1
        lstm_out_size = hidden_size * self.num_directions
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out_size),
            nn.Linear(lstm_out_size, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1)
        )
    
    def forward(self, x, lengths):
        lengths_clamped = lengths.clamp(min=1).cpu()
        packed = pack_padded_sequence(x, lengths_clamped, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        
        max_len = output.size(1)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        
        attn_scores = self.attention(output).squeeze(-1)
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)
        attn_weights = attn_weights.masked_fill(attn_weights.isnan(), 1.0 / max_len)
        
        context = torch.bmm(attn_weights.unsqueeze(1), output).squeeze(1)
        return self.classifier(context).squeeze(-1)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X_batch, len_batch, y_batch in loader:
        X_batch, len_batch, y_batch = X_batch.to(device), len_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        logits = model(X_batch, len_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        
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
        for X_batch, len_batch, y_batch in loader:
            X_batch, len_batch, y_batch = X_batch.to(device), len_batch.to(device), y_batch.to(device)
            
            logits = model(X_batch, len_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item()
            
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
    print("Bot Detection Model Training - BiLSTM Inter-Arrival Times")
    print("=" * 70)
    
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load data
    train_loader, val_loader, test_loader, pos_weight, all_user_ids, val_mask, test_mask, y_labels = load_and_prepare_iat_data()
    
    # Model initialization
    model = IAT_LSTM_Model(
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        bidirectional=True
    ).to(DEVICE)
    
    print(f"\nModel architecture:")
    print(f"BiLSTM (hidden: {HIDDEN_SIZE}, layers: {NUM_LAYERS}) -> Attention -> 64 -> 1")
    
    # Training setup
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    print("\nTraining model...")
    best_f1 = 0
    best_model_state = None
    epochs_no_improve = 0
    
    # Training loop with early stopping
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        
        scheduler.step(epoch)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
        
        if epochs_no_improve >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)
    
    print(f"Training completed. Best F1: {best_f1:.4f}")
    
    # evaluate
    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(model, test_loader, criterion, DEVICE)
    
    print("\n" + "=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)

    prec = precision_score(test_labels, test_preds, zero_division=0)
    rec = recall_score(test_labels, test_preds, zero_division=0)
    mcc = matthews_corrcoef(test_labels, test_preds)

    print(f"Accuracy:  {test_acc:.4f}")
    print(f"F1 Score:  {test_f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"MCC:       {mcc:.4f}")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(test_labels, test_preds))
    
    # save model
    model_path = os.path.join(MODELS_DIR, '02_lstm.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    # Save predictions for ensemble
    print("\nExtracting probabilities for ensemble...")
    val_uids = np.array(all_user_ids)[val_mask]
    test_uids = np.array(all_user_ids)[test_mask]
    
    def extract_probs(loader):
        model.eval()
        probs = []
        with torch.no_grad():
            for X_batch, len_batch, _ in loader:
                logits = model(X_batch.to(DEVICE), len_batch.to(DEVICE))
                probs.extend(torch.sigmoid(logits).cpu().numpy())
        return probs

    val_probs = extract_probs(val_loader)
    test_probs = extract_probs(test_loader)
    
    df_preds = pd.concat([
        pd.DataFrame({'user_id': val_uids, 'prob_lstm': val_probs, 'split': 'val', 'label': y_labels[val_mask]}),
        pd.DataFrame({'user_id': test_uids, 'prob_lstm': test_probs, 'split': 'test', 'label': y_labels[test_mask]})
    ])
    df_preds.to_csv(os.path.join(TEMP_DIR, 'predictions/preds_lstm.csv'), index=False)
    print("  Saved to predictions/preds_lstm.csv")

    save_metrics(
        filename=os.path.basename(__file__),
        seed=SEED, 
        acc=test_acc,
        prec=prec,
        recall=rec,
        f1=test_f1,
        mcc=mcc,
        note="LSTM model 200 trained on IAT features only // raw, Final run"
    )


if __name__ == '__main__':
    main()