

import os
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
from utils.save_metrics import save_metrics
import random

from torch.utils.data import WeightedRandomSampler

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEMP_DIR = './temp'
MODELS_DIR = './models'
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-4

SEED = random.randint(1, 10000)  # Generate a random seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"Device: {DEVICE}")
print(f"Loading embeddings from: {TEMP_DIR}")
print("=" * 70)


def load_embeddings() -> tuple:
    print("\nLoading embeddings...")
    
    # tweet embeddings
    tweet_file = os.path.join(TEMP_DIR, 'roberta_embeddings.pt')
    print(f"Loading tweets: {tweet_file}")
    tweet_data = torch.load(tweet_file, map_location='cpu', weights_only=False)
    tweet_embeddings = tweet_data['embeddings'].float()
    tweet_user_ids = tweet_data['user_ids']
    tweet_labels = tweet_data['labels'].long()
    tweet_splits = tweet_data['splits']
    print(f"Tweet embeddings shape: {tweet_embeddings.shape}")
    
    #  bio embeddings
    bio_file = os.path.join(TEMP_DIR, 'roberta_bio_embeddings.pt')
    print(f"  Loading bios: {bio_file}")
    bio_data = torch.load(bio_file, map_location='cpu', weights_only=False)
    bio_embeddings = bio_data['embeddings'].float()
    bio_user_ids = bio_data['user_ids']
    bio_labels = bio_data['labels'].long()
    bio_splits = bio_data['splits']
    print(f"Bio embeddings shape: {bio_embeddings.shape}")
    
    return (tweet_embeddings, tweet_user_ids, tweet_labels, tweet_splits,
            bio_embeddings, bio_user_ids, bio_labels, bio_splits)


def align_embeddings(tweet_emb, tweet_ids, tweet_labels, tweet_splits,
                      bio_emb, bio_ids, bio_labels, bio_splits) -> tuple:

    print("\nAligning embeddings...")
    
    # convert to sets for quick intersection
    tweet_set = set(tweet_ids)
    bio_set = set(bio_ids)
    common_ids = tweet_set & bio_set
    
    print(f"  Tweet users: {len(tweet_set)}")
    print(f"  Bio users: {len(bio_set)}")
    print(f"  Common users: {len(common_ids)}")
    
    # mapping
    tweet_id_to_idx = {uid: idx for idx, uid in enumerate(tweet_ids)}
    bio_id_to_idx = {uid: idx for idx, uid in enumerate(bio_ids)}
    
    # get indexes, labels, splits for common users
    tweet_indices = []
    bio_indices = []
    alignment_labels = []
    alignment_splits = []
    alignment_user_ids = []
    
    for uid in common_ids:
        if uid in tweet_id_to_idx and uid in bio_id_to_idx:
            # check if labels match
            t_label = tweet_labels[tweet_id_to_idx[uid]].item()
            b_label = bio_labels[bio_id_to_idx[uid]].item()
            
            if t_label == b_label:  # match
                tweet_indices.append(tweet_id_to_idx[uid])
                bio_indices.append(bio_id_to_idx[uid])
                alignment_labels.append(t_label)
                alignment_splits.append(tweet_splits[tweet_id_to_idx[uid]])
                alignment_user_ids.append(uid)
    
    tweet_indices = torch.tensor(tweet_indices)
    bio_indices = torch.tensor(bio_indices)
    
    # alligned embeddings
    aligned_tweet_emb = tweet_emb[tweet_indices]
    aligned_bio_emb = bio_emb[bio_indices]
    
    # concat
    combined_emb = torch.cat([aligned_tweet_emb, aligned_bio_emb], dim=1)
    
    print(f"Aligned embeddings shape: {combined_emb.shape}")
    print(f"Labels distribution: {np.bincount(alignment_labels)}")
    
    return (combined_emb, torch.tensor(alignment_labels, dtype=torch.long),
            alignment_splits, alignment_user_ids)


def create_dataloaders(embeddings, labels, splits, batch_size=BATCH_SIZE):
    print("\nCreating dataloaders...")
    
    # split indexes
    train_mask = np.array([s == 'train' for s in splits])
    val_mask = np.array([s == 'val' for s in splits])
    test_mask = np.array([s == 'test' for s in splits])
    
    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]
    test_indices = np.where(test_mask)[0]
    
    print(f"  Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    # datasets
    train_dataset = TensorDataset(embeddings[train_indices], labels[train_indices])
    val_dataset = TensorDataset(embeddings[val_indices], labels[val_indices])
    test_dataset = TensorDataset(embeddings[test_indices], labels[test_indices])
    
    
    # --- SEMPLER SETTING ---
    
    # Calculated class distribution in the training set
    train_labels_np = labels[train_indices].numpy()
    class_counts = np.bincount(train_labels_np)
    
    # Set the weight for each class: inverse of frequency
    class_weights = 1.0 / class_counts
    
    # Set the weight for each sample in the training set based on its class
    sample_weights = [class_weights[label] for label in train_labels_np]
    
    # Create the WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), # 
        replacement=True # Allows duplicates, which is necessary for oversampling
    )
    
    # create dataloaders

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    # No sampling for validation and test sets, we want to evaluate on the original distribution
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # --- SEMPLER SETTING ---


    print(f"Dataloaders created (Balanced Sampling Active)")
    
    print(f"Dataloaders created")
    
    return train_loader, val_loader, test_loader

# Classifcation model definition
class BotDetectionModel(nn.Module):
    
    def __init__(self, input_dim: int):
        super(BotDetectionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        # self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 256)
        # self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(256, 128)
        # self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.2)
        
        self.output = nn.Linear(128, 2)
    
    def forward(self, x):
        x = self.fc1(x)
        # x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        # x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        # x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        
        x = self.output(x)
        return x


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for embeddings, labels in loader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for embeddings, labels in loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    return total_loss / len(loader), accuracy, f1, all_preds, all_labels


def train_model(model, train_loader, val_loader, device, epochs=EPOCHS, lr=LEARNING_RATE):

    print("\nTraining model...")
    
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    best_f1 = 0
    best_model_state = None
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_f1)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch 
            # best_model_state = model.state_dict().copy()
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
    
    # load the best model state
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    print(f"Training completed. Best F1: {best_f1:.4f} saved from epoch {best_epoch+1}")
    return model


def main():
    print("\n" + "=" * 70)
    print("Bot Detection Model Training - Tweets + Bio RoBERTa Embeddings - Oversample")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # setup directories
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    
    # load embeddings
    (tweet_emb, tweet_ids, tweet_labels, tweet_splits,
     bio_emb, bio_ids, bio_labels, bio_splits) = load_embeddings()
    
    # align embeddings
    combined_emb, labels, splits, user_ids = align_embeddings(
        tweet_emb, tweet_ids, tweet_labels, tweet_splits,
        bio_emb, bio_ids, bio_labels, bio_splits
    )
    
    # dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(combined_emb, labels, splits)
    

    input_dim = combined_emb.shape[1]
    model = BotDetectionModel(input_dim).to(DEVICE)
    print(f"\nModel architecture:")
    print(f"  Input dim: {input_dim} (768 tweet + 768 bio)")
    print(f"  Layers: 768 → 512 → 256 → 128 → 2")
    
    model = train_model(model, train_loader, val_loader, DEVICE)
    
    # Evaluation on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, nn.CrossEntropyLoss(), DEVICE
    )
    
    print("\n" + "=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)
    print(f"Accuracy: {test_acc:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    print(f"Precision: {precision_score(test_labels, test_preds):.4f}")
    print(f"Recall: {recall_score(test_labels, test_preds):.4f}")
    print(f"Matthews Corrcoef: {matthews_corrcoef(test_labels, test_preds):.4f}")                 
    print("\nDetailed Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=['Human', 'Bot']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_labels, test_preds)
    print(cm)
    print(f"True Negatives: {cm[0, 0]}, False Positives: {cm[0, 1]}")
    print(f"False Negatives: {cm[1, 0]}, True Positives: {cm[1, 1]}")
    
    # Save model
    model_path = os.path.join(MODELS_DIR, '03roberta_oversample.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': 'BotDetectionModel',
        'input_dim': input_dim,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'timestamp': datetime.now().isoformat()
    }, model_path)
    print(f"\nModel saved to {model_path}")
    
    print("\n" + "=" * 70)


    # PROBABILITIES FOR ENSEMBLE
    print("\nExtracting Probabilities for Ensemble...")
    # get ids for val and test
    val_indices = np.where(np.array(splits) == 'val')[0]
    test_indices = np.where(np.array(splits) == 'test')[0]
    val_uids = [user_ids[i] for i in val_indices]
    test_uids = [user_ids[i] for i in test_indices]
    
    def extract_probs(loader):
        model.eval()
        probs = []
        with torch.no_grad():
            for emb, _ in loader:
                logits = model(emb.to(DEVICE))
                # softmax for positive class (bot)
                p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                probs.extend(p)
        return probs

    val_probs = extract_probs(val_loader)
    test_probs = extract_probs(test_loader)
    
    df_val = pd.DataFrame({'user_id': val_uids, 'prob_roberta': val_probs, 'split': 'val', 'label': [labels[i].item() for i in val_indices]})
    df_test = pd.DataFrame({'user_id': test_uids, 'prob_roberta': test_probs, 'split': 'test', 'label': [labels[i].item() for i in test_indices]})
    
    pd.concat([df_val, df_test]).to_csv(os.path.join(TEMP_DIR, 'predictions/preds_roberta_oversample.csv'), index=False)
    print(" RoBERTa probabilities saved to predictions/preds_roberta_oversample.csv")

    current_script_name = os.path.basename(__file__) 
    acc = test_acc
    prec = precision_score(test_labels, test_preds)
    recall = recall_score(test_labels, test_preds)
    f1 = test_f1
    mcc = matthews_corrcoef(test_labels, test_preds)

    save_metrics(
        filename=current_script_name,
        seed=SEED, 
        acc=acc,
        prec=prec,
        recall=recall,
        f1=f1,
        mcc=mcc,
        note="OVERSAMPLE - 1e-4"
    )


if __name__ == '__main__':
    main()
