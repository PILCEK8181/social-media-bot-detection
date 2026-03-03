import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             roc_curve, auc)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import seaborn as sns
import warnings


import ijson
import csv
import os
import gc

from torch.nn.utils.rnn import pad_sequence

from sklearn.metrics import (confusion_matrix, roc_curve, auc, 
                             accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)



warnings.filterwarnings('ignore')

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Config
SCRATCH_DIR = "./temp"
os.makedirs(SCRATCH_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(SCRATCH_DIR, "processed_timestamps.csv")
DATA_DIR = './data/twibot22/'

print(f"Running ijson extractor, output: {OUTPUT_FILE}")
print("Loading labels...")
df_labels = pd.read_csv(os.path.join(DATA_DIR, 'label.csv'))

# Remove 'u' from user id value
valid_users = set(df_labels['id'].astype(str).str.replace('u', '', regex=False))

print(f"number of users: {len(valid_users)}.")

del df_labels
gc.collect()

# CSV setup 
with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(['user_id', 'timestamp'])

    # Iterate over the tweet_x.json files
    for i in range(9):
        file_path = os.path.join(DATA_DIR, f'tweet_{i}.json')
        print(f"\nWorking with {file_path}...")
        
        try:
            with open(file_path, 'rb') as f_in:
                parser = ijson.items(f_in, 'item')
                
                count = 0
                saved = 0
                
                for tweet in parser:
                    # convert str id to int
                    uid = str(tweet.get('author_id', ''))
                    
                    # Create rows
                    if uid in valid_users:
                        created_at = tweet.get('created_at')
                        if created_at:
                            writer.writerow([uid, created_at])
                            saved += 1
                    
                    count += 1
                    if count % 2000000 == 0:
                        # Control prints
                        print(f"   -> finished {count} tweets... (saved: {saved})")
                        
            print(f" file {i} Fil=nished. Saved in total: {saved}")
            
        except FileNotFoundError:
            print(f" Filer {file_path} does not exitst.")
        except Exception as e:
            print(f"Error in file {file_path}: {e}")

print(f"\n Done, file: {OUTPUT_FILE}")

