# Reproduction Guide - Social Media Bot Detection

This guide provides step-by-step instructions to reproduce all results from the Social Media Bot Detection project.

**Dataset:** TwiBot-22 from NeurIPS 2022 ("Towards Graph-Based Twitter Bot Detection")
**Reference Date:** December 31, 2022
**Python Version:** 3.11+

> **Note:** All files are configured to replicate the paper's final results. However, variance may exist due to differences in hardware, random seeds, or data splits. The code can be easily updated to explore different configurations and obtain alternative results by modifying hyperparameters, feature selections, or model architectures in the respective training scripts. Additionally, file paths in the scripts may need to be adjusted according to your local directory structure and setup.

---

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Data Preparation](#data-preparation)
3. [Model Training](#model-training)
   - [Random Forest (01_rf.py)](#1-random-forest-metadata)
   - [LSTM (02_lstm.py)](#2-lstm-temporal-analysis)
   - [RoBERTa (03_roberta_oversample.py)](#3-roberta-content-analysis)
4. [Ensemble Meta-Classifier (00_meta_classifier.py)](#ensemble-meta-classifier)
5. [Demo/Testing (bot_detector.py)](#demotesting-with-demo-profiles)
6. [Expected Results](#expected-results)
7. [Speed Guide](#speed-guide)

---

## Environment Setup

### 1. Clone and Navigate
```bash
cd /path/to/social-media-bot-detection
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### 5. Create Required Directories
```bash
mkdir -p data/twibot22
mkdir -p models
mkdir -p temp/predictions
mkdir -p output
mkdir -p results
```

---

## Data Preparation

### Step 1: Obtain TwiBot-22 Dataset

The TwiBot-22 dataset must be acquired from the original authors:
- **Source:** https://github.com/LuoUndergradXJTU/TwiBot-22
- **Paper:** Feng et al. (NeurIPS 2022) - "TwiBot-22: Towards Graph-Based Twitter Bot Detection"

### Step 2: Dataset Structure

Once obtained, place the dataset in `data/twibot22/` with this structure:

```
data/twibot22/
├── user.json          # User profiles (ID, features, timestamps, metadata)
├── tweet0.json        # Tweet data 
├── tweet1.json        # Tweet data 
├── tweet2.json        # Tweet data 
├── ...                # More tweet files (tweet3-tweet8)
├── label.csv          # Binary labels (columns: id, label with values 'bot' or 'human')
├── split.csv          # Data splits (columns: id, split with values 'train', 'val', 'test')
└── edge.csv           # (Optional, Unused) Social graph edges
```

### Step 3: Verify Data Files

```bash
# Check file sizes and line counts
wc -l data/twibot22/*.csv
ls -lh data/twibot22/*.json

# Sample data structure
head -c 500 data/twibot22/user.json
```

---

## Model Training

### **Phase 1: Preprocessing**

Before training models, preprocess the data:

#### 1a. LSTM - Prepare Timestamp Data
```bash
python src/preprocess_lstm_timestamps.py
```

**Output:** `temp/processed_timestamps.csv`
- Contains timestamps of user tweets and creation dates

#### 1b. RoBERTa - Generate Tweet Embeddings
```bash
python src/preprocess_roberta_embeddings.py
```

**Output:** `temp/roberta_embeddings.pt`
- 768-dimensional embeddings for tweets
- Uses RoBERTa-base tokenizer and encoder
- Batch size: 128
- Device: CUDA if available, else CPU (not recommended, going to take forever)

#### 1c. RoBERTa - Generate Bio Embeddings
```bash
python src/preprocess_roberta_bio_embeddings.py
```

**Output:** `temp/roberta_bio_embeddings.pt`
- 768-dimensional embeddings for user bios
- Later concatenated with tweet embeddings for final input

---

### **Phase 2: Train Individual Models**

#### **1. Random Forest (Metadata Branch)**

```bash
python src/01_rf.py
```

**Configuration:**
- Input: 24 user profile features (option to pick from the feature sets in prepare_features function)
- Model: RandomForestClassifier (balanced class weights)
- Output: `models/01_rf.joblib`
- Predictions: `temp/predictions/preds_rf.csv`


---

#### **2. LSTM (Temporal Branch)**

```bash
python src/02_lstm.py
```

**Configuration:**
- Input: Inter-arrival times (IAT) from tweet timestamps
- Architecture: BiLSTM with 64 hidden units → 2 FC layers
- Device: CUDA if available
- Output: `models/02_lstm.pth`
- Predictions: `temp/predictions/preds_lstm.csv`


**Note:** Base model (unweighted) is unusable due to severe class imbalance.

---

#### **3. RoBERTa (Content Branch)**

```bash
python src/03_roberta_oversample.py
```

**Configuration:**
- Input: Tweet embeddings (768-dim) + bio embeddings (768-dim) = 1536-dim
- Architecture: Dense layers (1536 → 512 → 256 → 128 → 2)
- Device: CUDA if available
- Output: `models/03roberta_oversample.pth`
- Predictions: `temp/predictions/preds_roberta_oversample.csv`

**Data Preparation:**
- Tweet sampling: Top 20 tweets per user
- Missing tweets: Padding with zero vectors
- Embedding dimension: 768 (RoBERTa-base)

**Alternative Variants Available:**
- `03_roberta.py` - Base model (unweighted)
- `03_roberta_weighted.py` - Class weights approach

---

### **Phase 3: Ensemble Meta-Classifier**

After all three models are trained, stack their predictions:

```bash
python src/00_meta_classifier.py
```

**Configuration:**
- Input: Probability predictions from:
  - RF: `temp/predictions/preds_rf.csv`
  - LSTM: `temp/predictions/preds_lstm.csv` (unused in final ensemble due to poor performance)
  - RoBERTa (Oversampled): `temp/predictions/preds_roberta_oversample.csv`
- Meta-learner: Logistic Regression 
- Output: `models/00_meta_classifier.joblib`

**Expected Ensemble Metrics:**

| Metric | Value |
|--------|-------|
| Accuracy | ~0.7755 |
| Precision | ~0.6722 |
| Recall | ~0.4633 |
| F1-Score | ~0.5486 |
| MCC | ~0.4182 |

**Note:** Results may differ slightly, primarily due to RoBERTa training variance (standard deviation details are provided in the paper).




