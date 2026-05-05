
# Reproduction Guide - Social Media Bot Detection



Patrik Žáček - xzacekp00
BUT FIT (Brno University of Technology, Faculty of Information Technology) 
Bachelor's Thesis 2026 - Detection of Fake Accounts on Social Media Networks 



See `results/` and `doc/` folders for detailed metrics, ablation studies, and technical documentation.

---

This guide provides step-by-step instructions to reproduce all results from the Social Media Bot Detection project.

**Dataset:** TwiBot-22 from NeurIPS 2022 ("Towards Graph-Based Twitter Bot Detection")


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
├── label.csv          # Binary labels
├── split.csv          # Data splits 
└── edge.csv           # (Optional, Unused) Social graph edges
```


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
- Input: 24 user profile features (option to modify different feature sets)
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
  - LSTM: `temp/predictions/preds_lstm.csv` (not used in final ensemble)
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

---

## Demo/Testing with Demo Profiles

#### Run Detection on Demo Accounts

Install Playwright browsers:

```bash
playwright install
```
Run the bot detector in demo mode:

```bash
python src/bot_detector.py --mode demo --target Charles_Leclerc
```

**Note:** The username is case-sensitive and must match how it is saved in the demo folder (only in demo mode).

**Process:**
1. Loads all preprocessed models:
   - `models/01_rf.joblib` (RF classifier)
   - `models/03roberta_oversample.pth` (RoBERTa model)
   - `models/00_meta_classifier.joblib` (Ensemble stacker)
   
2. Processes demo profiles from `demo/` directory:
   - Extracts features from `profile_*.json` files
   - Loads tweet data from `tweets_*.json` files
   
3. Generates predictions:
   - RF probability
   - RoBERTa probability
   - Ensemble probability
   - Final prediction (bot/human)

4. Outputs results to `output/bot_detection_*_timestamp.json`

#### Analyze Results

Example output structure:
```json
{
  "username": "Charles_leclerc",
  "prediction": "HUMAN",
  "probability": 0.07318698147537055,
  "threshold": 0.7,
  "modality_scores": {
    "metadata": 0.009407240508423429,
    "text": 0.05896920710802078
  },
  "timestamp": "2026-05-01T12:36:05.529836+00:00",
  "display_name": "Charles Leclerc",
  "followers": 3798437,
  "following": 188,
  "tweets": 2406,
  "verified": true
}
```


---

### Key Findings

1. **Random Forest (Metadata):** Strong single-modality performance
   - Unweighted: 76.04% accuracy, 76.54% precision, 26.22% recall
   - Balanced (cost-sensitive): 70.24% accuracy, 49.57% precision, 61.60% recall
   - Top-15 Gini features achieve 70.77% accuracy with minimal feature loss
   - Metadata features are strong indicators of bot behavior
   
2. **BiLSTM (Temporal):** Limited effectiveness alone
   - Base model: 70.6% accuracy but 0% precision/recall (essentially non-predictive)
   - Class-weighted: 54.0% accuracy, 37.0% precision, 79.5% recall
   - Temporal patterns alone insufficient for classification without class balancing
   - Better at detecting bots when class-weighted (high recall) but high false positive rate
   
3. **RoBERTa (Content):** Moderate balanced performance with oversampling
   - Oversampling strategy: 67.2% accuracy, 46.4% precision, 72.3% recall (mean over 10 runs)
   - Class weighting: 67.0% accuracy, 46.2% precision, 72.2% recall
   - Text embeddings capture semantic bot behavior patterns
   - Requires class imbalance mitigation for meaningful recall
   
4. **Ensemble Meta-Classifier (Logistic Regression):** Best overall performance
   - No class weights: 77.55% accuracy, 67.22% precision, 46.33% recall (recommended)
   - Balanced class weights: 70.47% accuracy, 49.89% precision, 69.56% recall (precision-recall trade-off)
   - Combines RF (metadata accuracy) with RoBERTa (content semantics) effectively
   - Meta-learning leverages complementary modality strengths for robust detection

---

## Reproducing Specific Results

### To Reproduce Only Feature Ablation Study (Random Forest)

The ablation study results in `results/metrics.md` can be reproduced by modifying `src/01_rf.py`:

```python
# In 01_rf.py, modify the feature selection section:

ratios = ['follower_following_ratio', 'tweets_per_day', 'followers_per_tweet', 'listed_followers_ratio']

text_metrics = ['username_length', 'name_length', 'description_length', 'name_digit_count', 
               'name_special_char_count', 'username_digit_count', 'username_special_char_count']

flags = ['has_mention', 'has_hashtag', 'has_url_in_description', 'has_location', 'has_url_field',
         'verified', 'protected', 'default_profile_image', 'account_age_days']

top15 = ['log_followers_count', 'log_tweet_count', 'follower_following_ratio', 'description_length', 
         'log_listed_count', 'tweets_per_day', 'log_following_count', 'account_age_days', 'listed_followers_ratio', 
         'followers_per_tweet', 'name_length', 'verified', 'username_length', 'has_url_field', 'name_special_char_count']

top10 = ['log_followers_count', 'log_tweet_count', 'follower_following_ratio', 'description_length', 'log_listed_count', 
         'tweets_per_day', 'log_following_count', 'account_age_days', 'listed_followers_ratio', 'followers_per_tweet']
# Then retrain and evaluate
```

### To Reproduce RoBERTa Variants

Three versions are provided, each with different class imbalance strategies:

```bash
# Base model (unweighted)
python src/03_roberta.py

# Weighted loss (class_weight='balanced')
python src/03_roberta_weighted.py

# Oversampling (RECOMMENDED)
python src/03_roberta_oversample.py
```

---

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size or use CPU
```python
DEVICE = torch.device('cpu')  # Force CPU
BATCH_SIZE = 64  # Reduce batch size
```

### Issue: "Data files not found"
**Solution:** Verify data structure
```bash
# Check that files exist
ls -l data/twibot22/
# Expected: user.json, tweet.json, label.csv, split.csv
```

---

## File Dependencies

```
Data Files:
├── data/twibot22/
│   ├── user.json → 01_rf.py, preprocess_*.py
│   ├── tweet0.json - tweet8.json → 02_lstm.py, 03_roberta*.py, preprocess_*.py
│   ├── label.csv → All models
│   └── split.csv → All models

Preprocessing:
├── temp/processed_timestamps.csv ← preprocess_lstm_timestamps.py
├── temp/roberta_embeddings.pt ← preprocess_roberta_embeddings.py
└── temp/roberta_bio_embeddings.pt ← preprocess_roberta_bio_embeddings.py

Models:
├── models/01_rf.joblib ← 01_rf.py
├── models/02_lstm.pth ← 02_lstm.py
├── models/03roberta_oversample.pth ← 03_roberta_oversample.py
└── models/00_meta_classifier.joblib ← 00_meta_classifier.py

Predictions:
├── temp/predictions/preds_rf.csv ← 01_rf.py
├── temp/predictions/preds_lstm.csv ← 02_lstm.py
├── temp/predictions/preds_roberta_oversample.csv ← 03_roberta_oversample.py
└── Final predictions → 00_meta_classifier.py

Demo Testing:
├── demo/profile_*.json → bot_detector.py
├── demo/tweets_*.json → bot_detector.py
└── output/bot_detection_*.json ← bot_detector.py
```

---

## Execution Order Summary

**To fully reproduce all results:**

```bash
# 1. Preprocessing (one-time setup)
python src/preprocess_lstm_timestamps.py
python src/preprocess_roberta_embeddings.py
python src/preprocess_roberta_bio_embeddings.py

# 2. Train individual models (can run in parallel)
python src/01_rf.py
python src/02_lstm.py
python src/03_roberta_oversample.py

# 3. Create ensemble (requires all three models)
python src/00_meta_classifier.py

# 4. Test on demo data
python src/bot_detector.py

# 5. Review results
cat results/results.csv
ls -t output/bot_detection_*.json | head -5
```

**Note:** This is the main pipeline for reproducing the final results. Other files like `00_rf_classifier.py` and `detect_*.py` are used for ablation studies and individual model testing and are present for completeness, but the above steps will reproduce the core results as presented in the paper.

**Estimated Total Runtime:**
- Preprocessing: ~30 hours (GPU dependent)
- Model Training: ~2-4 hours (depends on GPU availability)
- Ensemble: ~20 minutes
- Demo Testing: <1 minute

---

# Speed Guide

Quick reference for running the project in the correct order.

### Quick Start (Assuming data is prepared)

If you already have the TwiBot-22 dataset and just want to run everything:

```bash
# Step 1: Setup environment (one-time only)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Step 2: Preprocess data (one-time only)
python src/preprocess_lstm_timestamps.py
python src/preprocess_roberta_embeddings.py
python src/preprocess_roberta_bio_embeddings.py

# Step 3: Train models (can run in parallel on separate terminals)
python src/01_rf.py
python src/02_lstm.py
python src/03_roberta_oversample.py

# Step 4: Create ensemble meta-classifier (requires Step 3 complete)
python src/00_meta_classifier.py

```

### Using run.sh (Automated Batch Script)

Alternatively, you can use the provided `run.sh` batch script, which runs all commands automatically. However, you'll need to tweak it for your environment:

**Setup:**


`run.sh` is configured for HPC cluster submission (PBS directives at the top). For local execution:
- Remove PBS directives (`#PBS` lines) or modify them
- Comment out `module purge` and `module add python/3.11.11-gcc-10.2.1`
- Adjust paths as needed


---



