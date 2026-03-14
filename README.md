# Social Media Bot Detection

A machine learning project for detecting bot accounts on Twitter using multiple approaches: Random Forest classification, LSTM temporal analysis, and RoBERTa-based embeddings.

**Dataset:** TwiBot-22 from the NeurIPS 2022 paper "Towards Graph-Based Twitter Bot Detection"

---

## Setup

### Requirements
- Python 3.11+
- CUDA-capable GPU (recommended for deep learning models)

### Installation

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
├── src/                              # Training scripts and detection models
│   └── utils/                        # Utility functions for metrics logging
│
├── notebooks/                         # Jupyter notebooks for analysis and exploration
│
├── models/                            # Trained model weights (not tracked in git)
│
├── data/                              # Dataset (not tracked in git)
│   └── twibot22/                      # TwiBot-22 dataset structure
│       ├── user.json                 # User profile metadata
│       ├── tweet.json                # Tweet data
│       ├── label.csv                 # Bot/Human labels (user_id, label)
│       ├── split.csv                 # Train/val/test split assignments
│       └── edge.csv                  # Social graph edges (optional for graph models)
│
├── demo/                              # Demo profiles for testing
│
├── output/                            # Detection results and predictions (JSON)
│
├── results/                           # Model evaluation metrics and analysis
│
├── doc/                               # Documentation and project notes
│
├── test/                              # Test utilities and validation scripts
│
├── run.sh                             # PBS job submission script for metacenter training
├── requirements.txt                  # Python dependencies
├── requirements_meta.txt             # Additional metacenter dependencies
└── LICENSE                           # Project license
```

---

## Dataset Structure

The TwiBot-22 dataset contains Twitter user data:

```
data/twibot22/
├── user.json          # User profiles (IDs, features, metadata)
├── tweet.json         # Tweet data (timestamps, text, engagement)
├── label.csv          # Binary labels: 'bot' or 'human'
├── split.csv          # Data split: 'train', 'val', 'test'
└── edge.csv           # (Optional) Social graph edges for graph models
```

**Paper Reference:** 
- Feng et al. (NeurIPS 2022) - "Towards Graph-Based Twitter Bot Detection"
- https://github.com/BunsenLabs/TwiBot-22

---

## Models & Approaches

### 1. **Random Forest**
- **Input:** User profile features (account age, followers, verification, etc.)
- **Output:** Bot probability
- **Key Advantage:** Fast inference, interpretable feature importance

### 2. **LSTM**
- **Input:** Tweet inter-arrival times (IAT) - temporal sequences
- **Architecture:** LSTM layers → fully connected layers
- **Output:** Bot likelihood
- **Key Insight:** Bots exhibit different tweeting patterns (more regular timing)

### 3. **RoBERTa Embeddings**
- **Input:** Tweet + bio text embeddings (768-dim each)
- **Architecture:** Dense layers for classification
- **Variants:** Baseline, weighted loss, oversampling, heterogeneous
- **Key Advantage:** Captures semantic patterns in bot behavior

<!-- ### 4. **Graph-Based Models**
- **Input:** User profiles + social network graph
- **Method:** Relational Graph Convolutional Networks
- **Advantage:** Leverages social connections for detection -->

### 5. **Ensemble/Meta-Classifier**
- Combines predictions from multiple models
- Weighted ensemble approach

---

## Quick Start

### Train Models

```bash
# Train models in src/ directory
python src/<model_script>.py
```

### Preprocess Data

```bash
# Generate embeddings and process timestamps via preprocessing scripts in src/
python src/<preprocess_script>.py
```

### Run Detection on New Profiles

```bash
# Detection scripts available in src/ directory
python src/detect.py <username>
```

---

## Key Features

- **Multi-modal approaches:** Combines user features, temporal patterns, and text embeddings
- **Balanced evaluation:** Handles class imbalance with weighted loss and oversampling
- **Efficient preprocessing:** Batched embedding generation with checkpointing
- **Production-ready:** Detection scripts for real-world usage
- **Comprehensive analysis:** Jupyter notebooks for model exploration

---

## Dependencies

- **Deep Learning:** torch, transformers (RoBERTa)
- **ML/Statistics:** scikit-learn, numpy, pandas, scipy
- **Visualization:** matplotlib, seaborn
- **Data Processing:** tqdm, ijson, beautifulsoup4
- **Utilities:** joblib, argparse

---

## Results

Detailed model evaluation metrics are available in the `results/` directory including accuracy, precision, recall, F1-scores, ROC-AUC scores, confusion matrices, and class-wise performance analysis.

---

## Analysis & Exploration

Jupyter notebooks in `notebooks/` directory provide detailed exploration and analysis of model performance and results.

---

## Notes

- The project uses HPC (PBS job scheduler) for training large models
- GPU required for RoBERTa embedding generation (~24h for full dataset)
- Models are periodically checkpointed during long preprocessing runs
- Demo profiles included in `demo/` for quick testing

---

## Author & License

See LICENSE file for project licensing details.
**Thesis Project**: Social Media Bot Detection using Machine Learning

