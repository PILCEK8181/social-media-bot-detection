# Directory Structure Overview

This document provides a brief overview of the project directory structure for the Social Media Bot Detection system.

## Root Level Files
- `LICENSE` - Project license file
- `README.md` - Main project documentation focused on reproduction of results
- `requirements.txt` - Python dependencies
- `run.sh` - Shell script for running the project

## demo/
Contains sample data files for demonstration purposes:
- User profile JSON files (e.g., `profile_Charles_Leclerc.json`, `profile_Cristiano.json`)
- Tweet data JSON files (e.g., `tweets_Charles_Leclerc.json`, `tweets_NASA.json`)

## doc/
Documentation files:
- `bot_detector.md` - Bot detection system documentation
- `directory.md` - This file (directory structure overview)
- `thesis.pdf` - Paper/thesis document detailing the research and implementation
- `scraper.md` - Data scraping documentation
- `utils.md` - Utility functions documentation

## models/
Trained machine learning models:
- `01_rf.joblib` - Random Forest model
- `03roberta_oversample.pth` - RoBERTa model (oversampled)

## output/
Bot detection results and outputs:
- JSON files with detection results for various accounts

## results/
Evaluation results:
- `metrics.md` - Performance metrics
- `results.csv` - Tabular results data

## src/
Source code directory containing all Python scripts:
- `00_meta_classifier.py` - Meta-classifier implementation
- `00_rf_classifier.py` - Random Forest classifier
- `01_rf.py` - Random Forest training script
- `02_lstm.py` - LSTM model implementation
- `03_roberta*.py` - RoBERTa model variants
- `bot_detector.py` - Main bot detection script
- `detect_rf.py`, `detect_roberta.py` - Detection scripts for specific models
- `preprocess_*.py` - Data preprocessing scripts
- `scrape.py` - Data scraping script
- `utils/` - Utility modules:
  - `evaluation.py` - Evaluation functions
  - `get_tweets.py` - Tweet retrieval utilities
  - `get_user.py` - User data utilities
  - `save_metrics.py` - Metrics saving functions



## test/
Testing scripts:
- `test_tensor_integrity.py` - Tensor integrity tests
- `test_timestamps_csv.py` - Timestamp CSV tests

## temp/
Used to store preprocessed data and expert model predictions.

## models/

Used to store trained machine learning models, including the Random Forest and RoBERTa variants.

## output/

Contains the results of bot detection analyses, including JSON files with detailed classification outputs for various accounts.
