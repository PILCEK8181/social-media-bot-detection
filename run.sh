#!/bin/bash
#PBS -N feature
#PBS -l select=1:ncpus=1:mem=8gb:ngpus=1:scratch_local=10gb
#PBS -l walltime=02:00:00
#PBS -j oe

# Cleanup
module purge
cd $PBS_O_WORKDIR


# load python module
module add python/3.11.11-gcc-10.2.1

# venv activation
source venv/bin/activate

# run the scripts


# PREPROCESSING
echo " -------------------------------------------------------------------------"

    #python src/preprocess_lstm_timestamps.py
    #python src/preprocess_roberta_embeddings.py
    #python src/preprocess_roberta_bio_embeddings.py


# TESTS
echo " -------------------------------------------------------------------------"

    #python test/test_timestamps_csv.py
    #python test/test_tensor_integrity.py
    


# TRAINING
echo " -------------------------------------------------------------------------"


    # python src/01_rf.py
    # python src/02_lstm.py
    # python src/03_roberta.py
    # python src/03_roberta_weighted.py
    # python src/03_roberta_oversample.py
    python src/00_meta_classifier.py
    python src/00_rf_classifier.py


# RUN
echo " -------------------------------------------------------------------------"

    # python src/detect.py Charles_leclerc
    # python src/detect.py dezo
    # python src/detect.py elonmusk
    # python src/detect.py NASA
    # python src/detect.py prezidentpavel

    # python src/detect_rf.py Charles_leclerc
    # python src/detect_rf.py dezo
    # python src/detect_rf.py elonmusk
    # python src/detect_rf.py NASA
    # python src/detect_rf.py prezidentpavel


# UTILS
echo " -------------------------------------------------------------------------"

    #python src/get_user.py
    #python src/get_tweets.py

echo " -------------------------------------------------------------------------"
