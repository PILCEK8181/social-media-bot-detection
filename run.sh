#!/bin/bash
#PBS -N FINALTEST
#PBS -l select=1:ncpus=1:mem=32gb:ngpus=1:scratch_local=10gb
#PBS -l walltime=04:00:00
#PBS -j oe

# Cleanup
module purge
cd $PBS_O_WORKDIR


# load python module
module add python/3.11.11-gcc-10.2.1

# venv activation
source venv/bin/activate

# run the scripts


## EXAMPLE
#python src/preprocess_roberta_bio_embeddings.py

