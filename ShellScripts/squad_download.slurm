#!/bin/bash

#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p cpu

mkdir ../Code/datasets/SQuAD
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json -O ../Code/datasets/squad/train-v2.0.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O ../Code/datasets/squad/dev-v2.0.json