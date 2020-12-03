#!/bin/bash

# X number of nodes with Y number of
# cores in each node.
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=28g
#SBATCH --job-name=language_model_mh

# Partition: cpu | res-gpu-small | res-gpu-large | ug-gpu-small
#SBATCH -p ug-gpu-small

# QOS: debug | short | long-high-prio | long-low-prio | long-cpu
#SBATCH --qos=debug

# TIME
#SBATCH -t 48:00:00

# Source the bash profile (required to use the module command)
source /etc/profile

# activate venv
source ../../venv/bin/activate

# Run your program (replace this with your program)
python3 ../Code/pre_training/main.py