#!/bin/bash

# X number of nodes with Y number of cores in each node.
# p - Partition: cpu | res-gpu-small | res-gpu-large | ug-gpu-small
# QOS: debug | short | long-high-prio | long-low-prio | long-cpu
# t - TIME

#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu
#SBATCH --job-name=language_model_mh
#SBATCH -p gpu-bigmem
#SBATCH --qos=long-high-prio
#SBATCH -t 7-0:0:0
#SBATCH --mem=56g
#SBATCH --mail-type=FAIL
#SBATCH --mail-user molly.hayward@durham.ac.uk

# Source the bash profile (required to use the module command)
source /etc/profile

# activate venv
source ../../venv/bin/activate

# Run your program (replace this with your program)
python3 ../Code/pre_training/main.py