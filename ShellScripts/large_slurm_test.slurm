#!/bin/bash

# X number of nodes with Y number of cores in each node.
# p - Partition: cpu | res-gpu-small | res-gpu-large | ug-gpu-small
# QOS: debug | short | long-high-prio | long-low-prio | long-cpu
# t - TIME


#SBATCH -c 8
#SBATCH -p gpu-bigmem
#SBATCH --qos=kgxj22-qos
#SBATCH --mem=110G

# Source the bash profile (required to use the module command)
source /etc/profile

# activate venv
source ../../venv/bin/activate

# Run your program (replace this with your program)
stdbuf -oL python3 ../Code/pre_training/main.py