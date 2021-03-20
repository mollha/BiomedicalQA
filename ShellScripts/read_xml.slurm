#!/bin/bash

# X number of nodes with Y number of cores in each node.
# p - Partition: cpu | res-gpu-small | res-gpu-large | ug-gpu-small
# QOS: debug | short | long-high-prio | long-low-prio | long-cpu
# t - TIME

#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu
#SBATCH -p ug-gpu-small
#SBATCH --qos=long-high-prio
#SBATCH -t 12:0:0
#SBATCH --mem=28g
#SBATCH --qos=long-high-prio

# Source the bash profile (required to use the module command)
source /etc/profile

# activate venv
source ../../venv/bin/activate

# Run your program (replace this with your program)
python3 ../Code/ReadXML.py