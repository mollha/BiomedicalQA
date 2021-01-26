#!/bin/bash

# X number of nodes with Y number of cores in each node.
# p - Partition: cpu | res-gpu-small | res-gpu-large | ug-gpu-small
# QOS: debug | short | long-high-prio | long-low-prio | long-cpu
# t - TIME

#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=debug
#SBATCH --job-name=sm_language
#SBATCH -p gpu-bigmem
#SBATCH --qos=long-high-prio
#SBATCH --mem=28g
#SBATCH --mail-type=FAIL
#SBATCH --mail-user molly.hayward@durham.ac.uk

# Source the bash profile (required to use the module command)
source /etc/profile

# activate venv
source ../../venv/bin/activate

# parse arguments
while [[ $# -gt 0 ]]
do
key="$1"
SIZE="small"
case $key in
    -c|--checkpoint)
    CHECKPOINT="$2"
    shift # past argument
    shift # past value
    ;;
esac
done

echo "CHECKPOINT = ${CHECKPOINT}"
echo "MODEL SIZE = ${SIZE}"

# Run your program (replace this with your program)
stdbuf -oL python3 ../Code/src/pre_training.py --checkpoint ${CHECKPOINT} --size ${SIZE}