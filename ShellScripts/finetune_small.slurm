#!/bin/bash

# X number of nodes with Y number of cores in each node.
# p - Partition: cpu | res-gpu-small | res-gpu-large | ug-gpu-small
# QOS: debug | short | long-high-prio | long-low-prio | long-cpu
# t - TIME

#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu
#SBATCH --job-name=sm_finetune
#SBATCH -p ug-gpu-small
#SBATCH --qos=short
#SBATCH -t 2:0:0
#SBATCH --mem=12g
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
    -pc|--p-checkpoint)
    PCHECKPOINT="$2"
    shift # past argument
    shift # past value
    ;;
    -fc|--f-checkpoint)
    FCHECKPOINT="$2"
    shift # past argument
    shift # past value
    ;;
    -qt|--question-type)
    QUESTIONTYPE="$2"
    shift # past argument
    shift # past value
    ;;
    -d|--dataset)
    DATASET="$2"
    shift # past argument
    shift # past value
    ;;
esac
done

echo "PRETRAIN CHECKPOINT = ${PCHECKPOINT}"
echo "FINETUNE CHECKPOINT = ${FCHECKPOINT}"
echo "QUESTION TYPE = ${QUESTIONTYPE}"
echo "DATASET = ${DATASET}"
echo "MODEL SIZE = ${SIZE}"

# Run your program (replace this with your program)
stdbuf -oL python3 ../Code/src/fine_tuning.py --p-checkpoint ${PCHECKPOINT} --f-checkpoint ${FCHECKPOINT} --question-type ${QUESTIONTYPE} --dataset ${DATASET} --size ${SIZE}