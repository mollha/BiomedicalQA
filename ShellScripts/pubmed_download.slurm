#!/bin/bash

#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p cpu

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

# get the data from the pubmed baseline - reject md5 files to get only the zipped data
wget -P "../Code/datasets/PubMed/raw_data" -r --no-parent --no-directories ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/ --reject "*.md5"