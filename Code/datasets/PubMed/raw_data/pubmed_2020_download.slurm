#!/bin/bash

#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p cpu


# get the data from the pubmed baseline - reject md5 files to get only the zipped data
wget -r --no-parent --no-directories ftp://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/ > ../out.log 2> ../err.log --reject "*.md5"