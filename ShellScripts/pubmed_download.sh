#!/bin/bash

cd ../Code/datasets/PubMed/raw_data

# get the data from the pubmed baseline - reject md5 files to get only the zipped data
wget -r --no-parent --no-directories ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/ > ../out.log 2> ../err.log --reject "*.md5"