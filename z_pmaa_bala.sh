#!/bin/bash

# Setup python path and env
source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh
conda activate H3
cd /share/hariharan/ck696/allclear/baselines/PMAA

# 05/22
# python train_0522.py --dataset_name "CTGAN_Sen2_MTC"
python train_0522.py --dataset_name "AllClear" --lambda_L1 50
# python train_0522.py --dataset_name "AllClear_v1" --lambda_L1 10