#!/bin/bash

# Setup python path and env
source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh
conda activate H3
cd /share/hariharan/ck696/allclear/baselines/PMAA

# print(befin process)
echo "Start process"

# 05/22
# python train_0522.py --dataset_name "CTGAN_Sen2_MTC"
# python train_0522.py --dataset_name "AllClear" --lambda_L1 100
# python train_0522.py --dataset_name "AllClear" --lambda_L1 50
# python train_0522.py --dataset_name "AllClear" --lambda_L1 25

# python train_0523.py --dataset_name "AllClear" --lambda_L1 100 --masked_L1 0
# python train_0523.py --dataset_name "AllClear" --lambda_L1 100 --masked_L1 1


# python train_0523.py --dataset_name "AllClear" --lambda_L1 50 --masked_L1 1 --pred_cloud 1
python train_0523.py --dataset_name "AllClear" --lambda_L1 50 --masked_L1 1 --pred_cloud 1 --lr 0.0002