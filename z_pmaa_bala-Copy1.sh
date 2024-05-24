#!/bin/bash
#SBATCH --mail-user=ck696@cornell.edu    # Email for status updates
#SBATCH --mail-type=END                  # Request status by email
#SBATCH --get-user-env                   # Retrieve the users login environment
#SBATCH -t 80:00:00                      # Time limit (hh:mm:ss)
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=128GB
#SBATCH --time=3000:00:00
#SBATCH --partition=bala
#SBATCH -N 1                             # Number of nodes
#SBATCH--output=watch_folder/%x-%j.log   # Output file name
#SBATCH --requeue                        # Requeue job if it fails

# Setup python path and env
source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh
conda activate H3
cd /share/hariharan/ck696/allclear/baselines/PMAA

# 05/22
# python train_0522.py --dataset_name "CTGAN_Sen2_MTC"
# python train_0522.py --dataset_name "AllClear" --lambda_L1 100
# python train_0522.py --dataset_name "AllClear" --lambda_L1 50
# python train_0522.py --dataset_name "AllClear" --lambda_L1 25

# python train_0523.py --dataset_name "AllClear" --lambda_L1 50 --masked_L1 0
# python train_0523.py --dataset_name "AllClear" --lambda_L1 50 --masked_L1 1
# python train_0523.py --dataset_name "AllClear" --lambda_L1 50 --masked_L1 1 --pred_cloud 1
# python train_0523.py --dataset_name "AllClear" --lambda_L1 50 --masked_L1 1 --pred_cloud 1 --lr 0.0002
python train_0523.py --dataset_name "AllClear" --lambda_L1 100 --masked_L1 1 --pred_cloud 1 --lr 0.0005