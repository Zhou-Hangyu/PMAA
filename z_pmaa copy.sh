#!/bin/bash
#SBATCH --mail-user=ck696@cornell.edu    # Email for status updates
#SBATCH --mail-type=END                  # Request status by email
#SBATCH --get-user-env                   # Retrieve the users login environment
#SBATCH -t 80:00:00                      # Time limit (hh:mm:ss)
#SBATCH --partition=gpu                  # Partition
#SBATCH --constraint="gpu-high|gpu-mid"  # GPU constraint
#SBATCH --gres=gpu:1                     # Number of GPUs
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=10                # Number of CPU cores per task
#SBATCH -N 1                             # Number of nodes
#SBATCH--output=watch_folder/%x-%j.log   # Output file name
#SBATCH --requeue                        # Requeue job if it fails

# Setup python path and env
source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh
conda activate H3
cd /share/hariharan/ck696/allclear/baselines/PMAA


# 05/22
# python train_0522.py --dataset_name "CTGAN_Sen2_MTC"
# python train_0522.py --dataset_name "AllClear_v1" --lambda_L1 50
# python train_0522.py --dataset_name "AllClear_v1" --lambda_L1 10



















#ss SBATCH --partition=hariharan                  # Partition
#ss SBATCH --nodelist=hariharan-compute-03   # Node
#ss SBATCH --gres=gpu:1                     # Number of GPUs

# 04/30
# python UNet3D_v47.py --mode "train" --lr 2e-5 --max_d0 128 --max_dim 512 --model_blocks "CCRRAA" --norm_num_groups 4
# python UNet3D_v47.py --mode "train" --lr 2e-5 --max_d0 128 --max_dim 512 --model_blocks "CCRAA" --norm_num_groups 4
# python UNet3D_v47.py --mode "train" --lr 2e-5 --max_d0 128 --max_dim 512 --model_blocks "CRRAA" --norm_num_groups 4
# python UNet3D_v47.py --mode "train" --lr 2e-5 --max_d0 128 --max_dim 512 --model_blocks "CRRAAA" --norm_num_groups 4

# 04/29
# python UNet3D_v46.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCCAA" --norm_num_groups 4
# python UNet3D_v46.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCRRAA" --norm_num_groups 4
# python UNet3D_v46.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCAAA" --norm_num_groups 4
# python UNet3D_v46.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCRAAA" --norm_num_groups 4

# python UNet3D_v46.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCRRAA" --train_bs 1 --time_span 12 --norm_num_groups 4
# python UNet3D_v46.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "RRRRAA" --train_bs 1 --time_span 12 --norm_num_groups 4

# 04 26
# python UNet3D_v45.py --mode "train" --lr 2e-5 --max_dim 768 --model_blocks "CRRAAA" --train_bs 1 --time_span 12 --norm_num_groups 4
# python UNet3D_v45.py --mode "train" --lr 2e-5 --max_dim 768 --model_blocks "CCRAAA" --train_bs 1 --time_span 12 --norm_num_groups 4
# python UNet3D_v45.py --mode "train" --lr 2e-5 --max_dim 768 --model_blocks "CRRAAA" --train_bs 1 --time_span 10 --norm_num_groups 4 OK
# python UNet3D_v45.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CRRAAA" --train_bs 1 --time_span 10 --norm_num_groups 4 OK
# python UNet3D_v45.py --mode "train" --lr 2e-5 --max_dim 768 --model_blocks "CCRAAA" --train_bs 1 --time_span 12 --norm_num_groups 4 x

# python UNet3D_v45.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCRAAA" --train_bs 1 --time_span 12 --norm_num_groups 4 OK
# python UNet3D_v45.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CRRAAA" --train_bs 1 --time_span 12 --norm_num_groups 4 --postfix "NoTimePerm"
# python UNet3D_v45.py --mode "train" --lr 2e-5 --max_dim 768 --model_blocks "CCRRAA" --train_bs 1 --time_span 12 --norm_num_groups 4 --postfix "NoTimePerm"
# python UNet3D_v45.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCRAAA" --train_bs 1 --time_span 12 --norm_num_groups 4 --postfix "NoTimePerm"
# python UNet3D_v45.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CRRAAA" --train_bs 1 --time_span 10 --norm_num_groups 4 --postfix "NoTimePerm"
# python UNet3D_v45.py --mode "train" --lr 2e-5 --max_dim 768 --model_blocks "CRRAAA" --train_bs 1 --time_span 10 --norm_num_groups 4

# python UNet3D_v44.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCRAAA" --train_bs 1 --time_span 12 --model_type "UNet3D_src" --image_size 224
# python UNet3D_v44.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCRAAA" --train_bs 1 --time_span 12 --model_type "UNet3D_src" --image_size 224 --norm_num_groups 4
# python UNet3D_v44.py --mode "train" --lr 2e-5 --max_dim 768 --model_blocks "CRRAAA" --train_bs 1 --time_span 12 --model_type "UNet3D_src" --image_size 224 --norm_num_groups 4
# python UNet3D_v44.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCAAA" --train_bs 1 --time_span 16 --model_type "UNet3D_src" --image_size 224
# OOM # python UNet3D_v44.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "RRRAAA" --train_bs 1 --time_span 12 --model_type "UNet3D_src" --image_size 224 --LPB 1


# 04-23 
# python UNet3D_v42.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCC" --norm_num_groups 2 --num_workers 0 --train_bs 2
# python UNet3D_v42.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCC" --norm_num_groups 2 --num_workers 0 --train_bs 2
# python UNet3D_v42.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCC" --norm_num_groups 2 --train_bs 1

# python UNet3D_v42.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCCA" --norm_num_groups 2 --train_bs 1 --time_span 9
# # python UNet3D_v42.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCCA" --norm_num_groups 2 --train_bs 1 --time_span 7
# python UNet3D_v42.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCCA" --norm_num_groups 2 --train_bs 1 --time_span 6
# python UNet3D_v42.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCCA" --norm_num_groups 2 --train_bs 1 --time_span 5


# python UNet3D_v42.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCCAA"  --norm_num_groups 2 --train_bs 1 --time_span 10 
# RUNNING: python UNet3D_v42.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCCAA"  --norm_num_groups 4 --train_bs 1 --time_span 10 --loss_threshold 0.5
# python UNet3D_v42.py --mode "train" --lr 4e-5 --max_dim 512 --model_blocks "CCCCAA"  --norm_num_groups 4 --train_bs 1 --time_span 10 --loss_threshold 0.5 --model_type "UNet3D"
# python UNet3D_v42.py --mode "train" --lr 4e-5 --max_dim 512 --model_blocks "CCCCAA"  --norm_num_groups 4 --train_bs 1 --time_span 10 --loss_threshold 0.5 --model_type "UNet3D_src"
# RUNNING: python UNet3D_v42.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCAAA"  --norm_num_groups 4 --train_bs 1 --time_span 10 --loss_threshold 0.5
# RUNNING: python UNet3D_v42.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCAAA"  --norm_num_groups 4 --train_bs 1 --time_span 10 --loss_threshold 0.5
# RUNNING: python UNet3D_v42.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCAAA"  --norm_num_groups 4 --train_bs 1 --time_span 10 --loss_threshold 0.5
# python UNet3D_v42.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCCAA"  --norm_num_groups 32 --train_bs 1 --time_span 10
# python UNet3D_v42.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCCAA"  --norm_num_groups 32 --train_bs 1 --time_span 8
# python UNet3D_v42.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCCAA"  --norm_num_groups 2 --train_bs 1 --time_span 8


# python UNet3D_v42.py --mode "train" --lr 4e-5 --max_dim 512 --model_blocks "CCCCAA"  --norm_num_groups 4 --train_bs 1 --time_span 10 --loss_threshold 0.5 --model_type "UNet3D_src"


# python UNet3D_v43.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "JCCCAA" --train_bs 1 --time_span 10 --model_type "UNet3D_src" --image_size 224
# python UNet3D_v43.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "JCCRAA" --train_bs 1 --time_span 10 --model_type "UNet3D_src" --image_size 224
# python UNet3D_v43.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "JCCAAA" --train_bs 1 --time_span 10 --model_type "UNet3D_src" --image_size 224 --postfix "TToken_TPerm_AddCloud"
# python UNet3D_v43.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCCAA" --train_bs 1 --time_span 10 --model_type "UNet3D_src" --image_size 224
# python UNet3D_v43.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCAAA" --train_bs 1 --time_span 10 --model_type "UNet3D_src" --image_size 224

# python UNet3D_v43.py --mode "train" --lr 2e-5 --max_dim 1024 --model_blocks "CCCAAA" --train_bs 1 --time_span 10 --model_type "UNet3D_src" --image_size 224 --postfix "TToken_TPerm_AddCloud"
# python UNet3D_v43.py --mode "train" --lr 2e-5 --max_dim 1024 --model_blocks "CCCAA" --train_bs 1 --time_span 10 --model_type "UNet3D_src" --image_size 224 --postfix "TToken_TPerm_AddCloud"
# python UNet3D_v43.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCAA" --train_bs 1 --time_span 10 --model_type "UNet3D_src" --image_size 224 --postfix "TToken_TPerm_AddCloud" --LPB 2


# python UNet3D_v43.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCAA" --train_bs 1 --time_span 10 --model_type "UNet3D" --image_size 224
# python UNet3D_v43.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCAAA" --train_bs 1 --time_span 10 --model_type "UNet3D" --image_size 224


# python UNet3D_v42.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCCA" --norm_num_groups 2 --train_bs 1 --time_span 14
# python UNet3D_v42.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCCA" --norm_num_groups 2 --train_bs 1 --time_span 12
# python UNet3D_v42.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCCA" --norm_num_groups 2 --train_bs 1 --time_span 10 --loss_threshold 0.5
# python UNet3D_v42.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCCA" --norm_num_groups 2 --train_bs 1 --time_span 8 --loss_threshold 0.5
# python UNet3D_v42.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCCA" --norm_num_groups 2 --train_bs 1 --time_span 6 --loss_threshold 0.5

# python UNet3D_v42.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCCC" --norm_num_groups 2 --train_bs 1 --time_span 14
# python UNet3D_v42.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCCC" --norm_num_groups 2 --train_bs 1 --time_span 12
# python UNet3D_v42.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCCC" --norm_num_groups 2 --train_bs 1 --time_span 10 --loss_threshold 0.5
# python UNet3D_v42.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCCC" --norm_num_groups 2 --train_bs 1 --time_span 8
# python UNet3D_v42.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCCC" --norm_num_groups 2 --train_bs 1 --time_span 6

# python UNet3D_v42.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCCC" --norm_num_groups 4 --train_bs 1 --time_span 14
# python UNet3D_v42.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCCC" --norm_num_groups 4 --train_bs 1 --time_span 12
# python UNet3D_v42.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCCC" --norm_num_groups 4 --train_bs 1 --time_span 10
# python UNet3D_v42.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCCC" --norm_num_groups 4 --train_bs 1 --time_span 8
# python UNet3D_v42.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCCC" --norm_num_groups 4 --train_bs 1 --time_span 6


# python UNet3D_v42_ori.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCCA" --norm_num_groups 2 --train_bs 1 --time_span 5

# 04 18
# python UNet3D_v37.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCCAA" --norm_num_groups 2 --num_workers 0
# python UNet3D_v37.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCCAA" --norm_num_groups 2 --num_workers 2
# python UNet3D_v37.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCCAA" --norm_num_groups 2 --num_workers 4


# 0417
# python UNet3D_v35.py --mode "train" --lr 2e-5 --max_dim 256 --model_blocks "CCCCCA" --wandb 0 --num_workers 4
# python UNet3D_v35.py --mode "train" --lr 2e-5 --max_dim 256 --model_blocks "CCCCAA" --wandb 0 --num_workers 4
# python UNet3D_v35.py --mode "train" --lr 3e-5 --max_dim 256 --model_blocks "CCCCAA" --wandb 0 --num_workers 4
# python UNet3D_v35.py --mode "train" --lr 3e-5 --max_dim 256 --model_blocks "CCCCC" --wandb 0 --num_workers 4

# python UNet3D_v35.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCCCA" --wandb 0 --num_workers 4 --postfix "init128"
# python UNet3D_v35.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCCCA" --wandb 0 --num_workers 4 --postfix "init128" --dilate_kernel 21 --norm_num_groups 4
# python UNet3D_v35.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCCAA" --wandb 0 --num_workers 4 --postfix "init128" --dilate_kernel 21 --norm_num_groups 4

# python UNet3D_v36.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCCAA" --dilate_kernel 21 --num_workers 4
# python UNet3D_v36.py --mode "train" --lr 2e-5 --max_dim 512 --model_blocks "CCCCAA" --dilate_kernel 21 --norm_num_groups 2 --num_workers 2

# python UNet3D_v36.py --mode "train" --lr 1e-5 --max_dim 256 --model_blocks "CCCCAA" --wandb 0 --dilate_kernel 13
# python UNet3D_v36.py --mode "train" --lr 2e-5 --max_dim 256 --model_blocks "CCCCAA" --wandb 0 --dilate_kernel 21
# python UNet3D_v36.py --mode "train" --lr 2e-5 --max_dim 256 --model_blocks "CCCCAA" --wandb 0 --dilate_kernel 31
# python UNet3D_v36.py --mode "train" --lr 2e-5 --max_dim 256 --model_blocks "CCCAAA" --wandb 0 --dilate_kernel 31

# 04/16
# python UNet3D_v35.py --mode "train" --lr 5e-5 --postfix "concur" --max_dim 256 --model_blocks "CCCCC"

# python UNet3D_v33.py --mode "train" --time_span 5 --lr 5e-5 --postfix "concur"
# python UNet3D_v33.py --mode "train" --time_span 8 --lr 5e-5 --postfix "concur"
# python UNet3D_v33.py --mode "train" --time_span 14 --lr 5e-5 --postfix "concur" --max_dim 256
# python UNet3D_v33.py --mode "train" --time_span 14 --lr 5e-5 --postfix "concur" --max_dim 256 --model_blocks "CCCA"
# python UNet3D_v33.py --mode "train" --time_span 14 --lr 5e-5 --postfix "concur" --max_dim 256 --model_blocks "CCCC"
# python UNet3D_v33.py --mode "train" --time_span 14 --lr 5e-5 --postfix "concur" --max_dim 256 --model_blocks "CCCCC"

# 04/15
# python UNet3D_v31.py --mode "train" --time_span 5 --lr 5e-5 --model_blocks "CCCAA" --visShadow 1 --LPB 1
# python UNet3D_v31.py --mode "train" --time_span 5 --lr 5e-5 --model_blocks "CCCCAA" --visShadow 1 --LPB 1 --postfix "vmax_lossthres1"
# python UNet3D_v31.py --mode "train" --time_span 8 --lr 5e-5 --model_blocks "CCCCAAA" --visShadow 1 --LPB 1

# python UNet3D_v32.py --mode "train" --time_span 5 --lr 5e-5 --model_blocks "CCCCAA" --visShadow 1 --LPB 1 --postfix "FixVisVmax"
# python UNet3D_v32.py --mode "train" --time_span 5 --lr 5e-5 --model_blocks "CCCCAA" --visShadow 1 --LPB 1 --postfix "FixVisVmax_AddEmbed" --dynamic_vmax 1
# python UNet3D_v32.py --mode "train" --time_span 5 --lr 5e-5 --model_blocks "CCCCAA" --visShadow 1 --LPB 1 --postfix "FixVisVmax_AddEmbed" --dynamic_vmax 2 --norm_num_groups 2
# python UNet3D_v32.py --mode "train" --time_span 6 --lr 5e-5 --model_blocks "CCCCAA" --visShadow 1 --LPB 1 --postfix "FixVisVmax_AddEmbed" --dynamic_vmax 1
# python UNet3D_v32.py --mode "train" --time_span 5 --lr 5e-5 --model_blocks "CCCCAA" --visShadow 1 --LPB 1 --postfix "FixVisVmax_AddEmbed" --dynamic_vmax 1 --MaskedInput 1
# python UNet3D_v32.py --mode "train" --time_span 5 --lr 5e-5 --model_blocks "CCCCAA" --visShadow 1 --LPB 1 --postfix "FixVisVmax_AddEmbed" --dynamic_vmax 1 --MaskedInput 2
# python UNet3D_v32.py --mode "train" --time_span 5 --lr 5e-5 --model_blocks "CCCCAA" --visShadow 1 --LPB 1 --postfix "FixVisVmax_AddEmbed" --dynamic_vmax 1 --norm_num_groups 2
# python UNet3D_v32.py --mode "train" --time_span 5 --lr 5e-5 --model_blocks "CCCCAA" --visShadow 1 --LPB 1 --postfix "FixVisVmax_AddEmbed" --dynamic_vmax 1 --norm_num_groups 4

# 04/13
# python UNet3D_v30.py --mode "train" --time_span 5 --lr 2e-5 --model_blocks "CCCAAA" --visShadow 1
# python UNet3D_v30.py --mode "train" --time_span 5 --lr 2e-5 --model_blocks "CCCCCCC"
# python UNet3D_v30.py --mode "train" --time_span 5 --lr 2e-5 --model_blocks "CCCAAA" --LPB 1
# python UNet3D_v30.py --mode "train" --time_span 5 --lr 2e-5 --model_blocks "CCCCCCC" --LPB 1
# python UNet3D_v30.py --mode "train" --time_span 8 --lr 2e-5 --model_blocks "CCCCCCC" --LPB 1
# python UNet3D_v30.py --mode "train" --time_span 8 --lr 2e-5 --model_blocks "CCCCACA" --LPB 1 --visShadow 1
# python UNet3D_v30.py --mode "train" --time_span 6 --lr 2e-5 --model_blocks "CCCCA" --LPB 1 --visShadow 1


# python UNet3D_v24.py --mode "train" --time_span 5 --lr 2e-5 --model_blocks "CCCAA" --refine_method "output" --refine_model_blocks "CCCA"
# python UNet3D_v24.py --mode "train" --time_span 5 --lr 2e-5 --model_blocks "CCCAA" --refine_method "residual" --refine_model_blocks "CCCA"
# python UNet3D_v24.py --mode "train" --time_span 5 --lr 2e-5 --model_blocks "CCCAA" --refine_method "single" --refine_model_blocks "CCCA"
# python UNet3D_v24.py --mode "train" --time_span 5 --lr 2e-5 --model_blocks "CCCAA" --refine_method "none"

# python UNet3D_v25_dis.py --mode "train" --time_span 5 --lr 5e-5 --model_blocks "CCCAAA" --dynamic_vmax 1 --LPB 1 --use_discriminator 1
# python UNet3D_v25_dis.py --mode "train" --time_span 5 --lr 2e-5 --model_blocks "CCCAAA" --dynamic_vmax 1 --LPB 1 --use_discriminator 1
# python UNet3D_v25_dis.py --mode "train" --time_span 10 --lr 2e-5 --model_blocks "CCCAAA" --dynamic_vmax 1 --LPB 1 --use_discriminator 1


# 04/12
# python UNet3D_v21.py --mode "train" --time_span 3 --lr 1e-4 --model_blocks "CCCAA" --dynamic_vmax 1
# python UNet3D_v21.py --mode "train" --time_span 5 --lr 1e-4 --model_blocks "CCCAA" --dynamic_vmax 1

# python UNet3D_v21.py --mode "train" --time_span 3 --lr 5e-5 --model_blocks "CCCAAA" --dynamic_vmax 1
# python UNet3D_v21.py --mode "train" --time_span 5 --lr 5e-5 --model_blocks "CCCAAA" --dynamic_vmax 1
# python UNet3D_v21.py --mode "train" --time_span 3 --lr 2e-5 --model_blocks "CCCAAA" --dynamic_vmax 1
# python UNet3D_v21.py --mode "train" --time_span 5 --lr 2e-5 --model_blocks "CCCAAA" --dynamic_vmax 1

# python UNet3D_v22.py --mode "train" --time_span 5  --lr 2e-5 --model_blocks "CCCAAA" --dynamic_vmax 1 --LPB 1
# python UNet3D_v22.py --mode "train" --time_span 5  --lr 2e-5 --model_blocks "CCCAAAA" --dynamic_vmax 1 --LPB 1
# python UNet3D_v22.py --mode "train" --time_span 10 --lr 2e-5 --model_blocks "CCCAAA" --dynamic_vmax 1 --LPB 1
# python UNet3D_v22.py --mode "train" --time_span 10 --lr 2e-5 --model_blocks "CCCAAAA" --dynamic_vmax 1 --LPB 1

# python UNet3D_v22.py --mode "train" --time_span 5  --lr 2e-5 --model_blocks "CCCAAA" --dynamic_vmax 1 --LPB 1
# python UNet3D_v22.py --mode "train" --time_span 5  --lr 2e-5 --model_blocks "CCCAAAA" --dynamic_vmax 1 --LPB 1
# python UNet3D_v22.py --mode "train" --time_span 10 --lr 2e-5 --model_blocks "CCCAAA" --dynamic_vmax 1 --LPB 1
# python UNet3D_v22.py --mode "train" --time_span 10 --lr 2e-5 --model_blocks "CCCAAAA" --dynamic_vmax 1 --LPB 1



# 04/11
# python UNet3D_v20.py --mode "train" --time_span 3 --lr 1e-4 --model_blocks "CCCAC" --dynamic_vmax 1
# python UNet3D_v20.py --mode "train" --time_span 5 --lr 1e-4 --model_blocks "CCCAC" --dynamic_vmax 1
# python UNet3D_v20.py --mode "train" --time_span 5 --lr 1e-4 --model_blocks "CCCCC" --dynamic_vmax 1
# python UNet3D_v20.py --mode "train" --time_span 5 --lr 1e-4 --model_blocks "CCCCC" --dynamic_vmax 1
# python UNet3D_v20.py --mode "train" --time_span 5 --lr 1e-4 --model_blocks "CCCCCC" --dynamic_vmax 1
# python UNet3D_v20.py --mode "train" --time_span 5 --lr 1e-4 --model_blocks "CCCCAA" --dynamic_vmax 1
# python UNet3D_v20.py --mode "train" --time_span 5 --lr 1e-4 --model_blocks "CCCA" --dynamic_vmax 1
# python UNet3D_v20.py --mode "train" --time_span 5 --lr 1e-4 --model_blocks "CCCC" --dynamic_vmax 1
# python UNet3D_v20.py --mode "train" --time_span 5 --lr 1e-4 --model_blocks "CCCCA" --dynamic_vmax 1

# python UNet3D_v20.py --mode "train" --time_span 7 --lr 1e-4 --model_blocks "CCCCC" --dynamic_vmax 1
# python UNet3D_v20.py --mode "train" --time_span 7 --lr 1e-4 --model_blocks "CCCCA" --dynamic_vmax 1
# python UNet3D_v20.py --mode "train" --time_span 7 --lr 1e-4 --model_blocks "CCCAA" --dynamic_vmax 1
# python UNet3D_v20.py --mode "train" --time_span 7 --lr 1e-4 --model_blocks "CCCAA" --dynamic_vmax 1
# python UNet3D_v20.py --mode "train" --time_span 7 --lr 1e-4 --model_blocks "CCCAAA" --dynamic_vmax 1

# python UNet3D_v20.py --mode "train" --time_span 11 --lr 1e-4 --model_blocks "CCCAAA" --dynamic_vmax 1
# python UNet3D_v20.py --mode "train" --time_span 11 --lr 1e-4 --model_blocks "CCCAA" --dynamic_vmax 1
# python UNet3D_v20.py --mode "train" --time_span 11 --lr 1e-4 --model_blocks "CCCCA" --dynamic_vmax 1
# python UNet3D_v20.py --mode "train" --time_span 11 --lr 1e-4 --model_blocks "CCCCC" --dynamic_vmax 1

# python UNet_v1u5_cond3d.py --mode "train" --time_span 3 --lr 1e-4 --train_bs 1 --model_blocks "CCCAA" --grad_accm 1
# python UNet_v1u5_cond3d.py --mode "train" --time_span 3 --lr 1e-4 --train_bs 1 --model_blocks "CCCAA" --grad_accm 1 #u52
# python UNet_v1u6_cond3d.py --mode "train" --time_span 3 --lr 1e-4 --train_bs 1 --model_blocks "CCCAA" --grad_accm 1
# python UNet_v1u6_cond3d.py --mode "train" --time_span 3 --lr 1e-4 --train_bs 1 --model_blocks "CCAAA" --grad_accm 1

# python UNet_v1u7_cond3d.py --mode "train" --time_span 3 --lr 1e-4 --train_bs 1 --model_blocks "CCCAA" --grad_accm 1 --noise_var 0.1
# python UNet_v1u7_cond3d.py --mode "train" --time_span 3 --lr 1e-4 --train_bs 1 --model_blocks "CCCAA" --grad_accm 1 --noise_var 0.2
# python UNet_v1u7_cond3d.py --mode "train" --time_span 3 --lr 1e-4 --train_bs 1 --model_blocks "CCCAA" --grad_accm 1 --noise_var 0.3

# python UNet_v1u7_cond3d.py --mode "train" --time_span 3 --lr 1e-4 --train_bs 1 --model_blocks "CCCAA" --grad_accm 1 --noise_var 0.2 --CloudMaskDilation 5
# python UNet_v1u7_cond3d.py --mode "train" --time_span 3 --lr 1e-4 --train_bs 1 --model_blocks "CCCAA" --grad_accm 1 --noise_var 0.2 --CloudMaskDilation 7

# python UNet_v1u8_cond3d.py --mode "train" --time_span 3 --lr 1e-4 --train_bs 1 --model_blocks "CCCAA" --grad_accm 1 --noise_var 0.2 --CloudMaskDilation 7 --CloudShadownScale 0.2 --dynamic_vmax 1
# python UNet_v1u8_cond3d.py --mode "train" --time_span 3 --lr 1e-4 --train_bs 1 --model_blocks "CCCAA" --grad_accm 1 --noise_var 0.2 --CloudMaskDilation 11 --CloudShadownScale 0.2 --dynamic_vmax 1
# python UNet_v1u8_cond3d.py --mode "train" --time_span 3 --lr 1e-4 --train_bs 1 --model_blocks "CCCAA" --grad_accm 1 --noise_var 0.2 --CloudMaskDilation 7 --CloudShadownScale 0.5 --dynamic_vmax 1
# python UNet_v1u8_cond3d.py --mode "train" --time_span 3 --lr 1e-4 --train_bs 1 --model_blocks "CCCAA" --grad_accm 1 --noise_var 0.2 --CloudMaskDilation 11 --CloudShadownScale 0.5 --dynamic_vmax 1

# python UNet_v1u9_cond3d.py --mode "train" --time_span 3 --lr 1e-4 --train_bs 1 --model_blocks "CCCAA" --grad_accm 1 --noise_var 0.2 --CloudMaskDilation 7 --CloudShadownScale 0.2 --dynamic_vmax 1 --loss_type "MAE"
# python UNet_v1u9_cond3d.py --mode "train" --time_span 3 --lr 1e-4 --train_bs 1 --model_blocks "CCCAA" --grad_accm 1 --noise_var 0.2 --CloudMaskDilation 7 --CloudShadownScale 0.5 --dynamic_vmax 1 --loss_type "MAE"

# 0410

# python UNet_v1u3_cond3d.py --mode "train" --time_span 3 --lr 1e-4 --train_bs 1 --model_blocks "CCCAC" --grad_accm 1
# python UNet_v1u4_cond3d.py --mode "train" --time_span 3 --lr 1e-4 --train_bs 1 --model_blocks "CCCAC" --grad_accm 1
# python UNet_v1u3_cond3d.py --mode "train" --time_span 3 --lr 1e-4 --train_bs 1 --model_blocks "CCCAC" --grad_accm 1
# python UNet_v1u3_cond3d.py --mode "train" --time_span 3 --lr 1e-4 --train_bs 3 --model_blocks "CCCAC" --grad_accm 1
# python UNet_v1u3_cond3d.py --mode "test" --time_span 3 --lr 1e-4 --train_bs 1 --model_blocks "CCCAC" --grad_accm 1
# python UNet_v1u3_cond3d.py --mode "test" --time_span 3 --lr 1e-4 --train_bs 2 --model_blocks "CCCAC" --grad_accm 1

# python UNet_v1_cond3d.py --mode "train" --time_span 3 --cld_sc 0.3 --vmax 0.5 --lr 2e-4 --train_bs 2 --samplesample_reuse --model_blocks "CCCCCC" # 3136398
# python UNet_v1_cond3d.py --mode "train" --time_span 3 --cld_sc 0.3 --vmax 0.5 --lr 2e-4 --train_bs 2 --model_blocks "CCCCC"  # 3136399
# python UNet_v1_cond3d.py --mode "train" --time_span 3 --cld_sc 0.3 --vmax 0.5 --lr 2e-4 --train_bs 2 --model_blocks "CCCCCA" # 3136410
# python UNet_v1_cond3d.py --mode "train" --time_span 3 --cld_sc 0.3 --vmax 0.5 --lr 2e-4 --train_bs 2 --model_blocks "CCCCA"  # 3136416
# python UNet_v1_cond3d.py --mode "train" --time_span 3 --cld_sc 0.3 --vmax 0.5 --lr 2e-4 --train_bs 2 --model_blocks "CCCAC"  # grad acc = 1
# python UNet_v1_cond3d.py --mode "train" --time_span 3 --cld_sc 0.3 --vmax 0.5 --lr 2e-4 --train_bs 2 --model_blocks "CCACC"  # grad acc = 1
# python UNet_v1_cond3d.py --mode "train" --time_span 3 --cld_sc 0.3 --vmax 0.5 --lr 2e-4 --train_bs 2 --model_blocks "CACCC"  # grad acc = 1
# python UNet_v1_cond3d.py --mode "train" --time_span 3 --cld_sc 0.3 --vmax 0.5 --lr 2e-4 --train_bs 2 --model_blocks "CACCC"  # grad acc = 1

# python UNet_v1_cond3d.py --mode "train" --time_span 3 --cld_sc 0.3 --vmax 0.5 --lr 1e-4 --train_bs 2 --sample_reuse 1 --model_blocks "CCCAC" --grad_accm 1
# python UNet_v1_cond3d.py --mode "train" --time_span 3 --cld_sc 0.3 --vmax 0.5 --lr 1e-4 --train_bs 2 --sample_reuse 1 --model_blocks "CCCAC" --grad_accm 4
# python UNet_v1_cond3d.py --mode "train" --time_span 3 --cld_sc 0.3 --vmax 0.5 --lr 5e-5 --train_bs 2 --sample_reuse 1 --model_blocks "CCCAC" --grad_accm 1
# python UNet_v1_cond3d.py --mode "train" --time_span 3 --cld_sc 0.3 --vmax 0.5 --lr 5e-5 --train_bs 2 --sample_reuse 1 --model_blocks "CCCAC" --grad_accm 4
# python UNet_v1_cond3d.py --mode "train" --time_span 3 --cld_sc 0.3 --vmax 0.5 --lr 2e-5 --train_bs 2 --sample_reuse 1 --model_blocks "CCCAC" --grad_accm 1
# python UNet_v1_cond3d.py --mode "train" --time_span 3 --cld_sc 0.3 --vmax 0.5 --lr 2e-5 --train_bs 2 --sample_reuse 1 --model_blocks "CCCAC" --grad_accm 4

# python UNet_v1_cond3d.py --mode "train" --time_span 3 --cld_sc 1.0 --vmax 0.5 --lr 1e-4 --train_bs 2 --sample_reuse 1 --model_blocks "CCCAC" --grad_accm 1
# python UNet_v1_cond3d.py --mode "train" --time_span 3 --cld_sc 1.0 --vmax 0.5 --lr 5e-5 --train_bs 2 --sample_reuse 1 --model_blocks "CCCAC" --grad_accm 1
# python UNet_v1_cond3d.py --mode "train" --time_span 3 --cld_sc 1.0 --vmax 0.5 --lr 2e-5 --train_bs 2 --sample_reuse 1 --model_blocks "CCCAC" --grad_accm 1


# 0409
# python UNet_v0.py --mode "train" --time_span 1 --cld_ns 0.3 --vmax 0.5
# python UNet_v1.py --mode "train" --time_span 3 --cld_sc 0.5 --vmax 0.5
# python UNet_v1.py --mode "train" --time_span 3 --cld_sc 0.5 --vmax 0.5 --model_blocks "CCCAAA"
# python UNet_v1.py --mode "train" --time_span 3 --cld_sc 0.5 --vmax 0.5 --model_blocks "CCCAAA" --learning_rate 1e-3
# python UNet_v1.py --mode "train" --time_span 3 --cld_sc 0.5 --vmax 0.5 --model_blocks "CCCCAA"
# python UNet_v1.py --mode "train" --time_span 3 --cld_sc 0.5 --vmax 0.5 --model_blocks "CCCCAA" --learning_rate 1e-3

# python UNet_v1.py --mode "train" --time_span 3 --cld_sc 0.3 --vmax 0.5 --learning_rate 2e-4
# python UNet_v1.py --mode "train" --time_span 3 --cld_sc 0.5 --vmax 0.5 --learning_rate 2e-4
# python UNet_v1.py --mode "train" --time_span 3 --cld_sc 1.0 --vmax 0.5 --learning_rate 2e-4

# python UNet_v1.py --mode "train" --time_span 3 --cld_sc 0.3 --vmax 0.5 --learning_rate 2e-4 --model_blocks "CCCAAA"
# python UNet_v1.py --mode "train" --time_span 3 --cld_sc 0.3 --vmax 0.5 --learning_rate 2e-4 --model_blocks "CCCCAA"
# python UNet_v1.py --mode "train" --time_span 3 --cld_sc 0.3 --vmax 0.5 --learning_rate 2e-4 --model_blocks "CCCAAC"

# python UNet_v1_cond2d.py --mode "train" --time_span 3 --cld_sc 0.3 --vmax 0.5 --learning_rate 2e-4
# python UNet_v1_cond3d.py --mode "train" --time_span 3 --cld_sc 0.3 --vmax 0.5 --learning_rate 2e-4
# python UNet_v1_cond2d.py --mode "train" --time_span 3 --cld_sc 0.3 --vmax 0.5 --learning_rate 4e-4
# python UNet_v1_cond3d.py --mode "train" --time_span 3 --cld_sc 0.3 --vmax 0.5 --learning_rate 4e-4

# python UNet_v1_cond3d.py --mode "train" --time_span 3 --cld_sc 0.3 --vmax 0.5 --learning_rate 2e-4
# python UNet_v1_cond3d.py --mode "train" --time_span 3 --cld_sc 0.3 --vmax 0.5 --learning_rate 2e-4 --model_blocks "CCCAAA"
# python UNet_v1_cond3d.py --mode "train" --time_span 3 --cld_sc 0.3 --vmax 0.5 --learning_rate 2e-4 --model_blocks "CCCCAA"
# python UNet_v1_cond3d.py --mode "train" --time_span 3 --cld_sc 0.3 --vmax 0.5 --learning_rate 2e-4 --model_blocks "CCCCCCC"

