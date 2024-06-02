export CUDA_VISIBLE_DEVICES=0
conda activate /share/hariharan/ck696/env_bh/anaconda/envs/allclear 
/share/hariharan/ck696/env_bh/anaconda/envs/allclear/bin/python train_0602.py --dataset_name "AllClear" --workers 8 --batch_size 6