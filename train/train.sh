#!/usr/bin/env python

init_from="" # Model path downloaded from huggingface
data_config=configs/data_Omni_center_512.yaml # OmniEdit training data config
lr=2e-5
wd=0.1
dropout=0.05
batchsize_per_gpu=2
max_seq_len=5120 # text + image(4096 for 1024 * 1024 resolution) --> token length
task=CelebA_New
exp_name=DDB
echo "exp name: $exp_name  node: $SLURMD_NODENAME"
mkdir -p output/"$task"/"$exp_name"
export CUDA_VISIBLE_DEVICES=4,5,6,7
torchrun --nproc_per_node=4 --master_port=25639 --nnodes=1 train/train_source_infor_new.py \
--batch_size ${batchsize_per_gpu} \
--accum_iter 4 \
--epochs 10 \
--warmup_epochs 0.001 \
--lr ${lr} \
--min_lr ${lr} \
--wd ${wd} \
--ckpt_max_keep 100 \
--clip_grad 4 \
--data_config $data_config \
--cache_ann_on_disk \
--num_workers 16 \
--output_dir output/"$task"/"$exp_name" \
--save_iteration_interval 1000 \
--max_seq_len ${max_seq_len} \
--dropout ${dropout} \
--init_from ${init_from} \
--mix_ratio 0.5 \
--mask_random_ratio 0.3 \
--info_metric "edit_diff" \
2>&1 | tee -a output/"$task"/"$exp_name"/output.log

echo "exp name: $exp_name"
