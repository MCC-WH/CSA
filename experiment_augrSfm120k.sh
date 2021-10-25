#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py \
--test_name TopkSIM_SFMAUG \
--input_dim 2048 \
--embed_dim 768 \
--num_heads 12 \
--depth 2 \
--topk 512 \
--test_topk 1024 \
--sim_len 512 \
--batch_size 256 \
--num_workers 4 \
--mlp_ratio 4 \
--qkv_bias \
--alpha 0.2 \
--drop_rate 0.1 \
--drop_path_rate 0.1 \
--attn_drop_rate 0.1 \
--dataset_names [gl18-tl-resnet50-gem-w,gl18-tl-resnet101-gem-w,gl18-tl-resnet152-gem-w] \
--device cuda \
--num_epochs 100 \
--stop_at_epoch 100 \
--warmup_epochs 20 \
--warmup_lr 0.001 \
--base_lr 0.1 \
--final_lr 0 \
--momentum 0.9 \
--weight_decay 0.000001 \
--update_every 1 \
--dist_url tcp://127.0.0.1:29501 \












