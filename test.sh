#!/bin/bash
python test.py \
--test_name gl18-tl-resnet101-gem-w \
--resume None \
--depth 2 \
--embed_dim 768 \
--num_heads 12 \
--topk 1024 \
--sim_len 512 \
--mlp_ratio 4 \
--qkv_bias \
--drop_rate 0.1 \
--drop_path_rate 0.1 \
--attn_drop_rate 0.1 \
--device cuda \