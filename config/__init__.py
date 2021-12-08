import argparse
import os
import torch
from torch import cuda
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    # training experment name
    parser.add_argument('--test_name', type=str, default='rerank-topk')

    # model related params
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--mlp_ratio', type=int, default=4)
    parser.add_argument('--qkv_bias', action='store_true')
    parser.add_argument('--drop_rate', type=float, default=0.0)
    parser.add_argument('--drop_path_rate', type=float, default=0.0)
    parser.add_argument('--attn_drop_rate', type=float, default=0.1)

    # training specific args
    parser.add_argument('--dataset_names', type=str, default='[rSfM120k-tl-resnet101-gem-w]')
    parser.add_argument('--resume', type=str, default=None)

    parser.add_argument('--topk', type=int, default=128)
    parser.add_argument('--sim_len', type=int, default=256)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=512)

    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda' if cuda.is_available() else 'cpu')

    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--stop_at_epoch', type=int, default=None)

    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--warmup_lr', type=float, default=0)
    parser.add_argument('--base_lr', type=float, default=0.01)
    parser.add_argument('--final_lr', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    parser.add_argument('--clip_max_norm', type=float, default=0)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--rank', type=int, default=None)
    parser.add_argument('--world_size', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--dist_backend', type=str, default='nccl')
    parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:29507')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--update_every', type=int, default=1)
    args = parser.parse_args()

    if args.stop_at_epoch is not None:
        if args.stop_at_epoch > args.num_epochs:
            raise Exception
    else:
        args.stop_at_epoch = args.num_epochs

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    return args
