import argparse
import os
import pickle
from collections import OrderedDict

from torch.utils import data
from utils.helpfunc import get_data_root

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from Dataset import FeatureFromList
from network import RerankTransformer
from utils import compute_map_and_print, load_pickle

DATASETS = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k']


@torch.no_grad()
def test(datasets, name, net, topk=128, anchor_len=512, device=torch.device('cuda'), R1M=True):
    net.eval()
    for dataset in datasets:
        # prepare config structure for the test dataset
        dataset = dataset.lower()
        if dataset not in DATASETS:
            raise ValueError('Unknown dataset: {}!'.format(dataset))
        gnd_fname = os.path.join(get_data_root(), 'annotations', dataset, 'gnd_{}.pkl'.format(dataset))
        cfg = load_pickle(gnd_fname)
        feature_prefix = os.path.join(get_data_root(), 'test', dataset, '{}.pkl'.format(name))
        if os.path.exists(feature_prefix):
            feature = load_pickle(feature_prefix)
            vecs = torch.tensor(feature['db']).float()
            qvecs = torch.tensor(feature['query']).float()
            if R1M:
                R1M_feature_prefix = os.path.join(get_data_root(), 'r1m', '{}.pkl'.format(name))
                if os.path.exists(R1M_feature_prefix):
                    R1M_feature = torch.tensor(load_pickle(R1M_feature_prefix)).float()
                    vecs = torch.cat((vecs, R1M_feature), dim=0)
                else:
                    raise ValueError('Prepare {} R1M feature first'.format(dataset))

            scores = torch.mm(vecs, qvecs.t())
            ranks = torch.argsort(scores, dim=0, descending=True)
            _, _, _ = compute_map_and_print(dataset, 'fist-stage', name, ranks.numpy(), cfg['gnd'])

            query_topk_indices = torch.Tensor(ranks[:topk, :]).t().long()
            anchor_indices = torch.Tensor(ranks[:anchor_len - 1, :]).t().long()
            loader = DataLoader(FeatureFromList(features=vecs, topk_indices=query_topk_indices),
                                batch_size=1,
                                shuffle=False,
                                num_workers=8,
                                pin_memory=True)
            rerank_scores = torch.zeros(qvecs.size(0), topk)
            for i, input in enumerate(loader):
                input = torch.cat((qvecs[i].unsqueeze(0).unsqueeze(0), input), dim=1)
                anchor_feature = vecs[anchor_indices[i]].unsqueeze(0)
                anchor_feature = torch.cat((qvecs[i].unsqueeze(0).unsqueeze(0), anchor_feature), dim=1)
                affinity_feature = torch.bmm(input, anchor_feature.permute(0, 2, 1))
                rerank_scores[i, :] = net.forward_feature(affinity_feature.to(device)).data.squeeze().cpu()
            rerank_indices = np.argsort(-rerank_scores.numpy(), axis=1)
            for i in range(rerank_scores.size(0)):
                ranks[:topk, i] = ranks[:topk, i][rerank_indices[i]]
            _, _, _ = compute_map_and_print(dataset, 'rerank-top{}'.format(topk), name, ranks, cfg['gnd'])
        else:
            raise ValueError('Prepare {} feature first'.format(dataset))


def main(args):
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    model = RerankTransformer(embed_dim=args.embed_dim,
                              topk_dim=args.anchor_len,
                              depth=args.depth,
                              num_heads=args.num_heads,
                              mlp_ratio=args.mlp_ratio,
                              qkv_bias=args.qkv_bias,
                              drop_rate=args.drop_rate,
                              attn_drop_rate=args.attn_drop_rate,
                              drop_path_rate=args.drop_path_rate).to(device)

    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(">> Loading checkpoint:\n>> '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print(">>>> loaded checkpoint:\n>>>> '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            raise ValueError(">> No checkpoint found at '{}'".format(args.resume))

    test(datasets=['roxford5k', 'rparis6k'], name=args.test_name, net=model, topk=args.topk, anchor_len=args.anchor_len, device=device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Testing End-to-End')

    # test experment name
    parser.add_argument('--test_name', type=str, default='rSfM120k-tl-resnet101-gem-w')

    # model related params
    parser.add_argument('--topk', type=int, default=128)
    parser.add_argument('--anchor_len', type=int, default=512)
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--mlp_ratio', type=int, default=4)
    parser.add_argument('--qkv_bias', action='store_true')
    parser.add_argument('--drop_rate', type=float, default=0.0)
    parser.add_argument('--drop_path_rate', type=float, default=0.0)
    parser.add_argument('--attn_drop_rate', type=float, default=0.1)

    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    main(args)
