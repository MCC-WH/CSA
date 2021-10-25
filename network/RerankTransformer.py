import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp.autocast_mode import autocast
from .Transformer_Block import Block, trunc_normal_, Feature_align


class RerankTransformer(nn.Module):
    def __init__(self, topk_dim=512, embed_dim=2048, depth=6, num_heads=4, mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm):
        super(RerankTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.outputdim = embed_dim
        self.t = 2.0

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.sim_drop = nn.Dropout(p=drop_rate)
        self.topk_proj = nn.Linear(in_features=topk_dim, out_features=embed_dim, bias=True)
        self.feature_align = Feature_align(embed_dim, embed_dim, topk_dim, 1)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer) for i in range(depth)])
        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'feature_align'}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @autocast()
    def forward_feature(self, x_topk: Tensor):
        topk_embed = self.topk_proj(self.sim_drop(x_topk))
        x = self.pos_drop(topk_embed)
        for blk in self.blocks:
            x = blk(x)
        norm_x = F.normalize(x, dim=-1)
        exp_sim = torch.sum(norm_x[:, 0, :].unsqueeze(1) * norm_x[:, 1:, :], dim=-1)
        return exp_sim

    @autocast()
    def forward(self, x_topk: Tensor):
        residual = x_topk
        topk_embed = self.topk_proj(self.sim_drop(x_topk))
        x = self.pos_drop(topk_embed)
        for blk in self.blocks:
            x = blk(x)
        norm_x = F.normalize(x, dim=-1)
        exp_sim = torch.exp(torch.sum(norm_x[:, 0, :].unsqueeze(1) * norm_x, dim=-1) / self.t)
        x = self.feature_align(x)
        MSE_loss = torch.pow(F.normalize(residual, dim=-1) - F.normalize(x, dim=-1), 2).sum(dim=-1).clamp(min=1e-4, max=1e4).sqrt().mean()
        return exp_sim, MSE_loss