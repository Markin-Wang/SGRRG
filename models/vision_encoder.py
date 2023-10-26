from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.forebacklearning import ForeBackLearning
from torch.cuda.amp import GradScaler, autocast, custom_fwd, custom_bwd
from math import inf
from timm.models.layers import trunc_normal_, to_2tuple
from modules.utils import load_embedding_layer, init_weights


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class VisionEncoder(nn.Module):
    def __init__(self, config):
        super(VisionEncoder, self).__init__()

        self.d_model = config['d_model']

        self.num_layers = config['num_layers_en']
        self.pe = config['pe']

        self.att_feat_size = config['d_vf']
        self.att_hid_size = config['d_model']

        self.input_encoding_size = config['d_model']
        self.drop_prob_lm = config['drop_prob_lm']

        self.use_sg = config['use_sg']
        self.sgave = config['sgave']

        if self.sgave:
            assert self.use_sg
            self.layers = clones(SceneGraphAidedEncoderLayer(config), self.num_layers)
        else:
            self.layers = clones(EncoderLayer(config), self.num_layers)
        self.norm = nn.LayerNorm(self.d_model)

        if self.pe == 'ape':
            self.ape = nn.Parameter(torch.zeros(1, self.num_patches, self.d_model))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        else:
            self.ape = None

    def forward(self, x, mask, sg_embeds=None, sg_masks=None):
        if self.ape is not None:
            x = x + self.ape
        for layer in self.layers:
            x, attn = layer(x, mask, sg_embeds, sg_masks)
        return self.norm(x)


class SceneGraphAidedEncoderLayer(nn.Module):
    def __init__(self, config):
        super(SceneGraphAidedEncoderLayer, self).__init__()
        self.d_model = config['d_model']
        self.num_heads = config['num_heads']
        self.pe = config['pe']
        self.d_ff = config['d_ff']
        self.dropout = config['dropout']

        self.self_attn = MultiHeadedAttention(self.num_heads, self.d_model, use_rpe=self.pe == 'rpe')
        self.cross_attn = MultiHeadedAttention(self.num_heads, self.d_model)
        self.feed_forward = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        self.sublayer = clones(SublayerConnection(self.d_model, self.dropout), 3)

    def forward(self, x, self_mask, sg_embed, cross_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, self_mask))
        # cross_mask bs x 1 x len_sg
        selected_bs = (cross_mask.squeeze(1) == 0).sum(-1) != 0
        # print(cross_mask[0])
        x[selected_bs] = self.sublayer[1](x[selected_bs],
                                          lambda x: self.cross_attn(x, sg_embed[selected_bs], sg_embed[selected_bs],
                                                                    cross_mask[selected_bs]))
        return self.sublayer[2](x, self.feed_forward), self.cross_attn.attn


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.d_model = config['d_model']
        self.num_heads = config['num_heads']
        self.pe = config['pe']
        self.d_ff = config['d_ff']
        self.dropout = config['dropout']

        self.self_attn = MultiHeadedAttention(self.num_heads, self.d_model, use_rpe=self.pe == 'rpe')
        self.feed_forward = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        self.sublayer = clones(SublayerConnection(self.d_model, self.dropout), 2)

    def forward(self, x, mask, sg_embeds=None, sg_masks=None):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward), self.self_attn.attn


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, use_rpe=False, height=7, width=7):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.use_rpe = use_rpe
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn, self.dropout = None, None
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        self.height, self.width = height, width

        if self.use_rpe:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * height - 1) * (2 * width - 1), h))  # 2*Wh-1 * 2*Ww-1, nH
            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(height)
            coords_w = torch.arange(width)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += height - 1  # shift to start from 0
            relative_coords[:, :, 1] += width - 1
            relative_coords[:, :, 0] *= 2 * width - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        d_k = query.size(-1)
        # with autocast(enabled=False):
        #     scores = (torch.matmul(query.float(), key.float().transpose(-2, -1)) / math.sqrt(d_k))
        scores = (torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k))

        if self.use_rpe:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.height * self.width, self.height * self.width, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            scores = scores + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # with autocast(enabled=False):
            # scores = scores.float().masked_fill(mask == 0, -inf)
            scores = scores + mask

        #     p_attn = F.softmax(scores, dim=-1)
        # else:
        p_attn = F.softmax(scores, dim=-1)
        # if value.dtype == torch.float16:
        #     p_attn = p_attn.half()
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)

        self.attn = p_attn.detach()
        x = torch.matmul(p_attn, value)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))
