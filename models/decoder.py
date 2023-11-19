from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast, custom_fwd, custom_bwd
from math import inf
from timm.models.layers import trunc_normal_, to_2tuple
from modules.utils import load_embedding_layer, init_weights


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.d_model = config['d_model']

        self.num_layers = config['num_layers_de']

        self.att_feat_size = config['d_vf']
        self.att_hid_size = config['d_model']

        self.input_encoding_size = config['d_model']
        self.drop_prob_lm = config['drop_prob_lm']
        self.use_sg = config['use_sg']
        self.sgade = config['sgade']
        self.hierarchical_attentino = config['hierarchical_attention']

        if self.sgade:
            assert self.use_sg
            self.layers = clones(SceneGraphAidedDecoderLayer(config), self.num_layers)
        else:
            self.layers = clones(DecoderLayer(config), self.num_layers)
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self, x, img_feats, self_mask, cross_mask, sg_embeds=None, sg_masks=None, bs_ids=None):
        attns = []
        pre_tsg_masks, max_nodes, sg_embeds_avg = None, None, None
        for layer in self.layers:
            if self.sgade:
                if self.hierarchical_attentino:
                    sg_masks_index = sg_masks == 0

                    sg_embeds_avg = [torch.max(sg_embeds[i, sg_masks_index[i, 0]], dim=0, keepdim=True)[0]
                                     if self.pooling == 'max' else
                                     torch.mean(sg_embeds[i, sg_masks_index[i, 0]], dim=0, keepdim=True)
                                     for i in range(sg_embeds.shape[0])
                                     ]
                    sg_embeds_avg = torch.cat(sg_embeds_avg, dim=0)
                x, attn, max_nodes, pre_tsg_masks = layer(x, img_feats, self_mask, cross_mask, sg_embeds, sg_masks,
                                                          bs_ids, sg_embeds_avg, pre_tsg_masks, max_nodes)
            else:
                x, attn = layer(x, img_feats, self_mask, cross_mask)

            # attns.append(attn)
        return self.norm(x), attns


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()

        self.d_model = config['d_model']
        self.num_heads = config['num_heads']
        self.pe = config['pe']
        self.d_ff = config['d_ff']
        self.dropout = config['dropout']

        self.self_attn = MultiHeadedAttention(self.num_heads, self.d_model)
        self.cross_attn = MultiHeadedAttention(self.num_heads, self.d_model)
        self.feed_forward = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        self.sublayer = clones(SublayerConnection(self.d_model, self.dropout), 3)

    def forward(self, x, img_feats, self_mask, cross_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, self_mask))
        x = self.sublayer[1](x, lambda x: self.cross_attn(x, img_feats, img_feats, cross_mask))
        return self.sublayer[2](x, self.feed_forward), self.cross_attn.attn


class SceneGraphAidedDecoderLayer(nn.Module):
    def __init__(self, config):
        super(SceneGraphAidedDecoderLayer, self).__init__()

        self.d_model = config['d_model']
        self.num_heads = config['num_heads']
        self.pe = config['pe']
        self.d_ff = config['d_ff']
        self.dropout = config['dropout']
        self.fuse_opt = config['fuse_opt']
        self.hierarchical_attention = config['hierarchical_attention']
        if self.fuse_opt == 'cat':
            self.fuse_proj = nn.Linear(self.d_model * 2, self.d_model)

        self.self_attn = MultiHeadedAttention(self.num_heads, self.d_model)
        self.cross_attn_img = MultiHeadedAttention(self.num_heads, self.d_model)

        if self.hierarchical_attention:
            self.cross_attn_subgraph = MultiHeadedAttention(self.num_heads, self.d_model)
            # self.cross_attn_whole = MultiHeadedAttention(self.num_heads, self.d_model)
            self.aggregate_attn = AggregateAttention(self.d_model, h=1)
        else:
            self.cross_attn_sg = MultiHeadedAttention(self.num_heads, self.d_model)

        self.feed_forward = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        self.sublayer = clones(SublayerConnection(self.d_model, self.dropout), 4)

    def forward(self, x, img_feats, self_mask, img_masks, sg_embeds, sg_masks, bs_ids=None, sg_embeds_avg=None,
                pre_tsg_masks=None, max_nodes=None):

        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, self_mask))

        x = self.sublayer[1](x, lambda x: self.cross_attn_img(x, img_feats, img_feats, img_masks))

        if self.hierarchical_attention:
            x_ = x[bs_ids.long()]  # expand to make each sub-graph in a sample has the same image query
            x_ = self.cross_attn_subgraph(x_, sg_embeds, sg_embeds, sg_masks)
            # [num_boxes, num_token, hs]
            if pre_tsg_masks is None:
                num_nodes = [(bs_ids == i).sum() for i in range(x.shape[0])]
                max_nodes = max(num_nodes)
                # [bs, num_token, num_subgraph, hs]
                sg_embeds = torch.zeros((x.shape[0], x.shape[1], max_nodes, x_.shape[-1]), device=x_.device, dtype=x_.dtype)
                pre_tsg_masks = torch.full((x.shape[0], x.shape[1], max_nodes), torch.finfo(sg_embeds.dtype).min,
                                           device=sg_embeds.device)
                for i in range(x.shape[0]):
                    print(x_[bs_ids == i].shape, sg_embeds.shape)
                    sg_embeds[i, :, :num_nodes[i]] = x_[bs_ids == i]
                    pre_tsg_masks[i, :, :num_nodes[i]] = 0.0

            else:
                sg_embeds = torch.zeros((x.shape[0], x.shape[1], max_nodes, x_.shape[-1]), device=x_.device, dtype=x_.dtype)
                sg_embeds[(pre_tsg_masks == 0).view(-1)] = x_
                sg_embeds = sg_embeds.view(x.shape[0], max_nodes, x_.shape[-1])

            x_ = self.sublayer[2](x, lambda x: self.aggregate_attn(x, sg_embeds_avg, sg_embeds,
                                                                      pre_tsg_masks.unsqueeze(1)))
            selected_bs = pre_tsg_masks.sum(-1) != 0
            x[selected_bs] = x_

        else:
            selected_bs = (sg_masks.squeeze(1) == 0).sum(-1) != 0
            # cross_mask bs x 1 x len_sg
            x_, sg_embeds_, sg_masks_ = x[selected_bs], sg_embeds[selected_bs], sg_masks[selected_bs]
            x_ = self.sublayer[2](x_, lambda x: self.cross_attn_sg(x, sg_embeds_, sg_embeds_, sg_masks_))
            if self.fuse_opt == 'att':
                x[selected_bs] = x_
            elif self.fuse_opt == 'cat':
                x[selected_bs] = self.fuse_proj(torch.cat([x[selected_bs], x_], dim=-1)).to(dtype=x.dtype)

        return self.sublayer[3](x, self.feed_forward), self.cross_attn_img.attn, max_nodes, pre_tsg_masks


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

        self.attn = p_attn
        x = torch.matmul(p_attn, value)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class AggregateAttention(nn.Module):
    def __init__(self, d_model, h=1, dropout=0):
        super(AggregateAttention, self).__init__()
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 3)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))]

        # x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        d_k = query.size(-1)

        scores = (torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k))

        if mask is not None:
            scores = scores + mask

        p_attn = F.softmax(scores, dim=-1)

        if self.dropout is not None:
            p_attn = self.dropout(p_attn)

        # self.attn = p_attn

        x = torch.matmul(p_attn, value)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
