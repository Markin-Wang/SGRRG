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
        self.hierarchical_attention = config['hierarchical_attention']

        if self.sgade:
            assert self.use_sg
            self.layers = clones(SceneGraphAidedDecoderLayer(config), self.num_layers)
        else:
            self.layers = clones(DecoderLayer(config), self.num_layers)
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self, x, img_feats, self_mask, cross_mask, sg_embeds=None, sg_masks=None, past_data=None):
        # past data is a dict
        attns = []
        for layer in self.layers:
            if self.sgade:
                x, attn = layer(x, img_feats, self_mask, cross_mask, sg_embeds, sg_masks, past_data)
            else:
                x, attn = layer(x, img_feats, self_mask, cross_mask)

            # attns.append(attn)
        return self.norm(x), attns

    def _to_bs_format_pool(self, bs_ids, node_embeds, node_masks, batch_size):

        bs_len = [(bs_ids == i).sum() for i in range(batch_size)]
        node_masks = node_masks == 0

        node_embeds = [torch.max(node_embeds[i, node_masks[i, 0]], dim=0, keepdim=True)[0]
                       # if self.pooling == 'max' else
                       # torch.mean(node_embeds[i, node_masks[i, 0]], dim=0, keepdim=True)
                       for i in range(node_embeds.shape[0])
                       ]
        node_embeds = torch.cat(node_embeds, dim=0)

        max_len = max(bs_len)
        reformed_node_embeds = torch.zeros(batch_size, max_len, node_embeds.shape[-1], device=node_embeds.device)
        reformed_node_masks = torch.full((batch_size, max_len), torch.finfo(node_embeds.dtype).min,
                                         device=node_embeds.device)
        for i in range(batch_size):
            reformed_node_embeds[i, :bs_len[i]] = node_embeds[bs_ids == i]
            reformed_node_masks[i, :bs_len[i]] = 0.0

        return reformed_node_embeds, reformed_node_masks.unsqueeze(1)


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
            self.aggregate_attn = AggregateTopAttention(self.d_model, h=self.num_heads)
            # self.query_proj = nn.Linear(self.d_model,self.d_model)
            # self.score_proj = nn.Linear(self.d_model, 1)
        else:
            self.cross_attn_sg = MultiHeadedAttention(self.num_heads, self.d_model)

        self.feed_forward = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        self.sublayer = clones(SublayerConnection(self.d_model, self.dropout), 4)

    def forward(self, x, img_feats, self_mask, img_masks, sg_embeds, sg_masks, past_data):

        bs_ids, selected_bs = past_data['bs_ids'], past_data['selected_bs']

        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, self_mask))

        x_img = self.sublayer[1](x, lambda x: self.cross_attn_img(x, img_feats, img_feats, img_masks))

        if self.hierarchical_attention:
            sg_embeds_pool, sg_embeds_pool_masks = past_data['sg_embeds_pool'], past_data['sg_embeds_pool_masks']
            if self.fuse_opt == 'add':
                x_ = x[bs_ids.long()]  # expand to make each sub-graph in a sample has the same image query
            else:
                x_ = x_img[bs_ids.long()]


            x_ = self.cross_attn_subgraph(x_, sg_embeds, sg_embeds, sg_masks)
            # [num_boxes, num_token, hs]

            if 'num_nodes' not in past_data:
                num_nodes= [(bs_ids == i).sum() for i in range(x_.shape[0])]
                max_node =  max(num_nodes)
                past_data.update({'num_nodes':num_nodes, 'max_node':max_node})
            else:
                num_nodes, max_node = past_data['num_nodes'], past_data['max_node']

            if 'pre_tsg_masks' not in past_data or past_data['pre_tsg_masks'].size(0) != x.size(0):
                # the second condition is for inference phase where the first forward has no beam search
                # [bs, num_token, num_subgraph, hs]
                sg_embeds = torch.zeros((x.shape[0], max_node, x.shape[1], x_.shape[-1]), device=x_.device,
                                        dtype=x_.dtype)
                pre_tsg_masks = torch.full((x.shape[0], max_node), torch.finfo(sg_embeds.dtype).min,
                                           device=sg_embeds.device)

                for i in range(x.shape[0]):
                    # print(x_[bs_ids == i].shape, sg_embeds.shape, sg_embeds_avg.shape)
                    sg_embeds[i, :num_nodes[i]] = x_[bs_ids == i]
                    pre_tsg_masks[i, :num_nodes[i]] = 0.0

                past_data['pre_tsg_masks'] = pre_tsg_masks

            else:
                pre_tsg_masks = past_data['pre_tsg_masks']
                sg_embeds = torch.zeros((x.shape[0] * max_node, x.shape[1], x_.shape[-1]), device=x_.device,
                                        dtype=x_.dtype)
                sg_embeds[(pre_tsg_masks == 0).view(-1)] = x_
                sg_embeds = sg_embeds.view(x.shape[0], max_node, x.shape[1], x_.shape[-1])
            # only perform attention for those samples having the sg

            if self.fuse_opt == 'add':
                x_ = self.sublayer[2](x[selected_bs], lambda x: self.aggregate_attn(x, sg_embeds_pool[selected_bs],
                                                                        sg_embeds[selected_bs],
                                                                        sg_embeds_pool_masks[selected_bs]))
                x_img[selected_bs] = x_img[selected_bs] + x_

            else:
                x_ = self.sublayer[2](x_img[selected_bs], lambda x: self.aggregate_attn(x, sg_embeds_pool[selected_bs],
                                                                        sg_embeds[selected_bs],
                                                                        sg_embeds_pool_masks[selected_bs]))
                x_img[selected_bs] = x_

        else:
            # selected_bs = (sg_masks.squeeze(1) == 0).sum(-1) != 0
            # cross_mask bs x 1 x len_sg
            if self.fuse_opt == 'att':
                x_, sg_embeds_, sg_masks_ = x_img[selected_bs], sg_embeds[selected_bs], sg_masks[selected_bs]
                x_ = self.sublayer[2](x_, lambda x: self.cross_attn_sg(x, sg_embeds_, sg_embeds_, sg_masks_))
                x_img[selected_bs] = x_

            elif self.fuse_opt == 'cat':
                x_, sg_embeds_, sg_masks_ = x_img[selected_bs], sg_embeds[selected_bs], sg_masks[selected_bs]
                x_ = self.sublayer[2](x_, lambda x: self.cross_attn_sg(x, sg_embeds_, sg_embeds_, sg_masks_))
                x_img[selected_bs] = self.fuse_proj(torch.cat([x[selected_bs], x_], dim=-1)).to(dtype=x.dtype)

            elif self.fuse_opt == 'add':
                x_img = self.sublayer[1](x, lambda x: self.cross_attn_img(x, img_feats, img_feats, img_masks))
                x_, sg_embeds_, sg_masks_ = x[selected_bs], sg_embeds[selected_bs], sg_masks[selected_bs]
                x_ = self.sublayer[2](x_, lambda x: self.cross_attn_sg(x, sg_embeds_, sg_embeds_, sg_masks_))
                x_img[selected_bs] = x_img[selected_bs] + x_

        return self.sublayer[-1](x_img, self.feed_forward), self.cross_attn_img.attn


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


# testing time unacceptable
class AggregateAttention(nn.Module):
    def __init__(self, d_model, h=1, dropout=0.1):
        super(AggregateAttention, self).__init__()
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                      zip(self.linears, (query, key))]

        # from [bs, max_nodes, num_len, head, dhs] to [bs, num_head, num_len, max_node, dhs]
        # print(1111,value.shape)
        value = self.linears[2](value).view(nbatches, value.size(1), value.size(2), self.h, self.d_k).transpose(1, 3)
        # print(2222,value.shape)

        # x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        d_k = query.size(-1)

        scores = (torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k))

        if mask is not None:
            scores = scores + mask

        p_attn = F.softmax(scores, dim=-1)

        if self.dropout is not None:
            p_attn = self.dropout(p_attn)

        # print(3333, p_attn.shape, value.shape)

        # self.attn = p_attn

        # p_attn to [bs, num_head, num_token, 1, max_node]
        x = torch.matmul(p_attn.unsqueeze(-2), value).squeeze(-2)
        # x [bs, num_head, num_token, 1, d_model]

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # print(4444,x.shape)
        return self.linears[-1](x)

class AggregateTopAttention(nn.Module):
    def __init__(self, d_model, h=1, dropout=0.1):
        super(AggregateTopAttention, self).__init__()
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                      zip(self.linears, (query, key))]

        # from [bs, max_nodes, num_len, head, dhs] to [bs, num_head, num_len, max_node, dhs]
        # print(1111,value.shape)
        value = self.linears[2](value).view(nbatches, value.size(1), value.size(2), self.h, self.d_k).transpose(1, 3)
        # print(2222,value.shape)

        # x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        d_k = query.size(-1)

        scores = (torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k))

        if mask is not None:
            scores = scores + mask

        # p_attn = F.softmax(scores, dim=-1)

        if self.dropout is not None:
            scores = self.dropout(scores)


        _, max_index =  torch.max(scores,dim=-1)

        index = max_index.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,-1,-1,value.size(-1))

        value = torch.gather(value, 2, index=index).squeeze(-2)

        # x [bs, num_head, num_token, 1, d_model]

        value = value.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](value)


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
