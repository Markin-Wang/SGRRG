import torch.nn as nn
import copy
from modules.standard_trans import PositionwiseFeedForward, SublayerConnection
from modules.utils import init_weights
from config import catid2attrange
import numpy as np
import torch
import math
import torch.nn.functional as F

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class SceneGraphEncoder(nn.Module):
    def __init__(self, config):
        super(SceneGraphEncoder, self).__init__()
        # self.box_resolution_before = config['box_res_before'] # the image resolution used to do object detection
        self.feature_size = config['d_vf']  # feature size output by the visual extractor
        self.num_attributes = config['num_attributes']
        self.hidden_size = config['d_model']
        self.num_layers_sgen = config['num_layers_sgen']
        self.num_classes = config['num_classes']
        self.num_heads = config['num_heads']
        self.drop_prob_lm = config['dropout']
        self.d_ff = config['d_ff']
        # self.dropout = config['dropout']
        self.att_select_threshold = config['att_select_threshold']
        self.att_pad_idx = config['att_pad_idx']
        self.use_region_type_embed = config['use_region_type_embed']
        self.encode_type = config['encode_type']
        self.zero_count = 0

        if self.use_region_type_embed:
            self.token_type_embeddings = nn.Embedding(self.num_classes + 1, self.hidden_size)
            self.token_type_embeddings.apply(init_weights)
        else:
            self.token_type_embeddings = nn.Embedding(2, self.hidden_size)
            self.token_type_embeddings.apply(init_weights)

        self.proj = nn.Linear(self.feature_size, self.hidden_size)
        self.proj.apply(init_weights)

        self.pre_dropout = nn.Dropout(p=self.drop_prob_lm)
        self.pre_dropout.apply(init_weights)

        self.att_embedding = nn.Embedding(self.num_attributes + 1, self.hidden_size,
                                          padding_idx=self.num_attributes)  # the last one is for the mask
        self.att_embedding.apply(init_weights)


        self.catid2attrange = catid2attrange

        #self.obj_norm = nn.BatchNorm1d(self.hidden_size)

        self.att_norm = nn.LayerNorm(self.hidden_size)
        self.att_norm.apply(init_weights)

        self.sg_encoder = SGEncoder(config)
        self.sg_encoder.apply(init_weights)

        attribute_masks = torch.full((self.num_classes, self.num_attributes), False)
        for i in range(attribute_masks.shape[0]):
            attribute_masks[i][self.catid2attrange[i][0]:self.catid2attrange[i][1] + 1] = True

        self.register_buffer(
            "attribute_masks", attribute_masks, persistent=False
        )


    def forward(self, boxes, box_feats, box_labels, batch_size, att_ids=None, att_probs=None):
        if att_ids is None:
            # for inference, generate the att_ids from the att_probs
            assert att_probs is not None
            att_labels = att_probs > self.att_select_threshold
            att_masks = self.attribute_masks[box_labels]
            att_labels = att_labels & att_masks
            att_ids = self._prepare_att(att_labels)

        att_masks = torch.zeros((box_feats.shape[0], att_ids.shape[-1])).to(box_feats.device)
        att_masks[att_ids == self.att_pad_idx] = torch.finfo(box_feats.dtype).min

        att_ids[att_ids == self.att_pad_idx] = self.num_attributes  # change to the unused idx for attribute embedding

        if self.use_region_type_embed:
            att_type = torch.full((att_ids.shape[0], 1), self.num_classes, device=att_ids.device)
            obj_type = box_labels
        else:
            att_type = torch.full((att_ids.shape[0], 1), 1, device=att_ids.device)
            obj_type = torch.full_like(box_labels, 0, dtype=torch.long)

        att_embed = self.att_embedding(att_ids) + self.token_type_embeddings(att_type)
        # att_embed = self.pre_dropout(self.att_norm(att_embed))
        att_embed = self.pre_dropout(att_embed)

        obj_embeds = self.proj(box_feats) + self.token_type_embeddings(obj_type)
        #obj_embeds = self.obj_norm(obj_embeds)
        obj_embeds = obj_embeds.unsqueeze(1)
        obj_masks = torch.full((obj_embeds.shape[0], 1), 0, device=obj_embeds.device)

        if self.encode_type == 'oa-c':
            node_embeds = torch.cat((obj_embeds, att_embed), dim=1)
            node_masks = torch.cat((obj_masks, att_masks), dim=1).unsqueeze(1)  # [bs,1,L]
            sg_embeds = self.sg_encoder(node_embeds, node_masks)
        elif self.encode_type == 'oa-d':
            node_embeds = torch.cat((obj_embeds, att_embed), dim=1)
            node_masks = torch.cat((obj_masks, att_masks), dim=1).unsqueeze(1)  # [bs,1,L]
            sg_embeds = self.sg_encoder(obj_embeds, obj_masks,node_embeds,node_masks)
        elif self.encode_type== 'oa-dc':
            sg_embeds = self.sg_encoder(obj_embeds,obj_masks,att_embed,att_masks)
        else:
            raise NotImplementedError


        sg_embeds, sg_masks = self._to_bs_format(boxes[:, 0], sg_embeds, node_masks, batch_size)

        return sg_embeds, sg_masks

    def _prepare_att(self, att_labels):
        max_len = max(torch.sum(att_labels, dim=1))
        bs = att_labels.shape[0]
        att_ids = torch.full((bs, max_len), self.att_pad_idx, device=att_labels.device)
        for i in range(att_labels.shape[0]):
            idx = torch.nonzero(att_labels[i]).flatten()
            att_ids[i, :len(idx)] = idx

        return att_ids

    def _to_bs_format(self, bs_ids, node_embeds, node_masks, batch_size):
        max_len = -1
        num_nodes = torch.sum(node_masks == 0, dim=-1)
        node_masks = node_masks.squeeze(1)
        reformed_node_list = []
        for i in range(batch_size):
            cur_bs_ids = bs_ids == i
            cur_num_nodes, cur_node_masks = num_nodes[cur_bs_ids], node_masks[cur_bs_ids]
            max_len = max(sum(cur_num_nodes), max_len)
            cur_node_embeds = node_embeds[cur_bs_ids][cur_node_masks == 0]
            reformed_node_list.append(cur_node_embeds)

        reformed_node_embeds = torch.zeros(batch_size, max_len, node_embeds.shape[-1], device=node_embeds.device)
        reformed_node_masks = torch.full((batch_size, max_len), torch.finfo(node_embeds.dtype).min,
                                         device=node_embeds.device)
        for i in range(batch_size):
            if len(reformed_node_list[i]) == 0:
                self.zero_count += 1
                # print(f'{self.zero_count} samples without any regions.')
                #print(len(reformed_node_list[i]),reformed_node_masks.shape,reformed_node_list[i].shape if len(reformed_node_list[i])>0 else 0)
            reformed_node_embeds[i, :len(reformed_node_list[i])] = reformed_node_list[i]
            reformed_node_masks[i, :len(reformed_node_list[i])] = 0.0

        return reformed_node_embeds, reformed_node_masks.unsqueeze(1)

class SGEncoder(nn.Module):
    def __init__(self, config):
        super(SGEncoder, self).__init__()

        self.d_model = config['d_model']
        self.encode_type = config['encode_type']

        self.num_layers = config['num_layers_en']

        self.att_feat_size = config['d_vf']
        self.att_hid_size = config['d_model']

        self.input_encoding_size = config['d_model']
        self.drop_prob_lm = config['drop_prob_lm']

        if self.encode_type == 'oa-c':
            # object-attribute coupled: node-att to node-att
            self.layers = clones(SGOACEncoderLayer(config), self.num_layers)
        elif self.encode_type == 'oa-d' or self.encode_type == 'oa-dc':
            # object-attribute decomposed: node to node-att then nodes to nodes
            # object-attribute decomposed completely: node to att then nodes to nodes
            self.layers = clones(SGOADEncoderLayer(config), self.num_layers)
        else:
            raise NotImplementedError
        self.norm = nn.LayerNorm(self.d_model)

        self.ape = None

    def forward(self, x, mask, sg_embeds=None,sg_masks=None):
        if self.ape is not None:
            x = x + self.ape
        for layer in self.layers:
            x, attn = layer(x, mask,sg_embeds, sg_masks)
        return self.norm(x)

class SGOADEncoderLayer(nn.Module):
    def __init__(self, config):
        super(SGOADEncoderLayer, self).__init__()
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
        x = self.sublayer[1](x, lambda x: self.cross_attn(x, sg_embed, sg_embed, cross_mask))
        return self.sublayer[2](x, self.feed_forward), self.cross_attn.attn


class SGOACEncoderLayer(nn.Module):
    def __init__(self, config):
        super(SGOACEncoderLayer, self).__init__()
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

