import torch.nn as nn
import copy
from modules.standard_trans import MultiHeadedAttention, PositionwiseFeedForward, SublayerConnection, Encoder, \
    EncoderLayer
from modules.utils import init_weights
from config import catid2attrange
import numpy as np
import torch


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


        self.zero_count = 0

        if self.use_region_type_embed:
            self.token_type_embeddings = nn.Embedding(self.num_classes + 1, self.hidden_size)
            self.token_type_embeddings.apply(init_weights)
        else:
            self.token_type_embeddings = nn.Embedding(2, self.hidden_size)
            self.token_type_embeddings.apply(init_weights)


        self.proj = nn.Linear(self.feature_size, self.hidden_size)

        self.pre_dropout = nn.Dropout(p=self.drop_prob_lm)

        self.proj.apply(init_weights)

        self.att_embedding = nn.Embedding(self.num_attributes + 1, self.hidden_size,
                                          padding_idx=self.num_attributes)  # the last one is for the mask

        self.att_embedding.apply(init_weights)

        self.catid2attrange = catid2attrange

        #self.obj_norm = nn.BatchNorm1d(self.hidden_size)

        self.att_norm = nn.LayerNorm(self.hidden_size)

        attention = MultiHeadedAttention(self.num_heads, self.hidden_size, use_rpe=None)
        ff = PositionwiseFeedForward(self.hidden_size, self.d_ff, self.drop_prob_lm)

        self.sg_encoder = Encoder(EncoderLayer(self.hidden_size, attention, ff, self.drop_prob_lm),
                                  self.num_layers_sgen, ape=None)
        self.sg_encoder.apply(init_weights)

        attribute_masks = torch.full((self.num_classes, self.num_attributes), False)
        for i in range(attribute_masks.shape[0]):
            attribute_masks[i][self.catid2attrange[i][0]:self.catid2attrange[i][1] + 1] = True

        self.register_buffer(
            "attribute_masks", attribute_masks, persistent=False
        )

    def forward(self, boxes, box_feats, box_labels, batch_size, att_ids=None, att_probs=None):
        if att_ids is None:
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

        att_embed = self.pre_dropout(self.att_norm(att_embed))

        obj_embeds = self.proj(box_feats) + self.token_type_embeddings(obj_type)

        #obj_embeds = self.obj_norm(obj_embeds)

        obj_embeds = obj_embeds.unsqueeze(1)

        node_embeds = torch.cat((obj_embeds, att_embed), dim=1)

        obj_masks = torch.full((node_embeds.shape[0], 1), 0, device=node_embeds.device)

        node_masks = torch.cat((obj_masks, att_masks), dim=1).unsqueeze(1)  # [bs,1,L]

        sg_embeds = self.sg_encoder(node_embeds, node_masks)

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
            #print(len(reformed_node_list[i]),reformed_node_masks.shape,reformed_node_list[i].shape if len(reformed_node_list[i])>0 else 0)
            reformed_node_embeds[i, :len(reformed_node_list[i])] = reformed_node_list[i]
            reformed_node_masks[i, :len(reformed_node_list[i])] = 0.0

        return reformed_node_embeds, reformed_node_masks.unsqueeze(1)
