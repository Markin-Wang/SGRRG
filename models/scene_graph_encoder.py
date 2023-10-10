import torch.nn as nn
import copy
from modules.standard_trans import MultiHeadedAttention, PositionwiseFeedForward,SublayerConnection, Encoder
from modules.utils import init_weights
from config import catid2attrange
import numpy as np

class SceneGraphEncoder(nn.Module):
    def __init__(self,config):
        super(SceneGraphEncoder, self).__init__()
        # self.box_resolution_before = config['box_res_before'] # the image resolution used to do object detection
        self.feature_size = config['d_vf'] # feature size output by the visual extractor
        self.num_attributes = config['num_attributes']
        self.hidden_size = config['d_model']
        self.num_layers_sgen = config['num_layers_sgen']
        self.num_classes = config['num_classes']
        self.num_heads = config['num_heads']

        self.token_type_embeddings = nn.Embedding(self.num_classes+1,self.hidden_size)
        self.token_type_embeddings.apply(init_weights)

        self.proj = nn.Sequential(
            nn.Linear(self.feature_size,self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.GELU(),
        )

        self.proj.apply(init_weights)

        self.d_ff = config['d_ff']
        self.dropout = config['dropout']
        self.att_select_threshold = config['att_select_threshold']
        self.att_pad_idx = config['att_pad_idx']

        self.att_embedding = nn.Embedding(self.num_attributes,self.hidden_size)

        self.att_embedding.apply(init_weights)

        self.catid2attrange = catid2attrange

        self.attribute_mask = np.full((self.num_classes,self.num_attributes),False)
        for i in range(self.attribute_mask.shape[0]):
            self.attribute_mask[i][self.catid2attrange[i]] = True

        attention = MultiHeadedAttention(self.num_heads, self.hidden_size, use_rpe=None)
        ff = PositionwiseFeedForward(self.hidden_size, self.d_ff, self.dropout)

        self.sg_encoder = Encoder(EncoderLayer(self.hidden_size, attention, ff, self.dropout), self.num_layers_sgen,ape=None)
        self.sg_encoder.apply(init_weights)


    def forward(self,boxes,box_feats,box_labels,att_ids=None,att_probs=None):
        if att_ids is None:
            assert att_logits is not None
            att_labels = att_probs > self.att_select_threshold
            att_masks = self.attribute_mask[box_labels]
            att_labels = att_labels & att_masks
            att_ids = self._prepare_att(att_labels)
        att_masks = torch.zeros((bs, max_len)).to(box_feats.device)
        att_masks[att_ids == self.att_pad_idx] = torch.finfo(type).min

        att_embed = self.att_embedding(att_ids) + self.token_type_embeddings(self.num_classes)

        obj_embed = self.proj(box_feats) + self.token_type_embeddings(box_labels.view(1,-1))

        node_embed = torch.cat((obj_embed,att_embed),dim=1)

        obj_masks = att_masks.full(node_embed.shape[0,],1)

        node_masks = torch.cat((obj_masks,att_masks),dim=1)

        sg_embed = self.sg_encoder(node_embed,node_masks)

        return sg_embed

    def _prepare_att(self,att_labels):
        max_len = torch.sum(att_labels,dim=1)
        bs = att_labels.shape[0]
        att_ids = torch.full((bs, max_len), self.att_pad_idx)
        for i in range(att_labels.shape[0]):
            idx = torch.nonzeros(att_labels[i]).flatten()
            att_ids[i,len(idx)] = idx

        return att_ids











