import torch.nn as nn
from modules.utils import init_weights
import torch
from .vision_encoder import clones
import math
import torch.nn.functional as F

class RegionSelector(nn.Module):
    def __init__(self, config):
        super(RegionSelector, self).__init__()
        self.feature_size = config['d_vf']
        self.use_mem = config['use_mem_r']
        self.drop_prob = config['dropout']
        # memory settings
        if self.use_mem:
            self.topk = config['topk']
            self.cmn = MultiThreadMemory(config['num_heads_r'], self.feature_size, topk=self.topk)
            self.memory = nn.Parameter(torch.FloatTensor(config['num_mem'], self.feature_size))
            self.mem_proj = nn.Linear(self.feature_size, self.feature_size)
            self.fuse_proj = nn.Sequential(
            nn.Linear(config['d_vf'] * 2, config['d_vf']),
            nn.GELU(),
            nn.Dropout(p=self.drop_prob),
        )
            self.cmn.apply(init_weights)
            nn.init.normal_(self.memory , 0, 1 / self.feature_size)
            self.mem_proj.apply(init_weights)
            self.fuse_proj.apply(init_weights)
            self.ff = nn.Linear(config['d_vf'], config['d_vf'])
            self.ff.apply(init_weights)
        # else:
        #     self.ff = nn.Sequential(
        #         nn.Linear(config['d_vf'], config['d_vf']),
        #         nn.GELU(),
        #         nn.Dropout(p=self.drop_prob),
        #     )



        self.region_head = nn.Linear(config['d_vf'], config['num_classes'])
        self.region_head.apply(init_weights)

        self.region_select_threshold = config['region_select_threshold']

    def forward(self, x, boxes=None, box_labels=None, box_masks=None):
        x = torch.mean(x, -2)

        if self.use_mem:
            x = self.ff(x)
            x = x.unsqueeze(1)
            mem = self.mem_proj(self.memory)
            #mem = mem.unsqueeze(0).expand(x.size(0),*mem.shape)
            # can broadcast in batch dimension
            responses = self.cmn(x, mem.unsqueeze(0), mem.unsqueeze(0))
            x = self.fuse_proj(torch.cat([x,responses],dim=-1))
            x = x.squeeze(1)

        logits = self.region_head(x)

        if box_masks is not None:
            return logits

        # generate box_masks for val/test
        region_probs = torch.sigmoid(logits)
        region_selected = region_probs > self.region_select_threshold  # [bs, 29]
        region_selected = region_selected.view(-1)  # [bs*29]
        num_box_categories = region_probs.size(1)

        box_labels_ = box_labels + boxes[:, 0] * num_box_categories
        # select boxes that detected by dino and predicted by the region selector
        box_masks = region_selected[box_labels_.to(torch.long)]
        return logits, region_probs, box_masks


class MultiThreadMemory(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, topk=32):
        super(MultiThreadMemory, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.topk = topk

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)

        query, key, value = \
            [l(x) for l, x in zip(self.linears, (query, key, value))]

        # query, key, value = \
        #     [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        #      for x in [query, key, value]]
        query = query.view(query.size(0), -1, self.h, self.d_k).transpose(1, 2)
        key = key.view(key.size(0), -1, self.h, self.d_k).transpose(1, 2)
        value = value.view(value.size(0), -1, self.h, self.d_k).transpose(1, 2)

        x, self.attn = self.memory_querying_responding(query, key, value, mask)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)

    def memory_querying_responding(self,query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores + mask

        selected_scores, idx = scores.topk(self.topk)

        # query [8, 8, 49/98, 64]) value [bs, num_head, num_mem, hs] score [8, 8, 49/98, 2048]
        # idx 8x8x98x32
        dummy_value = value.unsqueeze(2).expand(idx.size(0), idx.size(1), idx.size(2), value.size(-2),
                                                value.size(-1))  # [bs, num_head, num_token, num_mem, hs]
        dummy_idx = idx.unsqueeze(-1).expand(idx.size(0), idx.size(1), idx.size(2), idx.size(3),
                                             value.size(-1))  # [bs, num_head, num_token, topk, hs]

        selected_value = torch.gather(dummy_value, 3, dummy_idx)  # [8, 8, 98, 32, 64]

        p_attn = F.softmax(selected_scores, dim=-1)  # [8, 8, 98, 32]

        p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn.unsqueeze(3), selected_value).squeeze(3), p_attn

