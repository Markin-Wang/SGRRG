import torch.nn as nn
from modules.utils import init_weights
import torch


class RegionSelector(nn.Module):
    def __init__(self, config):
        super(RegionSelector, self).__init__()
        self.feature_size = config['d_vf']

        self.use_mem = config['use_mem_r']
        # memory settings
        if self.use_mem:
            self.topk = config['topk']
            self.cmn = MultiThreadMemory(config['num_heads'], self.feature_size, topk=self.topk)
            self.memory = nn.Parameter(torch.FloatTensor(config['num_mem'], self.feature_size))
            self.mem_proj = nn.Linear(self.feature_size, self.feature_size)
            self.fuse_proj = nn.Linear(self.feature_size * 2, self.feature_size)
            self.cmn.apply(init_weight)
            self.memory.apply(init_weights)
            self.mem_proj.apply(init_weights)
            self.fuse_proj.apply(init_weights)

        self.drop_prob = config['dropout']
        self.ff = nn.Sequential(
            nn.Linear(config['d_vf'], config['d_vf']),
            nn.GELU(),
            nn.Dropout(p=self.drop_prob),
        )
        self.ff.apply(init_weights)
        self.region_head = nn.Linear(config['d_vf'], config['num_classes'])
        self.region_head.apply(init_weights)
        self.region_select_threshold = config['region_select_threshold']

    def forward(self, x, boxes=None, box_labels=None, box_masks=None):
        x = self.ff(x)

        if self.use_mem:
            mem = self.mem_proj(self.memory)
            mem_masks = torch.full((x.size(0), 1, mem.size(0)), 0.0, dtype=x.dtype, device=x.device)
            responses = self.cmn(x, mem, mem, mem_masks)
            x = self.fuse_proj(torch.cat([x,responses],dim=-1))


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

    def forward(self, query, key, value, mask=None, layer_past=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        if layer_past is not None and layer_past.shape[2] == key.shape[1] > 1:
            query = self.linears[0](query)
            key, value = layer_past[0], layer_past[1]
            present = torch.stack([key, value])
        else:
            query, key, value = \
                [l(x) for l, x in zip(self.linears, (query, key, value))]

        if layer_past is not None and not (layer_past.shape[2] == key.shape[1] > 1):
            past_key, past_value = layer_past[0], layer_past[1]
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)
            present = torch.stack([key, value])

        query, key, value = \
            [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for x in [query, key, value]]

        x, self.attn = memory_querying_responding(query, key, value, mask=mask, dropout=self.dropout, topk=self.topk)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        if layer_past is not None:
            return self.linears[-1](x), present
        else:
            return self.linears[-1](x)
