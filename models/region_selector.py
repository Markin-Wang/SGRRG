import torch.nn as nn
from modules.utils import init_weights
import torch


class RegionSelector(nn.Module):
    def __init__(self, config):
        super(RegionSelector, self).__init__()

        self.region_head = nn.Linear(config['d_vf'], config['num_classes'])
        self.region_head.apply(init_weights)
        self.region_select_threshold = config['region_select_threshold']

    def forward(self,x,boxes=None,box_labels=None,box_masks=None):
        logits = self.region_head(x)

        if box_masks is not None:
            return logits

        # generate box_masks for val/test
        region_probs = torch.sigmoid(logits)
        region_selected = region_probs > self.region_select_threshold# [bs, 29]
        region_selected = region_selected.view(-1) # [bs*29]
        num_box_categories = region_probs.size(1)

        box_labels_ = box_labels + boxes[:,0] * num_box_categories
        # select boxes that detected by dino and predicted by the region selector
        box_masks = region_selected[box_labels_.to(torch.long)]
        return logits,region_probs,box_masks

