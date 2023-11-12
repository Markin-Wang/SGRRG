import torch.nn as nn
from modules.utils import init_weights
import torch
from .vision_encoder import clones
import math
import torch.nn.functional as F

class DiseaseHead(nn.Module):
    def __init__(self, config):
        super(DiseaseHead, self).__init__()
        self.disease_head = nn.Linear(config['d_vf'], config['num_diseases'])
        self.disease_head.apply(init_weights)

    def forward(self, x, boxes=None, box_labels=None, box_masks=None):
        x = torch.mean(x, -2)
        return self.disease_head(x)


