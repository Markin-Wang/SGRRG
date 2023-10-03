import torch.nn as nn
from torchvision.ops.roi_pool import RoIPool
from config import id2cat
from modules.utils import init_weights

class RegionSelector(nn.Module):
    def __init__(self, config):
        super(RegionSelector, self).__init__()

        self.region_head = nn.Linear(config['d_vf'], config['num_classes'])
        self.region_head.apply(init_weights)

    def forward(self,x):
        return self.region_head(x)

