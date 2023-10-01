from torch import nn
import torch
from typing import Any, List, Optional, Tuple, Union
from torch import Tensor, nn
from copy import deepcopy as clone
from torch.cuda.amp import autocast

class ForeBackLearning(nn.Module):
    def __init__(self, norm=None,dropout=None):
        super(ForeBackLearning, self).__init__()
        self.dropout = dropout
        self.norm = norm is not None
        if norm:
            self.fore_norm = norm
            # self.back_norm = clone(norm)
        if dropout:
            self.fore_dropout = dropout
            # self.back_dropout = clone(dropout)

    def forward(self,patch_feats,cam,logits):
        logits = torch.sigmoid(logits)
        labels = (logits >= 0.5).to(cam.dtype)
        cam = labels.unsqueeze(-1) * cam
        fore_map, _ = torch.max(cam, dim=1)
        #print(0, torch.isnan(fore_map).any())
        fore_map = self._normalize(fore_map)
        #print(1, torch.isnan(fore_map).any())
        fore_map = fore_map.unsqueeze(1)
        #back_map = 1-fore_map
        fore_rep = torch.matmul(fore_map, patch_feats)
        #back_rep = torch.matmul(back_map, patch_feats)
        if self.norm:
            fore_rep = self.fore_norm(fore_rep)
            # back_rep = self.back_norm(back_rep)
        if self.dropout:
            fore_rep = self.fore_dropout(fore_rep)
            # back_rep = self.back_dropout(back_rep)
        return fore_rep, None, fore_map.squeeze(1)

    def _normalize(self, cams: Tensor, spatial_dims: Optional[int] = None) -> Tensor:
        """CAM normalization."""
        with autocast(enabled=False):
            cams = cams.float()
            cams_max, cams_min = cams.max(-1,keepdim=True).values, cams.min(-1,keepdim=True).values
            cams = (cams-cams_min) / ((cams_max-cams_min) + 1e-12)
        return cams