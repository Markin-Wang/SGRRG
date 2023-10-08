import torch.nn as nn
from torchvision.ops.roi_pool import RoIPool
from config import id2cat
from modules.utils import init_weights
import torch

class AttributePredictor(nn.Module):
    def __init__(self, config):
        super(AttributePredictor, self).__init__()
        self.img_resolution = config['image_size']  # the image resolution used to do object detection
        self.feature_size = config['d_vf']  # feature size output by the visual extractor
        self.output_size = config['output_size']
        self.num_attributes = config['num_attributes']
        self.use_box_feats = config['use_box_feats']



        self.roi_pool = RoIPool(output_size=[self.output_size, self.output_size],
                                spatial_scale=self.img_resolution // self.feature_size)
        self.conv = nn.Conv2d(in_channels=self.feature_size, out_channels=self.feature_size,
                              kernel_size=self.output_size,
                              stride=1,
                              bias=False)
        self.conv.apply(init_weights)
        self.bn = nn.BatchNorm1d(config['d_vf'])
        self.relu = nn.LeakyReLU()

        # attribute_heads = []
        # self.name2head = {}
        # for i, (name, num_categories) in enumerate(id2cat):
        #     # Note that the head sequence definition should be the same as the category id assignment in the annotation
        #     attribute_heads.append(nn.Linear(config['d_vf'], num_categories))
        #     self.name2head[name] = i
        # for v in attribute_heads:
        #     v.apply(init_weights)
        # self.attribute_heads = nn.ModuleList(attribute_heads)
        self.norm = nn.LayerNorm
        self.attribuite_head = nn.Linear(config['d_vf'], self.num_attributes)

    def forward(self, x, boxes, box_labels, box_masks=None):
        if box_masks is not None:
            boxes,box_labels = boxes[box_masks],box_labels[box_masks]
        bs, num_tokens, feat_size = x.shape
        x = x.transpose(1, 2).reshape(bs, feat_size, int(num_tokens ** 0.5),
                                      int(num_tokens ** 0.5))  # transform to shape [BS,C,H,W]
        x = self.roi_pool(x, boxes)  # box feasts, N x C x output_size x output_size
        x = self.conv(x) # [bs, C, H(1), W(1)]
        x = torch.flatten(x,1)
        x= self.relu(self.bn(x))
        logits = self.attribuite_head(x)
        if self.use_box_feats:
            return x, logits
        return logits
