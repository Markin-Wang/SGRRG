import torch.nn as nn
from torchvision.ops.roi_pool import RoIPool
from config import id2cat
from modules.utils import init_weights

class AttributePredictor(nn.Module):
    def __init__(self, config):
        super(AttributePredictor, self).__init__()
        self.img_resolution = config['image_size'] # the image resolution used to do object detection
        self.feature_size = config['feature_size'] # feature size output by the visual extractor
        self.output_size = config['output_size']
        self.roi_pool = RoIPool(output_size=[self.output_size,self.output_size], spatial_scale=self.img_resolution //self.feature_size)
        self.conv = nn.Conv2d(in_channels=config['d_vf'],out_channels=config['d_vf'],kernel_size=self.output_size,
                              stride=1)
        self.conv.apply(init_weights)
        attribute_heads = []
        self.name2head = {}
        for i, (name, num_categories) in enumerate(id2cat):
            # Note that the head sequence definition should be the same as the category id assignment in the annotation
            attribute_heads.append(nn.Linear(config['d_vf'], num_categories))
            self.name2head[name] = i
        for v in attribute_heads:
            v.apply(init_weights)
        self.attribute_heads = nn.ModuleList(attribute_heads)

    def forward(self,x, boxes, box_labels):
        bs, num_tokens, feat_size = x.shape
        x = x.transpose(1, 2).reshape(bs,feat_size,int(num_tokens**0.5),int(num_tokens**0.5)) # transform to shape [BS,C,H,W]
        box_feats = self.roi_pool(x,boxes)
        print(11111,x.shape)
        box_feats = self.conv(box_feats)
        x = x.reshape(bs,feat_size,num_tokens).transpose(1,2)
        return x

