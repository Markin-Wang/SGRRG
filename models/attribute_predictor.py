import torch.nn as nn
from torchvision.ops.roi_pool import RoIPool
from config import id2cat
from modules.utils import init_weights

class AttributePredictor(nn.Module):
    def __int__(self,config):
        super(AttributePredictor, self).__init__()
        self.box_resolution_before = config['box_res_before'] # the image resolution used to do object detection
        self.feature_size = config['feature_size'] # feature size output by the visual extractor
        self.output_size = config['output_size']
        self.roi_pool = RoIPool(output_size=[2,2], spatial_scale=self.box_resolution_before//self.feature_size)
        self.conv = nn.Conv2d(in_channels=config['d_vf'],out_channels=config['d_vf'],kernel_size=output_size[0],
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

    def forward(self,x):
        bs, num_tokens, feat_size = x.shape
        x = x.transpose(1, 2).reshape(bs,feat_size,int(num_tokens**0.5),int(num_tokens**0.5)) # transform to shape [BS,C,H,W]
        x = self.roi_pool(x)
        x = self.conv(x)
        x = x.reshape(bs,feat_size,num_tokens).transpose(1,2)
        return x

