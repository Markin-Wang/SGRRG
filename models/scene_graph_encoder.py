import torch.nn as nn

class SceneGraphEncoder(nn.Module):
    def __int__(self,config):
        super(SceneGraphEncoder, self).__init__()
        self.box_resolution_before = config['box_res_before'] # the image resolution used to do object detection
        self.feature_size = config['feature_size'] # feature size output by the visual extractor
        self.num_attributes = config['num_attributes']
        self.attribute_embedding = nn.Embeding()


    def forward(self):
        pass
