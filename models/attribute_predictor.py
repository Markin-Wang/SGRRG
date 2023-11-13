import torch.nn as nn
from torchvision.ops.roi_pool import RoIPool
from config import cgnome_id2cat as id2cat
from modules.utils import init_weights
import torch


class AttributePredictor(nn.Module):
    def __init__(self, config):
        super(AttributePredictor, self).__init__()
        self.img_resolution = config['image_size']  # the image resolution used to do object detection
        self.spatial_scale = config['num_tokens'] / self.img_resolution  # feature size output by the visual extractor
        self.feature_size = config['d_vf']
        self.output_size = config['output_size']
        self.num_attributes = config['num_attributes']
        self.use_box_feats = config['use_box_feats']
        self.drop_prob = config['dropout']
        self.max_att = max(id2cat)
        self.num_classes = config['num_classes']
        self.att_pad_idx = config['att_pad_idx']
        self.use_amp = config['use_amp']
        self.roi_pool = RoIPool(output_size=[self.output_size, self.output_size],
                                spatial_scale= self.spatial_scale)
        self.disr_cls = config['disr_cls']
        self.disr_opt = config['disr_opt']
        if self.disr_opt == 'cls':
            self.disr_head = nn.Linear(config['d_vf'], 1)
            self.disr_head.apply(init_weights)

        # self.conv = nn.Conv2d(in_channels=self.feature_size, out_channels=self.feature_size,
        #                       kernel_size=self.output_size,
        #                       stride=1,
        #                       bias=False)

        # mention not use conv net as resolution in 224 two small, cause majority less than 2

        self.ff = nn.Sequential(
            nn.Linear(self.feature_size, self.feature_size),
            nn.LayerNorm(self.feature_size),
            nn.GELU(),
            nn.Dropout(p=self.drop_prob),
        )
        self.ff.apply(init_weights)

        attribute_heads = []
        self.name2head = {}
        for i, num_categories in enumerate(id2cat):
            # Note that the head sequence definition should be the same as the category id assignment in the annotation
            attribute_heads.append(nn.Linear(config['d_vf'], num_categories))
            # self.name2head[name] = i
        self.attribute_heads = nn.ModuleList(attribute_heads)
        # self.bn = nn.BatchNorm1d(self.feature_size)
        # self.relu = nn.LeakyReLU(inplace=True)
        # self.attribuite_head = nn.Linear(config['d_vf'], self.num_attributes)
        self.attribute_heads.apply(init_weights)

        # attribute_masks = torch.full((self.num_classes, self.max_att),-10000)
        # for i in range(attribute_masks.shape[0]):
        #     attribute_masks[i, :id2cat[i]] = 0
        #
        # self.register_buffer(
        #     "attribute_masks", attribute_masks, persistent=False
        # )

    # def forward(self, x, boxes, box_labels, box_masks=None):
    #     if box_masks is not None:
    #         boxes,box_labels = boxes[box_masks],box_labels[box_masks]
    #     bs, num_tokens, feat_size = x.shape
    #     x = x.transpose(1, 2).reshape(bs, feat_size, int(num_tokens ** 0.5),
    #                                   int(num_tokens ** 0.5))  # transform to shape [BS,C,H,W]
    #     x = self.roi_pool(x, boxes)  # box feasts, N x C x output_size x output_size
    #     x = torch.flatten(x,1)
    #     x = self.ff(x) # [bs, C, H(1), W(1)]
    #     logits = self.attribuite_head(x)
    #     return x, logits

    def forward(self, x, boxes, box_labels, box_masks=None):
        if box_masks is not None:
            boxes, box_labels = boxes[box_masks], box_labels[box_masks]
        bs, num_tokens, feat_size = x.shape
        x = x.transpose(1, 2).reshape(bs, feat_size, int(num_tokens ** 0.5),
                                      int(num_tokens ** 0.5))  # transform to shape [BS,C,H,W]
        x = self.roi_pool(x, boxes)  # box feasts, N x C x output_size x output_size
        x = torch.flatten(x, 1)
        x = self.ff(x)  # [bs, C, H(1), W(1)]
        if self.disr_cls:
            disr_logits = self.disr_head(x)
        else:
            disr_logits = None

        logits = torch.full((x.shape[0], self.max_att), self.att_pad_idx, device=x.device,
                            dtype=torch.float16 if self.use_amp else x.dtype)
        label_ids = torch.unique(box_labels)
        for label_id in label_ids.long():
            sample_ids = box_labels == label_id
            logits_i = self.attribute_heads[label_id](x[sample_ids])
            logits[sample_ids, :id2cat[label_id]] = logits_i
        # box order to form the logits
        return x, logits,disr_logits
