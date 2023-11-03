import torch
import torch.nn as nn
import torchvision.models as models
#from swintrans.models import build_model
#from vit.models.modeling import VisionTransformer, CONFIGS
from modules.utils import load_pretrained
import numpy as np
import torchvision
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from .swin_helpers import swin_adapt_position_encoding
from . import swin_transformer as swin
from .region_selector import RegionSelector

weight_mapping = {
    # 'efficientnet_b5':models.EfficientNet_B5_Weights.IMAGENET1K_V1,
    # 'efficientnet_b6':models.EfficientNet_B6_Weights.IMAGENET1K_V1,
    # 'densenet121':models.DenseNet121_Weights.IMAGENET1K_V1,
    # 'densenet161':models.DenseNet161_Weights.IMAGENET1K_V1,
    # 'resnet101':models.ResNet101_Weights.IMAGENET1K_V2,
    # 'swin_s':models.Swin_S_Weights.DEFAULT,
    # 'swin_t':models.Swin_T_Weights.DEFAULT,
    # 'swin_b':models.Swin_B_Weights.DEFAULT
}

class VisualExtractor(nn.Module):
    def __init__(self, logger = None, config = None):
        super(VisualExtractor, self).__init__()
        self.model_name = config['img_backbone']
        self.dataset_name = config['dataset_name']

        if config['img_backbone'].startswith('swin'):
            self.model = getattr(swin, config['img_backbone'])(
                pretrained=True, config=config,
            )
            # self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.num_features = config['d_vf']

        elif config['img_backbone'].startswith('resnet'):
            model = getattr(models, args.ve_name)(weights=weight_mapping[args.ve_name])
            self.num_features = model.fc.in_features
            modules = list(model.children())[:-2]
            self.model = nn.Sequential(*modules)

        elif config['img_backbone'].startswith('vit'):
            config = CONFIGS[args.ve_name]
            self.model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=n_classes)
            self.model.load_from(np.load(args.pretrained))
            self.model.head = None
            self.num_features = config.hidden_size

        elif config['img_backbone'].startswith('densenet'):
            # if args.pretrained:
            #     model = getattr(models, args.ve_name)(pretrained=False)
            #     state_dict = torch.load(args.pretrained)['model']
            #     logger.info(state_dict.keys())
            #     model.load_state_dict(state_dict, strict=False)
            # else:
            model = getattr(models, args.ve_name)(weights=weight_mapping[args.ve_name])
            self.num_features = model.classifier.in_features
            self.model = model.features

            self.avg_fnt = torch.nn.AvgPool2d(kernel_size=1, stride=1, padding=0)

        elif config['img_backbone'].startswith('efficient'):
            model = getattr(models, args.ve_name)(weights=weight_mapping[args.ve_name])
            self.model = model.features
            self.num_features = model.classifier[1].in_features
            self.avg_fnt = torch.nn.AvgPool2d(1)

        else:
            raise NotImplementedError

        self.region_cls_only = config['region_cls_only']

        if self.region_cls_only:
            self.region_selector = RegionSelector(config)


        # trunc_normal_(self.head.weight, std=1 / math.sqrt(self.num_features * n_classes))
        # nn.init.constant_(self.head.bias, 0)


    def forward(self, images, labels=None, mode='train'):
        if self.dataset_name == 'iu_xray':
            if self.model_name.lower().startswith('vit'):
                feats_1, attn_weights_1 = self.model.forward_patch_features(images[:, 0])
                feats_2, attn_weights_2 = self.model.forward_patch_features(images[:, 1])
                feats = torch.cat((feats_1, feats_2), dim=1)
                patch_feats, avg_feats = feats[:, 1:, :], feats[:, 0, :]
            elif self.model_name.lower().startswith('swin'):
                patch_feats_1 = self.model(images[:, 0])
                patch_feats_2 = self.model(images[:, 1])
                patch_feats = torch.cat((patch_feats_1, patch_feats_2), dim=1)
                avg_feats = torch.mean(patch_feats, -2)
            elif self.model_name.lower().startswith('resnet'):
                patch_feats_1 = self.model(images[:, 0])
                patch_feats_2 = self.model(images[:, 1])
                avg_feats_1 = F.adaptive_avg_pool2d(patch_feats_1, (1, 1)).squeeze().reshape(-1, patch_feats_1.size(1))
                avg_feats_2 = F.adaptive_avg_pool2d(patch_feats_2, (1, 1)).squeeze().reshape(-1, patch_feats_2.size(1))
                batch_size, feat_size, _, _ = patch_feats_1.shape
                patch_feats_1 = patch_feats_1.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
                patch_feats_2 = patch_feats_2.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
                patch_feats = torch.cat((patch_feats_1, patch_feats_2), dim=1)
                # avg_feats = torch.cat((avg_feats_1, avg_feats_2), dim=1)
                avg_feats = torch.mean(torch.cat((avg_feats_1.unsqueeze(1), avg_feats_2.unsqueeze(1)), dim=1), dim=1)
            elif self.model_name.lower().startswith('densenet'):
                patch_feats_1 = F.relu(self.model(images[:, 0]), inplace=True)
                patch_feats_2 = F.relu(self.model(images[:, 1]), inplace=True)
                # print(1111, torch.cat((patch_feats_1, patch_feats_2),dim=3).shape)
                avg_feats_1 = F.adaptive_avg_pool2d(patch_feats_1, (1, 1)).squeeze().reshape(-1, patch_feats_1.size(1))
                avg_feats_2 = F.adaptive_avg_pool2d(patch_feats_2, (1, 1)).squeeze().reshape(-1, patch_feats_2.size(1))

                # avg_feats = (avg_feats_1 + avg_feats_2)/2
                #avg_feats = F.adaptive_avg_pool2d(torch.cat((patch_feats_1, patch_feats_2), dim=3),
                #                                  (1, 1)).squeeze().reshape(-1, patch_feats_1.size(1))
                batch_size, feat_size, _, _ = patch_feats_1.shape
                patch_feats_1 = patch_feats_1.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
                patch_feats_2 = patch_feats_2.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
                patch_feats = torch.cat((patch_feats_1, patch_feats_2), dim=1)
                #avg_feats = torch.cat((avg_feats_1, avg_feats_2), dim=1)

                avg_feats = torch.mean(torch.cat((avg_feats_1.unsqueeze(1), avg_feats_2.unsqueeze(1)), dim=1), dim=1)
            else:
                patch_feats, avg_feats = None, None

        else:
            if self.model_name.lower().lower().startswith('vit'):
                feats, attn_weights = self.model.forward_patch_features(images)
                patch_feats, avg_feats = feats[:, 1:, :], feats[:, 0, :]
            elif self.model_name.lower().startswith('swin'):
                patch_feats = self.model(images)
            elif self.model_name.lower().startswith('resnet'):
                patch_feats = self.model(images)
                avg_feats = F.adaptive_avg_pool2d(patch_feats, (1, 1)).squeeze().reshape(-1, patch_feats.size(1))
                batch_size, feat_size, _, _ = patch_feats.shape
                patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
            elif self.model_name.lower().startswith('densenet'):
                patch_feats = F.relu(self.model(images), inplace=True)
                avg_feats = F.adaptive_avg_pool2d(patch_feats, (1, 1)).squeeze().reshape(-1, patch_feats.size(1))
                batch_size, feat_size, _, _ = patch_feats.shape
                patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
            elif self.model_name.lower().startswith('efficientnet'):
                patch_feats = self.model(images)
                avg_feats = self.avg_fnt(patch_feats)
                batch_size, feat_size, _, _ = patch_feats.shape
                patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
            else:
                patch_feats, avg_feats = None, None

        if self.region_cls_only:
            region_logits = self.region_selector(avg_feats)
            if mode != 'train':
                return region_logits, torch.sigmoid(region_logits)
            return region_logits

        return patch_feats
