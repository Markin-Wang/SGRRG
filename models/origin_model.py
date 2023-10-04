import torch
import torch.nn as nn
import numpy as np

from .visual_extractor import VisualExtractor
# from modules.my_encoder_decoder import EncoderDecoder as r2gen
from modules.standard_trans import EncoderDecoder as st_trans
# from modules.trans_both import EncoderDecoder as st_trans
from modules.cam_attn_con import CamAttnCon
from modules.my_encoder_decoder import LayerNorm
from modules.old_forebacklearning import ForeBackLearning
from modules.utils import load_ape
from modules.standard_trans import subsequent_mask
from modules.utils import init_weights
from .attribute_predictor import AttributePredictor
from .region_selector import RegionSelector

class RRGModel(nn.Module):
    def __init__(self, tokenizer, logger=None, config=None):
        super(RRGModel, self).__init__()
        self.config = config
        self.addcls = config['addcls']
        self.vis = config['vis']
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(logger, config)
        self.region_select_threshold = config['region_select_threshold']

        self.sub_back = config['sub_back']
        self.records = []
        self.att_cls = config['att_cls']
        # if config['ed_name'] == 'r2gen':
        #     self.encoder_decoder = r2gen(config, tokenizer)
        # elif config['ed_name'] == 'st_trans':

        self.encoder_decoder = st_trans(config)

        if self.att_cls:
            self.region_selector = RegionSelector(config)

            #self.attribute_predictor = AttributePredictor(config)

        # self.scene_graph_encoder =
       # if self.att_cls:

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def get_img_feats(self, images, seq):
        patch_feats, gbl_feats = self.get_img_feats(images)
        return self.encode_img_feats(patch_feats,seq)


    def extract_img_feats(self,images):
        if self.addcls:
            patch_feats, gbl_feats, logits = self.visual_extractor(images)
            # if self.fbl and labels is not None:
        else:
            patch_feats, gbl_feats = self.visual_extractor(images)
        return patch_feats, gbl_feats


    def encode_img_feats(self,patch_feats,seq):
        patch_feats, seq, att_masks, seq_mask = self.encoder_decoder._prepare_feature_forward(patch_feats, None, seq)
        patch_feats = self.encoder_decoder.model.encode(patch_feats, att_masks)
        return patch_feats, seq, att_masks, seq_mask


    def forward(self, images, targets=None, boxes=None, box_labels=None, return_feats=False,mode='sample'):
        region_logits, region_probs = None, None

        patch_feats, gbl_feats = self.extract_img_feats(images)
        if self.att_cls:
            region_logits = self.region_selector(gbl_feats)
            if mode != 'train' or return_feats:
                region_probs = torch.sigmoid(region_logits) > 0.5
            # attribute_logits = self.attribute_predictor(patch_feats,boxes,box_labels)

        encoded_img_feats, seq, att_masks, seq_mask = self.encode_img_feats(patch_feats, targets)

        if mode == 'train':
            output, align_attns = self.encoder_decoder(encoded_img_feats, seq, att_masks, seq_mask)
            if return_feats: return output, region_logits, region_probs, encoded_img_feats
            return output, region_logits
        elif mode == 'sample':
            return encoded_img_feats, region_probs
        else:
            raise ValueError

    def core(self, it,
             patch_feats,
             mask,
             state, ):

        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        seq_mask = subsequent_mask(ys.size(1), type=patch_feats.dtype).to(
                                                           patch_feats.device)
        out, attns = self.encoder_decoder.model.decode(patch_feats, mask, ys, seq_mask)
        out = self.encoder_decoder.logit(out)
        # text_embeds = self.forward_text_feats(ys, img_feats)
        # cls_feats_text = self.cross_modal_text_pooler(x)
        # out = self.head(text_embeds)
        # print(out[:,-1].argmax(dim=-1))
        return out[:, -1], [ys.unsqueeze(0)]
