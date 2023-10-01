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

class RRGModel(nn.Module):
    def __init__(self, tokenizer, logger=None, config=None):
        super(RRGModel, self).__init__()
        self.config = config
        self.addcls = config['addcls']
        self.vis = config['vis']
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(logger, config)
        self.fbl = config['fbl']
        self.wmse = config['wmse']
        self.attn_cam = config['attn_cam']

        self.sub_back = config['sub_back']
        self.records = []
        self.att_cls = config['att_cls']
        # if config['ed_name'] == 'r2gen':
        #     self.encoder_decoder = r2gen(config, tokenizer)
        # elif config['ed_name'] == 'st_trans':

        self.encoder_decoder = st_trans(config)

        #self.attribute_predictor = AttributePredictor(config)

        # self.scene_graph_encoder =
       # if self.att_cls:

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def get_img_feats(self, images, seq):
        if self.addcls:
            patch_feats, gbl_feats, logits, cams = self.visual_extractor(images)
            # if self.fbl and labels is not None:
            if self.fbl:
                fore_rep, back_rep, fore_map = self.fore_back_learn(patch_feats, cams, logits)
                if self.sub_back:
                    patch_feats = patch_feats - back_rep
                patch_feats = torch.cat((fore_rep, patch_feats), dim=1)
                return patch_feats, gbl_feats, logits, cams

        else:
            patch_feats, gbl_feats = self.visual_extractor(images)
        patch_feats, seq, att_masks, seq_mask = self.encoder_decoder._prepare_feature_forward(patch_feats, None, seq)
        patch_feats = self.encoder_decoder.model.encode(patch_feats, att_masks)
        return patch_feats, seq, att_masks, seq_mask

    def forward(self, images, targets=None, labels=None, mode='train', return_feats=False):
        fore_map, total_attns, weights, attns, idxs, align_attns_train = None, None, None, None, None, None
        if self.addcls:
            patch_feats, gbl_feats, logits, cams = self.get_img_feats(images, targets)
        else:
            patch_feats, seq, att_masks, seq_mask = self.get_img_feats(images, targets)

        if mode == 'train':
            output, fore_rep_encoded, target_embed, align_attns = self.encoder_decoder(patch_feats, seq, att_masks,
                                                                                       seq_mask,
                                                                                       mode='forward')
            if return_feats: return output, patch_feats
            return output
                # print(weights)
        elif mode == 'sample':
            return patch_feats
        else:
            raise ValueError

        return output, attns

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
