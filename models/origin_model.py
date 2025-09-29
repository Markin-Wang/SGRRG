import torch
import torch.nn as nn
import numpy as np

from .visual_extractor import VisualExtractor
# from modules.my_encoder_decoder import EncoderDecoder as r2gen
from modules.standard_trans import EncoderDecoder as st_trans
# from modules.trans_both import EncoderDecoder as st_trans
from modules.my_encoder_decoder import LayerNorm
from modules.utils import load_ape
from modules.standard_trans import subsequent_mask
from modules.utils import init_weights, init_weights_origin, compute_clip_loss
from .attribute_predictor import AttributePredictor
from .region_selector import RegionSelector
from .scene_graph_encoder import SceneGraphEncoder
from .vision_encoder import VisionEncoder
from .decoder import Decoder
from .word_embedding import BertEmbeddings, BasicEmbedding
from collections import defaultdict
from .head import DiseaseHead


class SGRRGModel(nn.Module):
    def __init__(self, tokenizer, logger=None, config=None):
        super(SGRRGModel, self).__init__()
        self.config = config
        self.vis = config['vis']
        self.tokenizer = tokenizer
        self.records = []
        self.att_cls = config['att_cls']
        self.region_cls = config['region_cls']
        self.dis_cls = config['dis_cls']
        self.use_box_feats = config['use_box_feats']
        self.use_obj_embeds = config['use_obj_embeds']
        self.use_sg = config['use_sg']
        self.sgave = config['sgave']
        self.sgade = config['sgade']
        self.hierarchical_attention = config['hierarchical_attention']
        self.hidden_size = config['d_model']
        self.d_ff = config['d_ff']

        self.clip = config['clip']
        self.clip_ve = config['clip_ve']

        self.att_feat_size = config['d_vf']
        self.use_ln = config['use_ln']

        self.drop_prob_lm = config['drop_prob_lm']
        self.use_dropout = config['use_dropout']

        self.visual_extractor = VisualExtractor(logger, config)
        # from pretrained, not init weight needed

        self.att_embed = nn.Sequential(
            nn.Linear(self.att_feat_size, self.hidden_size),
            *([nn.LayerNorm(self.hidden_size)] if self.use_ln else []),
            nn.GELU(),
            *([nn.Dropout(p=self.drop_prob_lm)] if self.use_dropout else []),
        )
        self.att_embed.apply(init_weights)

        self.vision_encoder = VisionEncoder(config)  # has list, may not recognize

        self.word_embedding = BasicEmbedding(config)

        self.decoder = Decoder(config)

        self.vision_encoder.apply(init_weights)
        self.word_embedding.apply(init_weights)
        self.decoder.apply(init_weights)

        # init_weights_origin((self.vision_encoder))
        # init_weights_origin(self.word_embedding)
        # init_weights_origin(self.decoder)

        self.rrg_head = nn.Linear(self.hidden_size, config['vocab_size'])
        self.rrg_head.apply(init_weights)

        self.disr_opt = config['disr_opt']

        if self.dis_cls:
            self.dis_head = DiseaseHead(config=config)

        if self.region_cls:
            self.region_selector = RegionSelector(config)  # init in that module

        if self.att_cls:
            assert self.region_cls, "To perform attribute classification, region classification should be enabled."
            self.attribute_predictor = AttributePredictor(config)  # init in that module

        if self.use_sg:
            assert self.att_cls and self.region_cls, 'region cls and attribute cls should be enabled.'
            self.scene_graph_encoder = SceneGraphEncoder(config)  # init in that module

        if self.clip:
            self.proj_image_clip = nn.Linear(self.hidden_size, self.hidden_size)
            self.proj_text_clip = nn.Linear(self.hidden_size, self.hidden_size)
            self.proj_image_clip.apply(init_weights)
            self.proj_text_clip.apply(init_weights)
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.loss_clip = nn.CrossEntropyLoss()



    # if self.att_cls:

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def get_img_feats(self, images, seq):
        patch_feats = self.get_img_feats(images)
        return self.encode_img_feats(patch_feats, seq)

    def get_text_feats(self, text_ids, img_feats, self_mask, cross_mask, sg_embeds=None, sg_masks=None, past_data=None):
        word_embed = self.word_embedding(text_ids)
        word_embed = self.decoder(word_embed, img_feats, self_mask, cross_mask, sg_embeds, sg_masks, past_data)
        return word_embed

    def extract_img_feats(self, images):
        patch_feats = self.visual_extractor(images)
        return patch_feats

    def encode_img_feats(self, patch_feats, att_masks, sg_embed=None, sg_mask=None):
        patch_feats = self.vision_encoder(patch_feats, att_masks, sg_embed, sg_mask)
        return patch_feats

    def forward(self, batch_dict, return_feats=False, split='train'):

        if split == 'train':
            return self.forward_train(batch_dict)

        return self.forward_test_gtsg(batch_dict, split)

    def forward_train(self, batch_dict):
        images, targets = batch_dict['image'], batch_dict['text']
        region_logits, region_probs, att_logits, att_probs, sg_embeds, sg_masks = None, None, None, None, None, None
        dis_logits, disr_logits, disr_ls_sg, orthogonal_ls = None, None, None, None
        box_abnormal_labels = None
        return_dicts = {}

        patch_feats = self.extract_img_feats(images)

        if self.dis_cls:
            dis_logits = self.dis_head(patch_feats)

        if self.region_cls:
            boxes, box_labels, box_masks = batch_dict['boxes'], batch_dict['box_labels'], batch_dict['box_masks']
            region_logits = self.region_selector(patch_feats, boxes, box_labels, box_masks)

            if self.disr_opt and self.disr_opt.startswith('con'):
                box_abnormal_labels = batch_dict['box_abnormal_labels'][box_masks]

        if self.att_cls:
            box_feats, att_logits, disr_logits = self.attribute_predictor(patch_feats, boxes, box_labels, box_masks,
                                                                          box_abnormal_labels)
            if self.use_box_feats:
                patch_feats = box_feats

        if self.use_sg:
            attribute_ids = batch_dict['attribute_ids']
            boxes, box_labels = boxes[box_masks], box_labels[box_masks]
            sg_embeds, sg_masks, obj_embeds, obj_masks_, disr_ls_sg, orthogonal_ls \
                = self.scene_graph_encoder(boxes, box_feats, box_labels, batch_size=patch_feats.size(0),
                                           att_ids=attribute_ids, box_abnormal_labels=box_abnormal_labels)

        # obtain seq mask, should generated in dataloader
        patch_feats, seq, att_masks, seq_masks = self.prepare_feature_forward(patch_feats, None, targets)

        if self.clip and self.clip_ve == 'extractor':
            avg_img_feats = torch.mean(patch_feats, dim=1)

        if self.sgave:
            if self.use_obj_embeds:
                patch_feats = self.encode_img_feats(patch_feats, att_masks, obj_embeds, obj_masks)
            else:
                patch_feats = self.encode_img_feats(patch_feats, att_masks, sg_embeds, sg_masks)
        else:
            patch_feats = self.encode_img_feats(patch_feats, att_masks)

        past_data = {}
        if self.sgade:
            if self.hierarchical_attention:
                sg_embeds_pool, sg_embeds_pool_masks = self._to_bs_format_pool(boxes[:, 0], sg_embeds, sg_masks,
                                                                               patch_feats.size(0))

                selected_bs = (sg_embeds_pool_masks.squeeze(1) == 0).sum(-1) != 0
                past_data.update({'sg_embeds_pool': sg_embeds_pool, 'sg_embeds_pool_masks': sg_embeds_pool_masks})
            else:
                selected_bs = (sg_masks[:,0]==0).sum(-1) != 0
            past_data.update({'selected_bs': selected_bs, 'bs_ids': boxes[:, 0]})

        text_embed, align_attns = self.get_text_feats(seq, patch_feats, self_mask=seq_masks, cross_mask=att_masks,
                                                      sg_embeds=sg_embeds, sg_masks=sg_masks, past_data=past_data)
        # output, align_attns = self.encoder_decoder(encoded_img_feats, seq, att_masks, seq_mask)
        output = self.rrg_head(text_embed)

        if self.clip:
            if self.clip_ve == 'encoder':
                avg_img_feats = torch.mean(patch_feats, dim=1)
            clip_mask = (seq.data > 0)
            clip_mask[:, 0] += True

            avg_text_feats = torch.stack([torch.mean(text_embed[i, clip_mask[i]], dim=0) for i in range(text_embed.shape[0])], dim=0)
            img_feats_clip, text_feats_clip = self.proj_image_clip(avg_img_feats), self.proj_text_clip(avg_text_feats)

            img_feats_clip = img_feats_clip / img_feats_clip.norm(dim=-1, keepdim=True)
            text_feats_clip = text_feats_clip / text_feats_clip.norm(dim=-1, keepdim=True)
            # clip_loss = compute_clip_loss(img_feats_clip, text_feats_clip)
            clip_loss = compute_clip_loss(img_feats_clip, text_feats_clip, self.logit_scale, self.loss_clip)
        else:
            clip_loss = None

        return_dicts.update({'rrg_preds': output,
                             'region_logits': region_logits,
                             'att_logits': att_logits,
                             'dis_logits': dis_logits,
                             'disr_logits': disr_ls_sg if self.disr_opt and 'sg' in self.disr_opt else disr_logits,
                             'orthogonal_ls': orthogonal_ls,
                             'clip_loss': clip_loss,
                             })

        return return_dicts

    def forward_test(self, batch_dict, split='val'):
        images, targets = batch_dict['image'], batch_dict['text']
        region_logits, region_probs, att_logits, att_probs = None, None, None, None
        dis_logits, dis_probs, disr_logits = None, None, None
        box_abnormal_labels = None
        return_dicts = {}
        att_probs_record = defaultdict(dict)

        patch_feats = self.extract_img_feats(images)

        if self.dis_cls:
            dis_logits = self.dis_head(patch_feats)
            dis_probs = torch.sigmoid(dis_logits)

        if self.region_cls:
            boxes, box_labels = batch_dict['boxes'], batch_dict['box_labels']
            # region_logits, region_probs, box_masks = self.region_selector(gbl_feats, boxes, box_labels, None)
            # box_masks = batch_dict['box_masks']
            region_logits, region_probs, box_masks = self.region_selector(patch_feats, boxes, box_labels, None)
            # region_probs = torch.sigmoid(region_logits)

            # boxes_temp = boxes[box_masks]
            # no_box_ids = []
            # all_ids = torch.unique(boxes_temp[:, 0])
            # for i in range(patch_feats.shape[0]):
            #     if i not in all_ids:
            #         no_box_ids.append(batch_dict['img_id'][i])

        if self.att_cls:
            # if self.disr_opt == 'con':
            #     box_abnormal_labels = batch_dict['box_abnormal_labels']
            box_feats, att_logits, disr_logits = self.attribute_predictor(patch_feats, boxes, box_labels,
                                                                          box_masks=box_masks)
            att_probs = torch.sigmoid(att_logits)
            boxes, box_labels = boxes[box_masks], box_labels[box_masks]
            # print(f'{len(boxes)/patch_feats.shape[0]:.2f} regions are selected to describe.')
            if split == 'test':
                for i in range(len(att_probs)):
                    bs_id, box_category = boxes[i, 0].long(), box_labels[i].long()
                    att_probs_record[bs_id.item()][box_category.item()] = att_probs[i].cpu()
            if self.use_box_feats:
                patch_feats = box_feats

        if self.use_sg:
            # boxes, box_labels = boxes[box_masks], box_labels[box_masks]
            # attribute_ids = batch_dict['attribute_ids']
            # sg_embeds, sg_masks = self.scene_graph_encoder(boxes, box_feats, box_labels, batch_size=patch_feats.size(0),
            #                                                att_ids=attribute_ids)
            sg_embeds, sg_masks, obj_embeds, obj_masks, disr_ls_sg, orthogonal_ls\
                = self.scene_graph_encoder(boxes, box_feats,box_labels,batch_size=patch_feats.size(0),att_probs=att_probs)

        patch_feats, seq, att_masks, seq_masks = self.prepare_feature_forward(patch_feats, None, targets)

        if self.sgave:
            assert self.use_sg
            if self.use_obj_embeds:
                patch_feats = self.encode_img_feats(patch_feats, att_masks, obj_embeds, obj_masks)
            else:
                patch_feats = self.encode_img_feats(patch_feats, att_masks, sg_embeds, sg_masks)
        else:
            patch_feats = self.encode_img_feats(patch_feats, att_masks)

        return_dicts.update({'encoded_img_feats': patch_feats,
                             'region_logits': region_logits,
                             'region_probs': region_probs,
                             'region_record': [boxes.cpu(),box_labels.cpu()] if self.region_cls else None,
                             'att_probs_record': att_probs_record,
                             'dis_logits': dis_logits,
                             'dis_probs': dis_probs,
                             'disr_logits': disr_logits,
                             # 'no_box_ids': no_box_ids,
                             'sg_embeds': sg_embeds if self.sgade else None,
                             'sg_masks': sg_masks if self.sgade else None,
                             'bs_ids': boxes[:, 0] if self.use_sg else None,
                             })

        return return_dicts

    def forward_test_gtsg(self, batch_dict, split='val'):
        images, targets = batch_dict['image'], batch_dict['text']
        region_logits, region_probs, att_logits, att_probs = None, None, None, None
        dis_logits, dis_probs, disr_logits = None, None, None
        box_abnormal_labels = None
        return_dicts = {}
        att_probs_record = defaultdict(dict)

        patch_feats = self.extract_img_feats(images)

        if self.dis_cls:
            dis_logits = self.dis_head(patch_feats)
            dis_probs = torch.sigmoid(dis_logits)



        if self.region_cls:
            boxes, box_labels, box_masks = batch_dict['boxes'], batch_dict['box_labels'], batch_dict['box_masks']
            region_logits = self.region_selector(patch_feats, boxes, box_labels, box_masks)


        if self.att_cls:
            box_feats, att_logits, disr_logits = self.attribute_predictor(patch_feats, boxes, box_labels, box_masks,
                                                                          box_abnormal_labels)
            if self.use_box_feats:
                patch_feats = box_feats

        if self.use_sg:
            attribute_ids = batch_dict['attribute_ids']
            boxes, box_labels = boxes[box_masks], box_labels[box_masks]
            sg_embeds, sg_masks, obj_embeds, obj_masks_, disr_ls_sg, orthogonal_ls \
                = self.scene_graph_encoder(boxes, box_feats, box_labels, batch_size=patch_feats.size(0),
                                           att_ids=attribute_ids, box_abnormal_labels=box_abnormal_labels)

        patch_feats, seq, att_masks, seq_masks = self.prepare_feature_forward(patch_feats, None, targets)

        if self.sgave:
            assert self.use_sg
            if self.use_obj_embeds:
                patch_feats = self.encode_img_feats(patch_feats, att_masks, obj_embeds, obj_masks)
            else:
                patch_feats = self.encode_img_feats(patch_feats, att_masks, sg_embeds, sg_masks)
        else:
            patch_feats = self.encode_img_feats(patch_feats, att_masks)

        return_dicts.update({'encoded_img_feats': patch_feats,
                             'region_logits': region_logits,
                             'region_probs': region_probs,
                             #'region_record': [boxes.cpu(),box_labels.cpu()] if self.region_cls else None,
                             'att_probs_record': att_probs_record,
                             'dis_logits': dis_logits,
                             'dis_probs': dis_probs,
                             'disr_logits': disr_logits,
                             # 'no_box_ids': no_box_ids,
                             'sg_embeds': sg_embeds if self.sgade else None,
                             'sg_masks': sg_masks if self.sgade else None,
                             'bs_ids': boxes[:, 0] if self.use_sg else None,
                             })

        return return_dicts

    def infer(self, text_ids, patch_feats, self_masks, cross_masks=None, sg_embeds=None, sg_masks=None, past_data=None):

        self_masks = self_masks[:, None, :].expand(patch_feats.size(0), text_ids.size(-1), text_ids.size(-1))
        self_masks = 1.0 - self_masks
        self_masks = self_masks.masked_fill(self_masks.bool(), torch.finfo(patch_feats.dtype).min)
        sub_mask = subsequent_mask(text_ids.size(-1), type=patch_feats.dtype).to(
            patch_feats.device)  # use add attention instead of filling
        self_masks = self_masks + sub_mask

        out, attns = self.get_text_feats(text_ids, patch_feats, self_masks, cross_masks, sg_embeds, sg_masks, past_data)

        return out

    def _to_bs_format_pool(self, bs_ids, node_embeds, node_masks, batch_size):

        bs_len = [(bs_ids == i).sum() for i in range(batch_size)]
        node_masks = node_masks == 0

        node_embeds = [torch.max(node_embeds[i, node_masks[i, 0]], dim=0, keepdim=True)[0]
                       if self.pooling == 'max' else
                       torch.mean(node_embeds[i, node_masks[i, 0]], dim=0, keepdim=True)
                       for i in range(node_embeds.shape[0])
                       ]
        node_embeds = torch.cat(node_embeds, dim=0)

        max_len = max(bs_len)
        reformed_node_embeds = torch.zeros(batch_size, max_len, node_embeds.shape[-1], device=node_embeds.device)
        reformed_node_masks = torch.full((batch_size, max_len), torch.finfo(node_embeds.dtype).min,
                                         device=node_embeds.device)
        for i in range(batch_size):
            reformed_node_embeds[i, :bs_len[i]] = node_embeds[bs_ids == i]
            reformed_node_masks[i, :bs_len[i]] = 0.0

        return reformed_node_embeds, reformed_node_masks.unsqueeze(1)

    def _prepare_feature(self, att_feats, att_masks):

        att_feats, seq, att_masks, seq_mask = self.prepare_feature_forward(att_feats, att_masks)
        memory = self.model.encode(att_feats, att_masks)

        # return fc_feats[..., :1], att_feats[..., :1], memory, att_masks
        return att_feats[..., :1], memory, att_masks, seq_mask

    def prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        att_feats = self.att_embed(att_feats)  # map the visual feature to encoding space

        if att_masks is None:
            att_masks = att_feats.new_zeros(att_feats.shape[:2], dtype=att_feats.dtype)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0).float()
            seq_mask[:, 0] += 1  # the first token is bos token with id 0
            seq_mask = seq_mask[:, None, :].expand(att_feats.size(0), seq.size(-1), seq.size(-1))
            seq_mask = 1.0 - seq_mask
            seq_mask = seq_mask.masked_fill(seq_mask.bool(), torch.finfo(att_feats.dtype).min)
            sub_mask = subsequent_mask(seq.size(-1), type=att_feats.dtype).to(
                att_feats.device)  # use add attention instead of filling
            seq_mask = seq_mask + sub_mask
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask
