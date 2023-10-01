import torch
import torch.nn as nn
import numpy as np

#from modules.visual_extractor import VisualExtractor
from .visual_extractor import VisualExtractor
from modules.my_encoder_decoder import EncoderDecoder as r2gen
#from modules.standard_trans import EncoderDecoder as st_trans
from modules.trans_both import EncoderDecoder as st_trans
from modules.cam_attn_con import  CamAttnCon
from modules.my_encoder_decoder import LayerNorm
from modules.old_forebacklearning import ForeBackLearning
from modules.utils import load_ape
from transformers import BertModel
from transformers import RobertaConfig, RobertaModel,BertConfig, BertModel
from modules.utils import init_weights
from .bert_model import BertCrossLayer, BertAttention,_prepare_decoder_attention_mask
from modules import heads


def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)

class RRGModel(nn.Module):
    def __init__(self, tokenizer, logger = None, config = None):
        super(RRGModel, self).__init__()
        self.vis = False
        self.tokenizer = tokenizer
        self.img_backbone = VisualExtractor(args, logger, config)

        if 'roberta' in config['text_backbone']:
            bert_config = RobertaConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )
        else:
            bert_config = BertConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )
        resolution_after = config['image_size']

        if 'roberta' in config['text_backbone']:
            self.text_backbone = RobertaModel.from_pretrained(config['text_backbone'])
        else:
            self.text_backbone = BertModel.from_pretrained(config['text_backbone'])

        self.cross_modal_text_transform = nn.Linear(config['input_text_embed_size'], config['hidden_size'])
        self.cross_modal_text_transform.apply(objectives.init_weights)
        self.cross_modal_image_transform = nn.Linear(config['input_image_embed_size'], config['hidden_size'])
        self.cross_modal_image_transform.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(init_weights)

        self.cross_modal_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config['num_top_layer'])])
        self.cross_modal_layers.apply(objectives.init_weights)
        self.cross_modal_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config['num_top_layer'])])
        self.cross_modal_pooler = heads.Pooler(config["hidden_size"])
        self.cross_modal_pooler.apply(objectives.init_weights)

        if config['vocab_size'] != self.text_transformer.embeddings.word_embeddings.weight.shape[0]:
            self.text_transformer.resize_token_embeddings(config['vocab_size'])

        self.head = heads.MLMHead(bert_config)
        self.head.apply(objectives.init_weights)

        self.BeamSearch = beam_search.BeamSearch(config)
        self.BeamSearch.set_tokenidx(self.tokenizer)


    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


    def forward(self, images, targets=None,labels=None, mode='train'):
        ret =  self.forward_feats(images)
        if mode == 'train':
            ret = forward_train(images,text_ids=targets)
            output, fore_rep_encoded, target_embed, align_attns = self.encoder_decoder(gbl_feats, patch_feats, targets, mode='forward')
            return output
        elif mode == 'sample':
            output, _, attns = self.encoder_decoder(gbl_feats, patch_feats, mode='sample')
            return output, attns

    def forward_train(self,images,text_ids):
        image_embeds, gbl_feats = self.forward_img_feats(images)
        text_embeds = self.forward_text_feats(text_ids,image_embeds)
        out = self.head(text_embeds)
        return out

    def forward_sample(self,images,text_ids):
        image_embeds, gbl_feats = self.forward_img_feats(images)
        text_embeds = self.forward_text_feats(text_ids,image_embeds)
        out = self.head(text_embeds)
        return out


    def forward_img_feats(self,images,text_ids):
        patch_feats, gbl_feats = self.img_backbone(images)
        patch_feats = self.cross_modal_image_transform(patch_feats)
        return patch_feats,gbl_feats

    def forward_text_feats(self,text_ids,image_embeds):
        text_embeds = self.text_transformer.embeddings(input_ids=text_ids)
        text_masks = text_embeds.new_full((text_ids.size(0), text_ids.size(1)), 1, dtype=torch.long)

        # # text_masks = subsequent_mask(ys.size(1))
        # device = text_embeds.device
        # attention_mask = subsequent_mask(ys.size(1)).to(device)
        # attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
        # extend_text_masks = attention_mask.unsqueeze(0)
        input_shape = text_ids.size()
        print(111,input_shape) # check not bidirection
        extend_text_masks = _prepare_decoder_attention_mask(text_masks, input_shape,
                                                            text_embeds, text_embeds.device)

        image_masks = torch.ones((img_feats.size(0), img_feats.size(1)), dtype=torch.long, device=device)
        extend_image_masks = self.text_transformer.get_extended_attention_mask(image_masks, image_masks.size(), device)

        # print(text_masks[0],extend_text_masks.shape)
        for layer in self.text_transformer.encoder.layer:
            text_embeds = layer(text_embeds, extend_text_masks)[0]

        text_embeds = self.cross_modal_text_transform(text_embeds)
        #text_embeds = text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks))

        for cross_layer in self.cross_modal_layers:
            text_embeds = cross_layer(text_embeds, image_embeds, extend_text_masks, extend_image_masks)[0]
        # cls_feats_text = self.cross_modal_text_pooler(x)

        return text_embeds


    def core(self,
                it,
                img_feats,
                state,
                     ):
        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        #out = self.model.decode(img_feats, mask, ys, subsequent_mask(ys.size(1)).to(img_feats.device))
        text_embeds = self.forward_text_feats(ys,img_feats)
        #cls_feats_text = self.cross_modal_text_pooler(x)
        out = self.head(text_embeds)
        #print(out[:,-1].argmax(dim=-1))
        return out[:, -1], [ys.unsqueeze(0)]




