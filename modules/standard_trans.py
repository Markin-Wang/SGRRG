from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.forebacklearning import ForeBackLearning
from torch.cuda.amp import GradScaler, autocast, custom_fwd, custom_bwd
from math import inf
from timm.models.layers import trunc_normal_, to_2tuple
from .core import CvTAttention
from .utils import load_embedding_layer

WORD_EMBED_SIZE = 200

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])



def subsequent_mask(size,type):
    mask = torch.full((size, size), torch.finfo(type).min)
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(type)
    # attn_shape = (1, size, size)
    # subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('float')
    # subsequent_mask = torch.from_numpy(subsequent_mask).to(type)
    # subsequent_mask = subsequent_mask.masked_fill(subsequent_mask.bool(),torch.finfo(subsequent_mask.dtype).min)
    return mask.unsqueeze(0)


class Transformer(nn.Module):
    def __init__(self, vis_encoder, text_encoder, decoder, src_embed, fbl=False):
        super(Transformer, self).__init__()
        self.vis_encoder = vis_encoder
        self.text_encoder = text_encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.fbl = fbl

    def forward(self, src, tgt, src_mask, tgt_mask, mode = 'train'):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask, mode)

    def encode(self, src, src_mask):
        return self.vis_encoder(self.src_embed(src), src_mask)

    def decode(self, hidden_states, src_mask, tgt, tgt_mask, mode = 'sample'):
        target_emb = self.text_encoder(tgt,tgt_mask)
        if self.fbl:
            hidden_states, src_mask = hidden_states[:,1:], src_mask[:,:,1:]
        if mode == 'train':
            out, align_attns = self.decoder(target_emb, hidden_states, src_mask, tgt_mask)
            return out, hidden_states[:,0], out, align_attns
        return self.decoder(target_emb, hidden_states, src_mask, tgt_mask)


class Encoder(nn.Module):
    def __init__(self, layer, N, ape=None):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.d_model)
        self.ape = ape

    def forward(self, x, mask):
        if self.ape is not None:
            x = x + self.ape
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class TextEncoder(nn.Module):
    def __init__(self, embed, layer, N=1, encode=False):
        super(TextEncoder, self).__init__()
        self.encode = encode
        self.embed = embed
        if self.encode:
            self.layers = clones(layer, N)
            self.norm = nn.LayerNorm(layer.d_model)

    def forward(self, x, mask):
        x = self.embed(x)
        if self.encode:
            for layer in self.layers:
                x = layer(x, mask)
            x = self.norm(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.d_model = d_model

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.d_model)

    def forward(self, x, hidden_states, src_mask, tgt_mask):
        attns = []
        for layer in self.layers:
            x, attn = layer(x, hidden_states, src_mask, tgt_mask)
            attns.append(attn)
        return self.norm(x), attns


class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(self.d_model, dropout), 3)

    def forward(self, x, hidden_states, src_mask, tgt_mask):
        m = hidden_states
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward), self.src_attn.attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, use_rpe=False, height=7, width=7):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.use_rpe = use_rpe
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn, self.dropout = None, None
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        self.height, self.width = height, width

        if self.use_rpe:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * height - 1) * (2 * width - 1), h))  # 2*Wh-1 * 2*Ww-1, nH
            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(height)
            coords_w = torch.arange(width)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += height - 1  # shift to start from 0
            relative_coords[:, :, 1] += width - 1
            relative_coords[:, :, 0] *= 2 * width - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        #x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        d_k = query.size(-1)
        # with autocast(enabled=False):
        #     scores = (torch.matmul(query.float(), key.float().transpose(-2, -1)) / math.sqrt(d_k))
        scores = (torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k))
        if self.use_rpe:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.height * self.width, self.height * self.width, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            scores = scores + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # with autocast(enabled=False):
            # scores = scores.float().masked_fill(mask == 0, -inf)
            scores = scores + mask

        #     p_attn = F.softmax(scores, dim=-1)
        # else:
        p_attn = F.softmax(scores, dim=-1)
        # if value.dtype == torch.float16:
        #     p_attn = p_attn.half()
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)

        self.attn = p_attn
        x = torch.matmul(p_attn, value)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class BWEmbeddings(nn.Module):
    def __init__(self, d_model, dropout, data_dir):
        super(BWEmbeddings, self).__init__()
        self.position_embed = PositionalEncoding(WORD_EMBED_SIZE, dropout)
        self.embed_layer = load_embedding_layer(data_dir)
        self.linear_layer = nn.Linear(WORD_EMBED_SIZE, d_model)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(in_place=True)
        trunc_normal_(self.linear_layer, std=.02)

    def forward(self, x):
        x = self.embed_layer(x)
        print(1111, x.shape)
        x = self.position_embed(x)
        return self.relu(self.dropout(self.linear_layer(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class EncoderDecoder(nn.Module):

    def make_model(self, tgt_vocab):
        c = copy.deepcopy
        # if self.cvt_attn:
        #     vis_attn_encode = CvTAttention(self.d_model,self.d_model,self.num_heads, kernel_size=self.kernel_size,
        #                                    stride_kv=self.stride_kv, stride_q=self.stride_q,padding_kv=self.padding_kv,
        #                                    padding_q=self.padding_q)
        # else:
        vis_attn_encode = MultiHeadedAttention(self.num_heads, self.d_model, use_rpe=self.pe == 'rpe')
        #text_attn_encode = MultiHeadedAttention(self.num_heads, self.d_model)
        attn_decode = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)

        position = PositionalEncoding(self.d_model, self.dropout)
        if self.pe == 'ape':
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.d_model ))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        else:
            self.absolute_pos_embed = None

        # if self.bwinit:
        #     embed_layer = BWEmbeddings(self.d_model,self.dropout,self.data_dir)
        # else:
        embed_layer = nn.Sequential(Embeddings(self.d_model, tgt_vocab),c(position))



        model = Transformer(
            Encoder(EncoderLayer(self.d_model, c(vis_attn_encode), c(ff), self.dropout), self.num_layers_en, ape=self.absolute_pos_embed),
            TextEncoder(
                #embed=nn.Sequential(Embeddings(self.d_model, tgt_vocab),c(position)),
                embed=embed_layer,
                layer=EncoderLayer(self.d_model, c(attn_decode), c(ff), self.dropout), N=self.num_layers_ten, encode=self.encode_text),
            Decoder(
                DecoderLayer(self.d_model, c(attn_decode), c(attn_decode), c(ff), self.dropout),
                self.num_layers_de),
            lambda x: x,
            fbl=self.fbl)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, config):
        super(EncoderDecoder, self).__init__()
        self.config = config
        self.num_layers_en = config['num_layers_en']
        self.num_layers_de = config['num_layers_de']
        self.d_model = config['d_model']
        self.d_ff = config['d_ff']
        self.num_heads = config['num_heads']
        self.dropout = config['dropout']
        self.fbl = config['fbl']
        self.encode_text = config['encode_text']
        self.num_layers_ten = config['num_layers_ten']
        self.data_dir = config['data_dir']

        self.num_patches = config['num_patches']
        if self.fbl:
            self.num_patches += 1
        self.pe = config['pe']

        self.model = self.make_model(config['vocab_size'])
        self.logit = nn.Linear(self.d_model, config['vocab_size'])
        self.att_feat_size = config['d_vf']
        self.att_hid_size = config['d_model']
        self.use_bn = config['use_bn']
        self.input_encoding_size = config['d_model']
        self.drop_prob_lm = config['drop_prob_lm']
        # self.cls_logit = nn.Linear(args.d_model, 14)
        self.att_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
                (nn.Linear(self.att_feat_size, self.input_encoding_size),
                 nn.ReLU(),
                 nn.Dropout(self.drop_prob_lm)) +
                ((nn.BatchNorm1d(self.input_encoding_size),) if self.use_bn == 2 else ())))

    def init_hidden(self, bsz):
        return []


    def _prepare_feature(self, att_feats, att_masks):

        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        memory = self.model.encode(att_feats, att_masks)

        # return fc_feats[..., :1], att_feats[..., :1], memory, att_masks
        return att_feats[..., :1], memory, att_masks, seq_mask

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        att_feats = self.att_embed(att_feats) # map the visual feature to encoding space

        if att_masks is None:
            att_masks = att_feats.new_zeros(att_feats.shape[:2], dtype=att_feats.dtype)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0).float()
            seq_mask[:, 0] += 1 # the first token is bos token with id 0
            seq_mask = seq_mask[:, None, :].expand(att_feats.size(0), seq.size(-1), seq.size(-1))
            seq_mask = 1.0 - seq_mask
            seq_mask = seq_mask.masked_fill(seq_mask.bool(), torch.finfo(att_feats.dtype).min)
            sub_mask = subsequent_mask(seq.size(-1),type = att_feats.dtype).to(att_feats.device) # use add attention instead of filling
            seq_mask = seq_mask + sub_mask
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask

    def forward(self, att_feats, seq, att_masks=None, seq_mask=None,mode='train'):
        # att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)
        out, fore_rep_encoded, target_embed, align_attns = self.model.decode(att_feats, att_masks, seq, seq_mask, mode='train')
        # out, fore_rep_encoded, target_embed, align_attns = self.model(att_feats, seq, att_masks, seq_mask)
        outputs = self.logit(out)
        # cls_logits = self.cls_logit(torch.mean(out,dim=1))
        return outputs, fore_rep_encoded, target_embed, align_attns

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):

        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        out, attns = self.model.decode(memory, mask, ys, subsequent_mask(ys.size(1),type=memory.dtype).to(memory.device))
        out = self.logit(out)
        return out[:, -1], [ys.unsqueeze(0)], attns
