import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from collections import OrderedDict
from torch.cuda.amp import autocast
from math import inf

class CvTAttention(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv=(1,),
                 padding_q=1,
                 # with_cls_token=True,
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads
        # head_dim = self.qkv_dim // num_heads
        self.scale = dim_out ** -0.5
        # self.with_cls_token = with_cls_token
        self.conv_proj_q = self._build_projection(
            dim_in, dim_out, kernel_size, padding_q,
            stride_q
        )
        self.conv_proj_k = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv
        )
        self.conv_proj_v = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv
        )

        # self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        # self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        # self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def _build_projection(self,
                          dim_in,
                          dim_out,
                          kernel_size,
                          padding,
                          stride):
        proj = nn.Sequential(OrderedDict([
            ('depth_conv', nn.Conv2d(
                dim_in,
                dim_in,
                kernel_size=kernel_size,
                padding='same',
                stride=stride,
                bias=False,
                groups=dim_in
            )),
            ('bn', nn.BatchNorm2d(dim_in)),
            ('point_conv', nn.Conv2d(
                dim_in,
                dim_out,
                kernel_size=1,
                padding=0,
                stride=1,
                bias=False,
                groups=1
            )),
            ('rearrange', Rearrange('b c h w -> b (h w) c')),
        ]))
        return proj

    def forward_conv(self, x, h, w):
        # if self.with_cls_token:
        #     cls_token, x = torch.split(x, [1, h*w], 1)


        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x)
        else:
            q = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x)
        else:
            k = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x)
        else:
            v = rearrange(x, 'b c h w -> b (h w) c')

        # if self.with_cls_token:
        #     q = torch.cat((cls_token, q), dim=1)
        #     k = torch.cat((cls_token, k), dim=1)
        #     v = torch.cat((cls_token, v), dim=1)

        return q, k, v

    def forward(self, x, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        num_token = x.shape[1]
        h, w = int(num_token**0.5), int(num_token**0.5)
        assert h*w == num_token
        q, k, v = self.forward_conv(x,h=h, w=w)

        q = rearrange(q, 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.num_heads)

        with autocast(enabled=False):
            attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
        if mask is not None:
            # with autocast(enabled=False):
            # scores = scores.float().masked_fill(mask == 0, -inf)
            attn_score = attn_score.masked_fill(mask == 0, float(-inf))

        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)

        x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        x = rearrange(x, 'b h t d -> b t (h d)')

        x = self.proj(x)
        x = self.proj_drop(x)

        return x