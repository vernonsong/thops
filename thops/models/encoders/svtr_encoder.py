# @Time : 2022/8/14 15:04
# @Author : songweinan
# @Software: PyCharm
# 自能成羽翼，何必仰云梯。
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks import (ConvModule, DropPath, build_activation_layer,
                             build_norm_layer)
from mmocr.models.textrecog.encoders import BaseEncoder

from thops.registry import MODELS


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='Swish'),
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activate = build_activation_layer(act_cfg)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activate(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvMixer(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            HW=(8, 25),
            local_k=(3, 3),
    ):
        super().__init__()
        self.HW = HW
        self.dim = dim
        self.local_mixer = nn.Conv2d(
            dim,
            dim,
            local_k,
            1,
            (local_k[0] // 2, local_k[1] // 2),
            groups=num_heads,
        )

    def forward(self, x):
        h = self.HW[0]
        w = self.HW[1]
        x = x.transpose([0, 2, 1]).reshape([0, self.dim, h, w])
        x = self.local_mixer(x)
        x = x.flatten(2).transpose([0, 2, 1])
        return x


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 mixer='Global',
                 HW=(8, 25),
                 local_k=(7, 11),
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.HW = HW
        if HW is not None:
            H = HW[0]
            W = HW[1]
            self.N = H * W
            self.C = dim
        if mixer == 'Local' and HW is not None:
            hk = local_k[0]
            wk = local_k[1]
            mask = torch.ones([H * W, H + hk - 1, W + wk - 1])
            for h in range(0, H):
                for w in range(0, W):
                    mask[h * W + w, h:h + hk, w:w + wk] = 0.
            mask_paddle = mask[:, hk // 2:H + hk // 2,
                               wk // 2:W + wk // 2].flatten(1)
            mask_inf = torch.full([H * W, H * W], fill_value=float('-inf'))
            mask = torch.where(mask_paddle < 1, mask_paddle, mask_inf)
            self.mask = mask[None, None, :]
        self.mixer = mixer

    def forward(self, x):
        if self.HW is not None:
            N = self.N
            C = self.C
        else:
            _, N, C = x.shape
        qkv = self.qkv(x).reshape(
            (-1, N, 3, self.num_heads, C // self.num_heads)).permute(
                (2, 0, 3, 1, 4))
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = (q.matmul(k.permute((0, 1, 3, 2))))
        if self.mixer == 'Local':
            attn += self.mask
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).permute((0, 2, 1, 3)).reshape((-1, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mixer='Global',
                 local_mixer=(7, 11),
                 HW=(8, 25),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_cfg=dict(type='Swish'),
                 norm_cfg=dict(type='LN'),
                 prenorm=True):
        super().__init__()
        _, self.norm1 = build_norm_layer(norm_cfg, dim)
        if mixer == 'Global' or mixer == 'Local':
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                mixer=mixer,
                HW=HW,
                local_k=local_mixer,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop)
        elif mixer == 'Conv':
            self.mixer = ConvMixer(
                dim, num_heads=num_heads, HW=HW, local_k=local_mixer)
        else:
            raise TypeError('The mixer must be one of [Global, Local, Conv]')
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        _, self.norm2 = build_norm_layer(norm_cfg, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_cfg=act_cfg,
            drop=drop)
        self.prenorm = prenorm

    def forward(self, x):
        if self.prenorm:
            x = self.norm1(x + self.drop_path(self.mixer(x)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


@MODELS.register_module()
class SVTREncoder(BaseEncoder):
    """参考https://github.com/1079863482/paddle2torch_PPOCRv3."""

    def __init__(
            self,
            in_channels,
            dims=64,  # XS
            depth=2,
            hidden_dims=120,
            use_guide=False,
            num_heads=8,
            qkv_bias=True,
            mlp_ratio=2.0,
            drop_rate=0.1,
            attn_drop_rate=0.1,
            drop_path=0.,
            qk_scale=None,
            init_cfg=None,
            **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.use_guide = use_guide
        self.conv1 = ConvModule(
            in_channels,
            in_channels // 8,
            kernel_size=3,
            padding=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='Swish'))
        self.conv2 = ConvModule(
            in_channels // 8,
            hidden_dims,
            kernel_size=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='Swish'))

        self.svtr_block = nn.ModuleList([
            Block(
                dim=hidden_dims,
                num_heads=num_heads,
                mixer='Global',
                HW=None,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_cfg=dict(type='Swish'),
                norm_cfg=dict(type='LN'),
                attn_drop=attn_drop_rate,
                drop_path=drop_path,
                prenorm=False) for i in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dims, eps=1e-6)
        self.conv3 = ConvModule(
            hidden_dims,
            in_channels,
            kernel_size=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='Swish'))
        self.conv4 = ConvModule(
            2 * in_channels,
            in_channels // 8,
            kernel_size=3,
            padding=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='Swish'))

        self.conv1x1 = ConvModule(
            in_channels // 8,
            dims,
            kernel_size=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='Swish'))
        self.out_channels = dims

    def forward(self, feat, img_metas=None):
        # for use guide
        if self.use_guide:
            z = feat.clone()
            z.stop_gradient = True
        else:
            z = feat
        # for short cut
        h = z
        # reduce dim
        z = self.conv1(z)
        z = self.conv2(z)
        # SVTR global block
        B, C, H, W = z.shape
        z = z.flatten(2).permute([0, 2, 1])
        for blk in self.svtr_block:
            z = blk(z)
        z = self.norm(z)
        # last stage
        z = z.reshape([-1, H, W, C]).permute([0, 3, 1, 2])
        z = self.conv3(z)
        z = torch.cat((h, z), dim=1)
        z = self.conv1x1(self.conv4(z))
        return z
