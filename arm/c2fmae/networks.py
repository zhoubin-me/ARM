import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from arm.c2fmae.mae import MaskedAutoencoderViT
from einops import rearrange
from arm.network_utils import DenseBlock, Conv3DBlock, Conv3DInceptionBlock 


class Qattention3DNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 out_dense: int,
                 voxel_size: int,
                 low_dim_size: int,
                 kernels: int,
                 norm: str = None,
                 activation: str = 'relu',
                 dense_feats: int = 32,
                 include_prev_layer = False,):
        super(Qattention3DNet, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._norm = norm
        self._activation = activation
        self._kernels = kernels
        self._low_dim_size = low_dim_size
        self._build_calls = 0
        self._voxel_size = voxel_size
        self._dense_feats = dense_feats
        self._out_dense = out_dense
        self._include_prev_layer = include_prev_layer

    def build(self):
        self._build_calls += 1
        if self._build_calls != 1:
            raise RuntimeError('Build needs to be called once.')
        
        emb_dim = 128
        self.mae = MaskedAutoencoderViT(
            img_size=64,
            in_chans=self._in_channels,
            patch_size=8,
            embed_dim=emb_dim,
            depth=4,
            num_heads=4,
            decoder_embed_dim=emb_dim,
            decoder_depth=4,
            decoder_num_heads=8,
            mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
        
        self.proprio_emb = nn.Linear(self._low_dim_size, emb_dim)

        self.y_post_proc = nn.Linear(emb_dim * 2, emb_dim)

        self.trans_post_proc = nn.Sequential(
            Conv3DInceptionBlock(
                emb_dim,
                emb_dim,
                activation="lrelu",
                residual=True,
            ),
            Conv3DBlock(
                emb_dim * 2,
                emb_dim,
                1,
                1,
            ),
            Conv3DInceptionBlock(
                emb_dim,
                emb_dim,
                activation="lrelu",
                residual=True,
            ),
            Conv3DBlock(
                emb_dim * 2,
                1,
                padding=1,
            ),
        )

        if self._out_dense > 0:
            self.rot_grip_proc = nn.Sequential(
                DenseBlock(emb_dim * 3, 128, activation='lrelu'),
                DenseBlock(128, 128, activation='lrelu'),
                DenseBlock(128, self._out_dense, activation=None)
            )

    def forward(self, x, proprio):
        proprio = self.proprio_emb(proprio).unsqueeze(1)
        return self.mae.forward(x, proprio)

    def forward_encoder(self, x, proprio):
        proprio = self.proprio_emb(proprio).unsqueeze(1)
        y, _, _, x_patch = self.mae.forward_encoder(x, proprio, 0.0)
        y_feat = torch.cat((y[:, 1:], x_patch), dim=-1)
        y_feat = self.y_post_proc(y_feat)

        trans = rearrange(
            y_feat,
            'b (p1 p2 p3) d -> b d p1 p2 p3', 
            p1=self._voxel_size, 
            p2=self._voxel_size, 
            p3=self._voxel_size)
        trans = self.trans_post_proc(trans)

        if self._out_dense > 0:
            hm = F.softmax(y_feat, dim=1)
            feat = torch.sum(hm * y_feat, dim=1)
            low_feat = y_feat.max(dim=1)[0]
            feat = torch.cat((feat, low_feat, y[:, 0]), dim=1)
            rot_grip = self.rot_grip_proc(feat)
        else:
            rot_grip = None

        return trans, rot_grip


