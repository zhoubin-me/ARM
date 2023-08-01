import torch
import torch.nn as nn


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

        self.inp_proc = Conv3DBlock(10, emb_dim // 2, 1, 1)
        self.trans_proc = Conv3DBlock(emb_dim, emb_dim // 2, 1, 1)

        self.post_proc = nn.Sequential(
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
            DenseBlock(emb_dim, 128, activation='lrelu'),
            DenseBlock(128, 128, activation='lrelu'),
            DenseBlock(128, self._out_dense, activation=None)
        )

    def forward(self, x, proprio):
        proprio = self.proprio_emb(proprio).unsqueeze(1)
        return self.mae.forward(x, proprio)

    def forward_encoder(self, x, proprio, x_q):
        proprio = self.proprio_emb(proprio).unsqueeze(1)
        y, _, _ = self.mae.forward_encoder(x, proprio, 0.0)
        rot_grip = y[:, :1]
        rot = self.rot_grip_proc(rot_grip) if self._out_dense > 0 else None

        trans = y[:, 2:]
        trans = rearrange(
            trans, 
            'b (p1 p2 p3) c -> b c p1 p2 p3', 
            p1=self._voxel_size, 
            p2=self._voxel_size, 
            p3=self._voxel_size)
        
        trans_ = self.trans_proc(trans)
        x_q = self.inp_proc(x_q)
        trans = torch.cat([trans_, x_q], dim=1)
        trans = self.post_proc(trans)
        return trans, rot


