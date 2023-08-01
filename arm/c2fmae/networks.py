import torch
import torch.nn as nn


from functools import partial

from arm.c2fmae.mae import MaskedAutoencoderViT
from einops import rearrange

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
        
        emb_dim = 256
        self.mae = MaskedAutoencoderViT(
            img_size=64,
            in_chans=self._in_channels,
            patch_size=8,
            embed_dim=emb_dim,
            depth=8,
            num_heads=8,
            decoder_embed_dim=emb_dim,
            decoder_depth=4,
            decoder_num_heads=8,
            mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))

        self.trans_final = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self._out_channels)
        )

        self.proprio_emb = nn.Linear(self._low_dim_size, emb_dim)

        if self._out_dense > 0:
            self.rot_grip_final = nn.Sequential(
                nn.Linear(emb_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, self._out_dense)
            )


    def forward(self, x, proprio):
        proprio = self.proprio_emb(proprio).unsqueeze(1)
        return self.mae.forward(x, proprio)

    def forward_encoder(self, x, proprio):
        proprio = self.proprio_emb(proprio).unsqueeze(1)
        x, _, _ = self.mae.forward_encoder(x, proprio, 0.0)
        trans = x[:, 2:]
        rot = x[:, :1]
        trans = self.trans_final(trans)
        rot = self.rot_grip_final(rot) if self._out_dense > 0 else None
        p = self._voxel_size
        trans = rearrange(trans, 'b (p1 p2 p3) 1 -> b p1 p2 p3', p1=p, p2=p, p3=p)
        return trans, rot
