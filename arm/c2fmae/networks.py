import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from arm.c2fmae.mae import MaskedAutoencoderViT
from einops import rearrange
from arm.network_utils import DenseBlock, Conv3DBlock, Conv3DInceptionBlock, Conv2DUpsampleBlock, Conv2DBlock


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
        patch_size = 4 if self._dense_feats > 0 else 8
        img_size = 128 if self._dense_feats > 0 else 64
        self.mae = MaskedAutoencoderViT(
            img_size=img_size,
            in_chans=3,
            patch_size=patch_size,
            embed_dim=emb_dim,
            depth=4,
            num_heads=4,
            decoder_embed_dim=emb_dim,
            decoder_depth=4,
            decoder_num_heads=8,
            mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))

        self.proprio_emb = nn.Linear(self._low_dim_size, emb_dim)
        self.img_emb = nn.Conv2d(3, emb_dim, 1, 1)
        self.img_up = Conv2DUpsampleBlock(emb_dim, emb_dim, 1, 8)
        self.img_down = Conv2DBlock(emb_dim * 2, self._in_channels, 3, 1)

        emb_dim = 64

        self.translation_head = nn.Sequential(
            nn.Conv3d(10, emb_dim, 1, 1),
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
            self.rot_grip_head = nn.Sequential(
                nn.Conv3d(10, emb_dim, 1, 1),
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
                    self._out_dense,
                    16,
                    16,
                    padding=0
                ),
            )

    def forward(self, x, proprio):
        proprio = self.proprio_emb(proprio).unsqueeze(1)
        return self.mae.forward(x, proprio)

    def forward_encoder(self, imgs, proprio):
        proprio = self.proprio_emb(proprio).unsqueeze(1)
        feat, _, _ = self.mae.forward_encoder(imgs, proprio, 0.0)
        img_emb = self.img_emb(imgs)

        feat = rearrange(
            feat[:, 1:],
            'b (p1 p2) d -> b d p1 p2',
            p1=16,
            p2=16)

        feat = self.img_up(feat)
        feat = torch.cat((feat, img_emb), dim=1)
        feat = self.img_down(feat)
        return feat

    def forward_head(self, voxel_grid):
        translation = self.translation_head(voxel_grid)
        if self._out_dense > 0:
            rot_grip = self.rot_grip_head(voxel_grid)
            rot_grip = rearrange(rot_grip, 'b c d h w -> b (c d h w)')
        else:
            rot_grip = None
        return translation, rot_grip
