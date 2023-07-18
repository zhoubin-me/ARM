import timm.models.vision_transformer
import torch
import torch.nn as nn

from arm.network_utils import DenseBlock, Conv3DBlock, Conv3DInceptionBlock
from arm.c2fmae.perceiver import Perceiver


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer."""

    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        embed_dim,
        depth,
        num_heads,
        skip_connection=True,
        unet_features=True,
        spatial_features=True,
        **kwargs
    ):
        super(VisionTransformer, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            **kwargs
        )
        # remove the classifier
        if hasattr(self, "pre_logits"):
            del self.pre_logits
        del self.head

        self._depth = depth
        self._skip_connection = skip_connection
        self._unet_features = unet_features
        self._spatial_features = spatial_features
        self._ss = nn.Softmax(dim=1)
        self._output_patch_norm = nn.LayerNorm(embed_dim)

        output_cls_token_dim = embed_dim
        if self._unet_features:
            output_cls_token_dim = embed_dim * depth
            if self._spatial_features:
                output_cls_token_dim *= 2

        self._cls_token_norm = nn.LayerNorm(output_cls_token_dim)

    def extract_feat(self, x):
        B = x.shape[0]
        patch_emb = self.patch_embed(x)
        x = patch_emb

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        output_cls_tokens = []
        spatial_softmax = []
        for blk in self.blocks:
            x = blk(x)
            output_cls_tokens.append(x[:, 0])

            # The spatial features compute the spatial mean of the learned class
            # tokens \mathbb{E}_{p(x)}[cls(x)].
            # Essentially, we treat VIT tokens x as distribution logits, whose
            # probability p(x) can be computed by taking a softmax in the feature
            # space. Next, the spatial mean of class tokens can be directly obtained
            # by taking the weighted average over class tokens \sum p(x)cls(x). This
            # feature followes the original C2F-ARM which gives statistics of the
            # spatial distribution at each feature level and greatly improves the
            # stability and sample efficiency of the framework.
            if self._spatial_features:
                features = x[:, 1:]
                probs = self._ss(features)
                softmax_features = (probs * self.pos_embed[:, 1:]).sum(dim=1)
                spatial_softmax.append(softmax_features)

        patches = x[:, 1:]

        if self._skip_connection:
            patches += patch_emb
            patches = self._output_patch_norm(patches)

        if self._unet_features:
            cls_token = torch.cat(output_cls_tokens, dim=-1)

            if self._skip_connection:
                cls_token += torch.max(patch_emb, dim=1)[0].repeat([1, self._depth])

            if self._spatial_features:
                spatial_feats = torch.cat(spatial_softmax, dim=-1)
                cls_token = torch.cat([cls_token, spatial_feats], dim=-1)

        else:
            cls_token = output_cls_tokens[-1]

        cls_token = self._cls_token_norm(cls_token)

        return cls_token, patches

    def forward(self, x):
        cls_token, patches = self.extract_feat(x)
        return cls_token, patches


class QattentionVIT(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        embed_dim,
        depth,
        num_heads,
        kernels,
        dense_feats,
        out_dense,
        post_proc_method: str = "conv",
        low_dim_size: int = 0,
        activation: str = "relu",
        include_prev_layer: bool = False,
        unet_features: bool = True,
        spatial_features: bool = True,
        pos_embed_3d_type: str = "learned",
    ) -> None:
        super().__init__()
        self._img_size = img_size
        self._patch_size = patch_size
        self._in_chans = in_chans
        self._embed_dim = embed_dim
        self._depth = depth
        self._num_heads = num_heads
        self._build_calls = 0
        self._include_prev_layer = include_prev_layer
        self._kernels = kernels
        self._low_dim_size = low_dim_size
        self._activation = activation
        self._out_dense = out_dense
        self._dense_feats = dense_feats
        self._post_proc_method = post_proc_method

        if self._post_proc_method == "none":
            self._vox_post_proc = False
        elif self._post_proc_method in ["conv", "inception", "perceiver"]:
            self._vox_post_proc = True

        self._unet_features = unet_features
        self._spatial_features = spatial_features
        self._pos_embed_3d_type = pos_embed_3d_type
        self._vox_grid = None

    def set_vox_grid(self, vox_grid):
        self._vox_grid = vox_grid

    def build(self):
        self._build_calls += 1
        if self._build_calls != 1:
            raise RuntimeError("Build needs to be called once.")

        num_chans = self._in_chans
        if self._include_prev_layer:
            num_chans *= 2

        if self._low_dim_size > 0:
            self._proprio_preprocess = DenseBlock(
                self._low_dim_size, self._kernels // 2, None, self._activation
            )
            num_chans += self._kernels

        assert self._kernels % 2 == 0
        self._input_preproc_pcd = DenseBlock(
            3, self._kernels // 2, "layer", self._activation
        )
        self._input_preproc_rgb = DenseBlock(
            3, self._kernels // 2, "layer", self._activation
        )
        self._input_preproc = DenseBlock(
            self._kernels, self._kernels // 2, None, self._activation
        )

        self._vit = VisionTransformer(
            self._img_size,
            self._patch_size,
            self._kernels,
            self._embed_dim,
            self._depth,
            self._num_heads,
            unet_features=self._unet_features,
            spatial_features=self._spatial_features,
        )

        self._out_layer = nn.Sequential(
            nn.Linear(self._embed_dim, self._kernels),
            nn.ELU(),
            nn.Linear(self._kernels, 1),
        )

        voxel_size = self._vox_grid._voxel_size
        voxel_feature_size = self._vox_grid._voxel_feature_size
        self._voxel_size = voxel_size
        self._voxel_feature_size = voxel_feature_size
        if self._vox_post_proc:
            if self._post_proc_method == "inception":
                self._3d_convs = nn.Sequential(
                    Conv3DInceptionBlock(
                        voxel_feature_size,
                        voxel_feature_size,
                        activation="lrelu",
                        residual=True,
                    ),
                    Conv3DBlock(
                        voxel_feature_size * 2,
                        voxel_feature_size,
                        1,
                        1,
                    ),
                    Conv3DInceptionBlock(
                        voxel_feature_size,
                        voxel_feature_size,
                        activation="lrelu",
                        residual=True,
                    ),
                    Conv3DBlock(
                        voxel_feature_size * 2,
                        1,
                        padding=1,
                    ),
                )
            elif self._post_proc_method == "conv":
                self._3d_convs = nn.Sequential(
                    Conv3DBlock(
                        voxel_feature_size,
                        voxel_feature_size // 2,
                        padding=1,
                        activation="lrelu",
                    ),
                    Conv3DBlock(
                        voxel_feature_size // 2,
                        1,
                        padding=1,
                    ),
                )

            elif self._post_proc_method == "perceiver":
                self._perceiver = Perceiver(
                    input_channels=voxel_feature_size,
                    input_axis=3,
                    num_freq_bands=6,
                    depth=2,
                    max_freq=10,
                    num_latents=64,
                    latent_dim=64,
                )

                if self._pos_embed_3d_type == "learned":
                    self._pos_emb_3d = nn.Parameter(
                        torch.randn(
                            (1, voxel_feature_size, voxel_size, voxel_size, voxel_size)
                        )
                    )

        if self._out_dense > 0:
            cls_token_dim = (
                self._embed_dim * 4 if self._unet_features else self._embed_dim
            )
            if self._unet_features and self._spatial_features:
                cls_token_dim *= 2

            if self._post_proc_method in ["conv", "inception"]:
                fake_3d_tensor = torch.zeros(
                    [1, voxel_feature_size, voxel_size, voxel_size, voxel_size]
                )
                conv3d_output_dim = 0
                for conv in self._3d_convs:
                    fake_3d_tensor = conv(fake_3d_tensor)
                    conv3d_output_dim += fake_3d_tensor.size(1)

                cls_token_dim += conv3d_output_dim * 2

            self._dense_out = nn.Sequential(
                DenseBlock(cls_token_dim, self._dense_feats, None, self._activation),
                DenseBlock(
                    self._dense_feats, self._dense_feats, None, self._activation
                ),
                DenseBlock(self._dense_feats, self._out_dense, None, None),
            )

        pool_size = self._img_size // self._vox_grid._voxel_size
        self._avg_pool = nn.AvgPool2d(
            pool_size,
            stride=pool_size,
        )

    def forward(self, rgb, pcd, proprio, pcd_2d, prev_layer_x, bounds=None):
        # x = ins.view(-1, self._img_size, self._img_size, 6)
        # rgb = x[..., :3]
        # pcd = x[..., 3:]
        # b = ins.shape[0]
        b = rgb.shape[0]
        if self._include_prev_layer:
            raise NotImplementedError

        rgb = self._input_preproc_rgb(rgb)
        pcd = self._input_preproc_pcd(pcd)
        x = torch.cat([rgb, pcd], dim=-1)
        x = self._input_preproc(x)

        if self._low_dim_size > 0:
            p = self._proprio_preprocess(proprio)
            p = p.unsqueeze(1).unsqueeze(1).repeat(1, self._img_size, self._img_size, 1)
            x = torch.cat([x, p], dim=-1)
            x = torch.permute(x, (0, 3, 1, 2))

        cls_token, patches = self._vit(x)
        num_patches = self._img_size // self._patch_size
        patches = patches.view(b, num_patches, num_patches, -1)

        trans = patches
        if not self._vox_post_proc:
            trans = self._out_layer(patches)

        coords = self._avg_pool(pcd_2d.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).reshape(b, -1, 3)

        voxel_grid = self._vox_grid.coords_to_bounding_voxel_grid(
            coords, coord_features=trans, coord_bounds=bounds
        ).permute(0, 4, 1, 2, 3)

        if self._post_proc_method in ["conv", "inception"]:
            vox_means = []
            vox_maxs = []
            for conv in self._3d_convs:
                voxel_grid = conv(voxel_grid)
                voxel_mean = torch.mean(voxel_grid, dim=(-1, -2, -3))
                voxel_max = torch.max(
                    voxel_grid.view(*voxel_grid.shape[:2], -1), dim=-1
                )[0]
                vox_means.append(voxel_mean)
                vox_maxs.append(voxel_max)

            vox_means = torch.cat(vox_means, dim=-1)
            vox_maxs = torch.cat(vox_maxs, dim=-1)
        elif self._post_proc_method == "perceiver":
            if self._pos_embed_3d_type == "learned":
                voxel_grid += self._pos_emb_3d

            voxel_grid = torch.permute(voxel_grid, (0, 2, 3, 4, 1))
            voxel_grid = self._perceiver.forward(voxel_grid)
            voxel_grid = torch.permute(voxel_grid, (0, 4, 1, 2, 3)).contiguous()

        self.latent_dict = {
            "trans_out": patches,
        }

        rot_and_grip_out = None
        if self._out_dense > 0:
            rot_feature = cls_token
            if self._post_proc_method in ["conv", "inception"]:
                rot_feature = torch.cat([cls_token, vox_maxs, vox_means], dim=-1)
            rot_and_grip_out = self._dense_out(rot_feature)
            self.latent_dict.update(
                {
                    "dense2": rot_and_grip_out,
                }
            )

        return voxel_grid, rot_and_grip_out