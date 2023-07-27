from timm.models import VisionTransformer
import torch
import torch.nn as nn
from einops import rearrange

def main():
    model_args = dict(img_size=128, patch_size=16, embed_dim=192, depth=12, num_heads=3)
    model = VisionTransformer(
        **model_args
    )
    decoder = nn.ConvTranspose2d(192, 3, 16, 16)
    x = torch.rand(7, 3, 128, 128)
    y = model.forward_features(x)[:, 1:]
    y = rearrange(y, 'b (h w) d -> b d h w', h=8, w=8)
    z = decoder(y)
    print(y.shape)
    print(z.shape)    
if __name__ == '__main__':
    main()