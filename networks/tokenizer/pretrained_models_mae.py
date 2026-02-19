import torch
import timm
import random
from torch import nn

from modules.masking import mae_random_masking
from typing import Tuple, Union
from timm.models.vision_transformer import VisionTransformer

class VisionTransformerWithPretrainedWts(VisionTransformer):
    def __init__(self, patch_size, img_size, mask_ratio, **kwargs):
        """
        pretrained_cfg: pass the same kwargs you’d pass to timm.create_model

        """
        super().__init__(img_size=img_size,**kwargs)
        self.num_patches = (img_size // patch_size) ** 2
        self.mask_ratio = mask_ratio

        self.patch_embed.proj = nn.Conv2d(3, 768, kernel_size=patch_size, stride=patch_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))

    def forward_features(self, x, mask_ratio):
        x = self.patch_embed.proj(x)                    
        x = x.flatten(2).transpose(1, 2)   # [B, N, D]

        x = x + self.pos_embed.to(x.device)

        x, mask, ids_restore = mae_random_masking(x, mask_ratio)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x, mask, ids_restore

class Encoder(nn.Module):
    def __init__(
        self, 
        mask_ratio,
        resolution: Union[Tuple[int, int], int], 
        channels: int = 3, 
        pretrained_encoder = 'MAE',
        patch_size: int = 16,
        z_channels: int = 768,
        e_dim: int = 8,
        normalize_embedding: bool = True,
        # **ignore_kwargs
    ) -> None:
        # Initialize parent class with the first patch size
        super().__init__()
        self.image_size = resolution
        self.patch_size = patch_size
        self.channels = channels
        self.normalize_embedding = normalize_embedding
        self.z_channels = z_channels
        self.e_dim = e_dim
        self.mask_ratio = mask_ratio
        
        self.init_transformer(pretrained_encoder)

    def init_transformer(self, pretrained_encoder):
        if pretrained_encoder == 'VIT_DINO':
            pretrained_encoder_model = 'timm/vit_base_patch16_224.dino'
        elif pretrained_encoder == 'VIT_DINOv2':
            pretrained_encoder_model = 'timm/vit_base_patch14_dinov2.lvd142m'
        elif pretrained_encoder == 'MAE':
            pretrained_encoder_model = 'timm/vit_base_patch16_224.mae'
        elif pretrained_encoder == 'MAE_VIT_L':
            pretrained_encoder_model = 'timm/vit_large_patch16_224.mae'
        elif pretrained_encoder == 'VIT':
            pretrained_encoder_model = 'timm/vit_large_patch32_224.orig_in21k'
        elif pretrained_encoder == 'CLIP32':
            pretrained_encoder_model = 'timm/vit_base_patch32_clip_224.openai'
        elif pretrained_encoder == 'CLIP':
            pretrained_encoder_model = 'timm/vit_base_patch16_clip_224.openai'
        elif pretrained_encoder == 'base':
            pretrained_encoder_model = 'timm/vit_base_patch16_224'
        elif pretrained_encoder == 'large':
            pretrained_encoder_model = 'timm/vit_large_patch16_224'
       

        # self.encoder = timm.create_model(pretrained_encoder_model, img_size=self.image_size, patch_size=self.patch_size, pretrained=False, dynamic_img_size=True).train()
        pretrained_model = timm.create_model(pretrained_encoder_model, img_size=self.image_size, patch_size=self.patch_size, pretrained=True)
        state_dict = pretrained_model.state_dict()
        self.encoder = VisionTransformerWithPretrainedWts(patch_size=self.patch_size, img_size=self.image_size, 
                                                          mask_ratio=self.mask_ratio) 
        state_dict['pos_embed'] = state_dict['pos_embed'][:, 1:, :]

        missing, unexpected = self.encoder.load_state_dict(state_dict, strict=False)

        print(f"Loaded with {len(missing)} missing keys and {len(unexpected)} unexpected keys")
        print("Missing keys:")
        print(missing)
        print("Unexpected Keys: ")
        print(unexpected)
    
    def forward(self, img: torch.FloatTensor) -> torch.FloatTensor:
        h, mask, ids_restore = self.encoder.forward_features(img, mask_ratio=self.mask_ratio) # [B, N, D]
        
        B = mask.shape[0]
        H_patch = W_patch = self.image_size // self.patch_size
        mask_2d = mask.view(B, H_patch, W_patch)
        pixel_mask = mask_2d.repeat_interleave(self.patch_size, dim=1)\
                     .repeat_interleave(self.patch_size, dim=2)
        pixel_mask = pixel_mask.unsqueeze(1)  # [B, 1, 256, 256]
        
        return h, pixel_mask, ids_restore
