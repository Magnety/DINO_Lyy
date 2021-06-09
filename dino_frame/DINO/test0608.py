#import dino_frame.DINO.vision_transformer as vits
#print(vits.__dict__.keys())
import math
import warnings
import torch.nn as nn

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=(32,128,128), patch_size=(8,16,16), in_chans=1, embed_dim=768):
        super().__init__()
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])*(img_size[2] // patch_size[2])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        #self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
    def forward(self, x):
        #B, C, H, W = x.shape
        bs, c, d, h, w = x.shape
        print(self.num_patches)
        x = self.proj(x)
        print(x.shape)
        x = x.flatten(2)
        print(x.shape)

        x = x.transpose(-1, -2)
        print(x.shape)

        #x = x + self.position_embeddings
        return x

embed = PatchEmbed(img_size=(16,64,64))
x = torch.rand((2,1,16,64,64))
print(x.shape)
x = embed(x)
cls_token = nn.Parameter(torch.zeros(1, 1, 768))
print("cls_token.shape:",cls_token.shape)
cls_tokens = cls_token.expand(2, -1, -1)
print("cls_token.shape:",cls_token.shape)
x = torch.cat((cls_tokens, x), dim=1)
print("x.shape:",x.shape)