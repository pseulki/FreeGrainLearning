"""Define CAST model for classification following DeiT convention.

Modified from:
    https://github.com/facebookresearch/moco-v3/blob/main/vits.py
    https://github.com/facebookresearch/deit/blob/main/models.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial, reduce
from operator import mul

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import PatchEmbed
from timm.models.layers import trunc_normal_

from cast_models.utils import segment_mean_nd
from cast_models.graph_pool import GraphPooling
from cast_models.modules import Pooling, ConvStem

__all__ = [
    'cast_small',
    'cast_small_deep',
    'cast_base',
    'cast_base_deep',
]


class CAST(VisionTransformer):
    def __init__(self, nb_classes, *args, **kwargs):
        depths = kwargs['depth']
        # These entries do not exist in timm.VisionTransformer.
        num_clusters = kwargs.pop('num_clusters', [64, 32, 16, 8])
        kwargs['depth'] = sum(kwargs['depth'])
        super().__init__(**kwargs)

        # Do not tackle dist_token.
        assert self.dist_token is None, 'dist_token is not None.'
        assert self.head_dist is None, 'head_dist is not None.'
 
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        trunc_normal_(self.pos_embed, std=.02)

        if len(nb_classes) == 3:
            self.num_classes = nb_classes[0]
            self.num_family = nb_classes[1]
            self.num_manufacturer = nb_classes[2]
        elif len(nb_classes) == 2:
            self.num_classes = nb_classes[0]
            self.num_family = nb_classes[1]
            self.num_manufacturer = 0


        #####################

        self.family_head = nn.Linear(self.embed_dim, self.num_family) if self.num_family > 0 else nn.Identity()
        if len(nb_classes) == 3:
            self.manufacturer_head = nn.Linear(self.embed_dim, self.num_manufacturer) if self.num_manufacturer > 0 else nn.Identity()


        # self.family_head.apply(self._init_weights)
        # self.manufacturer_head.apply(self._init_weights)
        # --------------------------------------------------------------------------
        # Encoder specifics
        cumsum_depth = [0]
        for d in depths:
            cumsum_depth.append(d + cumsum_depth[-1])

        blocks = []
        pools = []
        for ind, depth in enumerate(depths):

            # Build Attention Blocks.
            blocks.append(self.blocks[cumsum_depth[ind]:cumsum_depth[ind+1]])

            # Build Pooling layers
            pool = Pooling(
                pool_block=GraphPooling(
                    num_clusters=num_clusters[ind],
                    d_model=kwargs['embed_dim'],
                    l2_normalize_for_fps=False))
            # Last graph pooling is not needed
            if ind == len(depths) - 1:
                for param in pool.pool_block.fc1.parameters():
                    param.requires_grad = False
                for param in pool.pool_block.fc2.parameters():
                    param.requires_grad = False
                for param in pool.pool_block.centroid_fc.parameters():
                    param.requires_grad = False
            pools.append(pool)

        self.blocks1, self.pool1 = blocks[0], pools[0]
        self.blocks2, self.pool2 = blocks[1], pools[1]
        self.blocks3, self.pool3 = blocks[2], pools[2]
        self.blocks4, self.pool4 = blocks[3], pools[3]
        # --------------------------------------------------------------------------
        self.feat_dim = 64
        self.feats_layer = nn.Linear(self.embed_dim*self.feat_dim, 512) 
        self.feats_layer.apply(self._init_weights)

    def _block_operations(self, x, cls_token, x_pad_mask,
                          nn_block, pool_block, norm_block):
        """Wrapper to define operations per block.
        """
        # Forward nn block with cls_token and x
        cls_x = torch.cat([cls_token, x], dim=1)
        cls_x = nn_block(cls_x).type_as(x)
        cls_token, x = cls_x[:, :1, :], cls_x[:, 1:, :]

        # Perform pooling only on x
        cls_token, pool_logit, centroid, pool_pad_mask, pool_inds = (
            pool_block(cls_token, x, x_pad_mask)
        )
        if x.size(1) == self.feat_dim:
            feats = cls_x[:, 1:]
        else:
            feats = None
        # Generate output by cls_token
        if norm_block is not None:
            out = norm_block(cls_x)[:, 0]
        else:
            out = cls_x[:, 0]

        return (x, cls_token, pool_logit, centroid,
                pool_pad_mask, pool_inds, out, feats)

    def forward_features(self, x, y):
        x = self.patch_embed(x) # NxHxWxC Bx28x28x384
        N, H, W, C = x.shape
        # Collect features within each segment
        y = y.unsqueeze(1).float()
        y = F.interpolate(y, x.shape[1:3], mode='nearest')
        y = y.squeeze(1).long()  # Bx28x28
        x = segment_mean_nd(x, y) # Bx196x384   
        # Create padding mask
        ones = torch.ones((N, H, W, 1), dtype=x.dtype, device=x.device)
        avg_ones = segment_mean_nd(ones, y).squeeze(-1)
        x_padding_mask = avg_ones <= 0.5

        # Collect positional encodings within each segment
        pos_embed = self.pos_embed[:, 1:].view(1, H, W, C).expand(N, -1, -1, -1) 
        pos_embed = segment_mean_nd(pos_embed, y)  #Bx196x384

        # Add positional encodings
        x = self.pos_drop(x + pos_embed)  # Bx196x384

        # Add class token.
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) 
        cls_token = cls_token + self.pos_embed[:, :1]

        # intermediate results
        intermediates = {}

        # Block1
        (block1, cls_token1, pool_logit1, centroid1,
         pool_padding_mask1, pool_inds1, out1, _) = self._block_operations(
            x, cls_token, x_padding_mask,
            self.blocks1, self.pool1, None)
        #
        intermediates1 = {
            'logit1': pool_logit1, 'centroid1': centroid1, 'block1': block1,
            'padding_mask1': x_padding_mask, 'sampled_inds1': pool_inds1,
        }
        intermediates.update(intermediates1)


        # Block2
        (block2, cls_token2, pool_logit2, centroid2,
         pool_padding_mask2, pool_inds2, out2, feats) = self._block_operations(
            centroid1, cls_token1, pool_padding_mask1,
            self.blocks2, self.pool2, None)
        # cls_token2: Bx1x384, pool_padding_mask2: Bx32, centroid2: Bx32x384, out2: Bx384 logit: Bx64x32
        intermediates2 = {
            'logit2': pool_logit2, 'centroid2': centroid2, 'block2': block2,
            'padding_mask2': pool_padding_mask1, 'sampled_inds2': pool_inds2, 'out2': out2, 
                        'feats': feats,
        }
        intermediates.update(intermediates2)

        # Block3
        (block3, cls_token3, pool_logit3, centroid3,
         pool_padding_mask3, pool_inds3, out3, _) = self._block_operations(
            centroid2, cls_token2, pool_padding_mask2,
            self.blocks3, self.pool3, None)
        # cls_token3: Bx1x384, pool_padding_mask3: Bx16, centroid3: Bx16x384, out3: Bx384 logit: Bx32x16
        intermediates3 = {
            'logit3': pool_logit3, 'centroid3': centroid3, 'block3': block3,
            'padding_mask3': pool_padding_mask2, 'sampled_inds3': pool_inds3, 'out3': out3,

        }
        intermediates.update(intermediates3)

        # Block4
        (block4, cls_token4, pool_logit4, centroid4,
         pool_padding_mask4, pool_inds4, out4, _) = self._block_operations(
            centroid3, cls_token3, pool_padding_mask3,
            self.blocks4, self.pool4, self.norm)
        # cls_token4: Bx1x384, pool_padding_mask4: Bx8, centroid4: Bx8x384, out4: Bx384 logit: Bx16x8
        out4 = self.pre_logits(out4)

        intermediates4 = {
            'logit4': pool_logit4, 'centroid4': centroid4, 'block4': block4,
            'padding_mask4': pool_padding_mask3, 'out4': out4, 'sampled_inds4': pool_inds4,
            
        }
        intermediates.update(intermediates4)

        return intermediates

    def forward(self, x, y):
        intermediates = self.forward_features(x, y)  # B x 384

        if self.num_manufacturer:
            manu_out = self.manufacturer_head(intermediates['out4']) # B x 1000
            family_out = self.family_head(intermediates['out3'])
            out = self.head(intermediates['out2']) 
            feats = intermediates['feats']
            feats = self.feats_layer(feats.view(feats.size(0), -1)) #feats = self.feats_layer(feats)

            return out, family_out, manu_out, feats
    
        else:
            family_out = self.family_head(intermediates['out4'])
            out = self.head(intermediates['out3']) 

            return out, family_out


@register_model
def cast_small(pretrained=False, **kwargs):
    # minus one ViT block
    model = CAST(
        patch_size=8, embed_dim=384, num_clusters=[64, 32, 16, 8],
        depth=[3, 3, 3, 2], num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def cast_small_deep(pretrained=False, **kwargs):
    # minus one ViT block
    model = CAST(
        patch_size=8, embed_dim=384, num_clusters=[64, 32, 16, 8],
        depth=[6, 3, 3, 3], num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def cast_base(pretrained=False, **kwargs):
    # minus one ViT block
    model = CAST(
        patch_size=8, embed_dim=768, num_clusters=[64, 32, 16, 8],
        depth=[3, 3, 3, 2], num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def cast_base_deep(pretrained=False, **kwargs):
    # minus one ViT block
    model = CAST(
        patch_size=8, embed_dim=768, num_clusters=[64, 32, 16, 8],
        depth=[6, 3, 3, 3], num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model
