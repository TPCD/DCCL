# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from functools import partial

import torch
import torch.nn as nn

from torch.nn.init import trunc_normal_

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if return_attention:
            return x, attn
        else:
            return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x, return_all_patches=False):
        x = self.prepare_tokens(x)# 64 197 768
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if return_all_patches:
            return x
        else:
            return x[:, 0]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                x, attn = blk(x, return_attention=True)
                x = self.norm(x)
                return x, attn

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class VisionTransformerWithLinear(nn.Module):

    def __init__(self, base_vit, num_classes=200):

        super().__init__()

        self.base_vit = base_vit
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x, return_features=False):

        features = self.base_vit(x)
        features = torch.nn.functional.normalize(features, dim=-1)
        logits = self.fc(features)

        if return_features:
            return logits, features
        else:
            return logits

    @torch.no_grad()
    def normalize_prototypes(self):
        w = self.fc.weight.data.clone()
        w = torch.nn.functional.normalize(w, dim=1, p=2)
        self.fc.weight.copy_(w)




class Attribute_Classifier(nn.Module):
    def __init__(self, dict_attribute, in_dim, projected_dim, norm_type='bn', use_independent_projection=True):
        super().__init__()
        print(dict_attribute)
        self.num_attribute_class = len(dict_attribute.keys())
        self.num_attribute_all = sum([len(v) for v in dict_attribute.values()])
        self.classifier_list = nn.ModuleList()
        self._log_softmax = nn.LogSoftmax(dim=1)
        if norm_type is None or norm_type == 'none':
            use_norm = False
        elif norm_type == 'bn':
            norm_class = nn.BatchNorm1d
            use_norm = True
        elif norm_type == 'ln':
            norm_class = nn.LayerNorm
            use_norm = True
        else:
            raise NotImplementedError

        if use_independent_projection:
            self.shared_projected_layer = None
        else:
            if use_norm:
                self.shared_projected_layer = nn.Sequential(nn.Linear(in_dim, projected_dim * 2),
                                                            norm_class(projected_dim * 2),
                                                            nn.GELU(),
                                                            nn.Linear(projected_dim * 2, projected_dim),
                                                            norm_class(projected_dim),
                                                            )
            else:
                self.shared_projected_layer = nn.Sequential(nn.Linear(in_dim, projected_dim * 2),
                                                            nn.GELU(),
                                                            nn.Linear(projected_dim * 2, projected_dim)
                                                            )

        for key in dict_attribute.keys():
            if use_independent_projection:
                layers = [nn.Linear(in_dim, projected_dim)]
                if use_norm:
                    layers.append(norm_class(projected_dim))
                layers.append(nn.GELU())
                layers.append(nn.Linear(projected_dim, len(dict_attribute[key]) + 1)) # 1 for no present
                self.classifier_list.append(nn.Sequential(*layers))
            else:
                _classifier = nn.Linear(projected_dim, len(dict_attribute[key]) + 1)  # 1 for no present
                self.classifier_list.append(_classifier)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        probability = []
        if self.shared_projected_layer is None:
            attribute_embedding = []
            for classifier in self.classifier_list:
                projected = classifier[0](x)
                _attribute_embedding = classifier[1](projected)
                attribute_embedding.append(_attribute_embedding.detach().clone())
                activate = classifier[2](_attribute_embedding)
                logit = classifier[3](activate)
                probability.append(self._log_softmax(logit))
            attribute_embedding = torch.cat(attribute_embedding, dim=1)
        else:
            attribute_embedding = self.shared_projected_layer(x)
            for classifier in self.classifier_list:
                logit = classifier(attribute_embedding)
                probability.append(self._log_softmax(logit))
        if self.training == True:
            return probability
        else:
            return attribute_embedding



class Attribute_Classifier2(nn.Module):
    def __init__(self, dict_attribute, in_dim, projected_dim, out_dim, norm_type='bn'):
        super().__init__()
        print(dict_attribute)
        self.num_attribute_class = len(dict_attribute.keys())
        self.num_attribute_all = sum([len(v) for v in dict_attribute.values()])
        self.classifier_list = nn.ModuleList()
        self._log_softmax = nn.LogSoftmax(dim=1)
        if norm_type is None or norm_type == 'none':
            use_norm = False
        elif norm_type == 'bn':
            norm_class = nn.BatchNorm1d
            use_norm = True
        elif norm_type == 'ln':
            norm_class = nn.LayerNorm
            use_norm = True
        else:
            raise NotImplementedError

        for key in dict_attribute.keys():
            individual_head = nn.ModuleList()

            embedding_layers = [nn.Linear(in_dim, projected_dim)]
            if use_norm:
                embedding_layers.append(norm_class(projected_dim))
            embedding_layers.append(nn.GELU())
            embedding_layers = nn.Sequential(*embedding_layers)
            output_layers = [nn.Linear(projected_dim, out_dim)]
            if use_norm:
                output_layers.append(norm_class(out_dim))
            output_layers = nn.Sequential(*output_layers)
            individual_head.append(embedding_layers)
            individual_head.append(output_layers)
            individual_head.append(nn.Linear(out_dim, len(dict_attribute[key]) + 1)) # 1 for no present

            self.classifier_list.append(individual_head)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        probability = []
        attribute_embedding = []
        for individual_head in self.classifier_list:
            projected = individual_head[0](x)
            _attribute_embedding = individual_head[1](projected)
            attribute_embedding.append(_attribute_embedding.detach().clone())
            logit = individual_head[2](_attribute_embedding)
            probability.append(self._log_softmax(logit))
        attribute_embedding = torch.cat(attribute_embedding, dim=1)

        if self.training == True:
            return probability
        else:
            return attribute_embedding


class Attribute_Classifier3(nn.Module):
    def __init__(self, dict_attribute, in_dim, projected_dim, norm_type='bn', use_independent_projection=True):
        super().__init__()
        print(dict_attribute)
        self.num_attribute_class = len(dict_attribute.keys())
        self.num_attribute_all = sum([len(v) for v in dict_attribute.values()])
        self.classifier_list = nn.ModuleList()
        self._log_softmax = nn.LogSoftmax(dim=1)
        if norm_type is None or norm_type == 'none':
            use_norm = False
        elif norm_type == 'bn':
            norm_class = nn.BatchNorm1d
            use_norm = True
        elif norm_type == 'ln':
            norm_class = nn.LayerNorm
            use_norm = True
        else:
            raise NotImplementedError

        if use_independent_projection:
            self.shared_projected_layer = None
        else:
            if use_norm:
                self.shared_projected_layer = nn.Sequential(nn.Linear(in_dim, projected_dim * 2),
                                                            norm_class(projected_dim * 2),
                                                            nn.GELU(),
                                                            nn.Linear(projected_dim * 2, projected_dim),
                                                            norm_class(projected_dim),
                                                            )
            else:
                self.shared_projected_layer = nn.Sequential(nn.Linear(in_dim, projected_dim * 2),
                                                            nn.GELU(),
                                                            nn.Linear(projected_dim * 2, projected_dim)
                                                            )

        for key in dict_attribute.keys():
            if use_independent_projection:
                layers = [nn.Linear(in_dim, projected_dim)]
                if use_norm:
                    layers.append(norm_class(projected_dim))
                layers.append(nn.GELU())
                layers.append(nn.Linear(projected_dim, len(dict_attribute[key]) + 1)) # 1 for no present
                self.classifier_list.append(nn.Sequential(*layers))
            else:
                _classifier = nn.Linear(projected_dim, len(dict_attribute[key]) + 1)  # 1 for no present
                self.classifier_list.append(_classifier)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        probability = []
        if self.shared_projected_layer is None:
            attribute_embedding = []
            for classifier in self.classifier_list:
                projected = classifier[0](x)
                _attribute_embedding = classifier[1](projected)
                attribute_embedding.append(_attribute_embedding.detach().clone())
                activate = classifier[2](_attribute_embedding)
                logit = classifier[3](activate)
                probability.append(self._log_softmax(logit))
            attribute_embedding = torch.cat(attribute_embedding, dim=1)
        else:
            attribute_embedding = self.shared_projected_layer(x)
            for classifier in self.classifier_list:
                logit = classifier(attribute_embedding)
                probability.append(self._log_softmax(logit))
        if self.training == True:
            return probability
        else:
            return attribute_embedding


class Attribute_BN_Classifier(nn.Module):
    def __init__(self, dict_attribute, in_dim, projected_dim, norm_type='bn', use_independent_projection=True):
        super().__init__()
        print(dict_attribute)
        self.num_attribute_class = len(dict_attribute.keys())
        self.num_attribute_all = sum([len(v) for v in dict_attribute.values()])
        self.classifier_list = nn.ModuleList()
        self._log_softmax = nn.LogSoftmax(dim=1)
        if norm_type is None or norm_type == 'none':
            use_norm = False
        elif norm_type == 'bn':
            norm_class = nn.BatchNorm1d
            use_norm = True
        elif norm_type == 'ln':
            norm_class = nn.LayerNorm
            use_norm = True
        else:
            raise NotImplementedError

        if use_independent_projection:
            self.shared_projected_layer = None
        else:
            if use_norm:
                self.shared_projected_layer = nn.Sequential(nn.Linear(in_dim, projected_dim * 2),
                                                            norm_class(projected_dim * 2),
                                                            nn.GELU(),
                                                            nn.Linear(projected_dim * 2, projected_dim),
                                                            norm_class(projected_dim),
                                                            )
            else:
                self.shared_projected_layer = nn.Sequential(nn.Linear(in_dim, projected_dim * 2),
                                                            nn.GELU(),
                                                            nn.Linear(projected_dim * 2, projected_dim)
                                                            )

        for key in dict_attribute.keys():
            if use_independent_projection:
                layers = [nn.Linear(in_dim, projected_dim)]
                if use_norm:
                    layers.append(norm_class(projected_dim))
                layers.append(nn.GELU())

                layers.append(BNClassifier(projected_dim, len(dict_attribute[key]) + 1))  # 1 for no present
                self.classifier_list.append(nn.Sequential(*layers))
            else:
                _classifier = BNClassifier(projected_dim, len(dict_attribute[key]) + 1)  # 1 for no present
                self.classifier_list.append(_classifier)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        probability = []
        if self.shared_projected_layer is None:
            attribute_embedding = []
            for classifier in self.classifier_list:
                projected = classifier[0](x)
                _attribute_embedding = classifier[1](projected)
                activate = classifier[2](_attribute_embedding)
                logit = classifier[3](activate)
                if not self.training:
                    attribute_embedding.append(logit.detach().clone())
                probability.append(self._log_softmax(logit))
            if not self.training:
                attribute_embedding = torch.cat(attribute_embedding, dim=1)
        else:
            attribute_embedding = self.shared_projected_layer(x)
            for classifier in self.classifier_list:
                logit = classifier(attribute_embedding)
                probability.append(self._log_softmax(logit))
        if self.training == True:
            return probability
        else:
            return attribute_embedding


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class BNClassifier(nn.Module):
    '''bn + fc'''

    def __init__(self, in_dim, class_num):
        super(BNClassifier, self).__init__()

        self.in_dim = in_dim
        self.class_num = class_num

        self.bn = nn.BatchNorm1d(self.in_dim)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.in_dim, self.class_num, bias=False)

        self.bn.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        feature = self.bn(x)
        if not self.training:
            return feature
        else:
            cls_score = self.classifier(feature)
            return cls_score

class Attribute_Classifier5(nn.Module):
    def __init__(self, dict_attribute, in_dim, projected_dim, norm_type='bn', use_independent_projection=True):
        super().__init__()
        print(dict_attribute)
        self.num_attribute_class = len(dict_attribute.keys())
        self.num_attribute_all = sum([len(v) for v in dict_attribute.values()])
        self.classifier_list = nn.ModuleList()
        self._log_softmax = nn.LogSoftmax(dim=1)
        if norm_type is None or norm_type == 'none':
            use_norm = False
        elif norm_type == 'bn':
            norm_class = nn.BatchNorm1d
            use_norm = True
        elif norm_type == 'ln':
            norm_class = nn.LayerNorm
            use_norm = True
        else:
            raise NotImplementedError

        if use_independent_projection:
            self.shared_projected_layer = None
        else:
            if use_norm:
                self.shared_projected_layer = nn.Sequential(nn.Linear(in_dim, projected_dim),
                                                            norm_class(projected_dim),
                                                            nn.GELU(),
                                                            )
            else:
                self.shared_projected_layer = nn.Sequential(nn.Linear(in_dim, projected_dim),
                                                            nn.GELU(),
                                                            )

        for key in dict_attribute.keys():
            if use_independent_projection:
                layers = [nn.Linear(in_dim, projected_dim)]
                if use_norm:
                    layers.append(norm_class(projected_dim))
                layers.append(nn.GELU())
                layers.append(nn.Linear(projected_dim, len(dict_attribute[key]) + 1))  # 1 for no present
                self.classifier_list.append(nn.Sequential(*layers))
            else:
                _classifier = nn.Linear(projected_dim, len(dict_attribute[key]) + 1)  # 1 for no present
                self.classifier_list.append(_classifier)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        probability = []
        if self.shared_projected_layer is None:
            attribute_embedding = []
            for classifier in self.classifier_list:
                projected = classifier[0](x)
                bn_projected = classifier[1](projected)
                attribute_embedding.append(bn_projected.detach().clone())
                activate = classifier[2](bn_projected)
                logit = classifier[3](activate)
                # attribute_embedding.append(logit.detach().clone())
                probability.append(self._log_softmax(logit))
            attribute_embedding = torch.cat(attribute_embedding, dim=1)
        else:
            attribute_embedding = self.shared_projected_layer(x)
            for classifier in self.classifier_list:
                logit = classifier(attribute_embedding)
                probability.append(self._log_softmax(logit))
        if self.training == True:
            return probability
        else:
            return attribute_embedding


class Attribute_Classifier6ind(nn.Module):
    def __init__(self, dict_attribute, in_dim):
        super().__init__()
        print(dict_attribute)
        self.num_attribute_class = len(dict_attribute.keys())
        self.num_attribute_all = sum([len(v) for v in dict_attribute.values()])
        self.classifier_list = nn.ModuleList()
        self._log_softmax = nn.LogSoftmax(dim=1)

        for key in dict_attribute.keys():
                _classifier = nn.Linear(in_dim, len(dict_attribute[key]) + 1)  # 1 for no present
                self.classifier_list.append(_classifier)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        probability = []
        for i, classifier in enumerate(self.classifier_list):
            logit = classifier(x[i])
            probability.append(self._log_softmax(logit))

        if self.training == True:
            return probability
        else:
            return torch.cat(x, dim=1)

class Attribute_Classifier7ind(nn.Module):
    def __init__(self, dict_attribute, in_dim):
        super().__init__()
        print(dict_attribute)
        self.num_attribute_class = len(dict_attribute.keys())
        self.num_attribute_all = sum([len(v) for v in dict_attribute.values()])
        self.classifier_list = nn.ModuleList()
        self._log_softmax = nn.LogSoftmax(dim=1)
        self._dropout = nn.Dropout(0.2)

        for key in dict_attribute.keys():
                _classifier = nn.Linear(in_dim, len(dict_attribute[key]) + 1)  # 1 for no present
                self.classifier_list.append(_classifier)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        probability = []
        for i, classifier in enumerate(self.classifier_list):
            dropped_x = self._dropout(x[i])
            logit = classifier(dropped_x)
            probability.append(logit)

        if self.training == True:
            return probability
        else:
            return torch.cat(x, dim=1)



class Attribute_Classifier8ind(nn.Module):
    def __init__(self, dict_attribute, in_dim):
        super().__init__()
        print(dict_attribute)
        self.num_attribute_class = len(dict_attribute.keys())
        self.num_attribute_all = sum([len(v) for v in dict_attribute.values()])
        self.classifier_list = nn.ModuleList()
        self._log_softmax = nn.LogSoftmax(dim=1)
        self._dropout = nn.Dropout(0.2)
        self.global_classifier = nn.Linear(in_dim, 200)

        for key in dict_attribute.keys():
                _classifier = nn.Linear(in_dim, len(dict_attribute[key]) + 1)  # 1 for no present
                self.classifier_list.append(_classifier)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, use_log_softmax=False):

        probability = []
        for i, classifier in enumerate(self.classifier_list):
            dropped_x = self._dropout(x[i])
            logit = classifier(dropped_x)
            if use_log_softmax:
                logit = self._log_softmax(logit)
            probability.append(logit)

        if self.training == True:
            return probability
        else:
            return torch.cat(x, dim=1)

