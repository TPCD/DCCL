import math
from functools import partial

import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import trunc_normal_
import copy
from model.vision_transformer import vit_small, vit_base
from model.meta_graph import GraphConvolution

class vit_backbone(nn.Module):
    def __init__(self, vit_backbone_model, grad_from_block=11):
        super().__init__()
        self.patch_embed = copy.deepcopy(vit_backbone_model.patch_embed)
        self.cls_token = copy.deepcopy(vit_backbone_model.cls_token)
        self.pos_embed = copy.deepcopy(vit_backbone_model.pos_embed)

        self.pos_drop = copy.deepcopy(vit_backbone_model.pos_drop)
        bottom_blocks = vit_backbone_model.blocks[:grad_from_block]
        self.bottom_blocks = copy.deepcopy(bottom_blocks)
        self.out_feat_dim = bottom_blocks[-1].norm1.weight.shape[0]
        del bottom_blocks

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

    def forward(self, x, return_all_patches=True):
        x = self.prepare_tokens(x)  # 64 197 768
        for blk in self.bottom_blocks:
            x = blk(x)
        if return_all_patches:
            return x
        else:
            return x[:, 0]


class vit_branch(nn.Module):
    def __init__(self, vit_backbone_model, grad_from_block=11):
        super().__init__()
        top_blocks = vit_backbone_model.blocks[grad_from_block:]
        self.top_blocks = copy.deepcopy(top_blocks)
        self.norm = copy.deepcopy(vit_backbone_model.norm)

    def forward(self, x, return_all_patches=False):
        for blk in self.top_blocks:
            x = blk(x)
        x = self.norm(x)
        if return_all_patches:
            return x
        else:
            return x[:, 0]


class AttributeTransformer(nn.Module):
    def __init__(self, vit_backbone_model, grad_from_block=11):
        super().__init__()
        self.feature_extractor = vit_backbone(vit_backbone_model=vit_backbone_model,
                                              grad_from_block=grad_from_block)
        for m in self.feature_extractor.parameters():
            m.requires_grad = False

        self.contrastive_branch = vit_branch(vit_backbone_model=vit_backbone_model,
                                             grad_from_block=grad_from_block)
        self.attribute_branch = vit_branch(vit_backbone_model=vit_backbone_model,
                                           grad_from_block=grad_from_block)

        del vit_backbone_model
        torch.cuda.empty_cache()

    def forward(self, x):
        z = self.feature_extractor(x, return_all_patches=True)
        at_embedding = self.attribute_branch(z, return_all_patches=False)
        co_embedding = self.contrastive_branch(z, return_all_patches=False)
        if self.training:
            return co_embedding, at_embedding
        else:
            return torch.cat((co_embedding, at_embedding), dim=1)


from torch.nn import MaxPool1d, AvgPool1d
import torch.nn.functional as F
from einops import rearrange


class ChannelMaxPoolFlat(MaxPool1d):
    def forward(self, input):
        if len(input.size()) == 4:
            n, c, w, h = input.size()
            pool = lambda x: F.max_pool1d(
                x,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                self.ceil_mode,
                self.return_indices,
            )
            out = rearrange(
                pool(rearrange(input, "n c w h -> n (w h) c")),
                "n (w h) c -> n c w h",
                n=n,
                w=w,
                h=h,
            )
            return out.squeeze()
        elif len(input.size()) == 3:
            n, c, l = input.size()
            pool = lambda x: F.max_pool1d(
                x,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                self.ceil_mode,
                self.return_indices,
            )
            out = rearrange(
                pool(rearrange(input, "n c l -> n l c")),
                "n l c -> n c l",
                n=n,
                l=l
            )
            return out.squeeze()
        else:
            raise NotImplementedError


class ChannelAvgPoolFlat(AvgPool1d):
    def forward(self, input):
        if len(input.size()) == 4:
            n, c, w, h = input.size()
            pool = lambda x: F.avg_pool1d(
                x,
                self.kernel_size,
                self.stride,
                self.padding,
                self.ceil_mode,
            )
            out = rearrange(
                pool(rearrange(input, "n c w h -> n (w h) c")),
                "n (w h) c -> n c w h",
                n=n,
                w=w,
                h=h,
            )
            return out.squeeze()
        elif len(input.size()) == 3:
            n, c, l = input.size()
            pool = lambda x: F.avg_pool1d(
                x,
                self.kernel_size,
                self.stride,
                self.padding,
                self.ceil_mode,
            )
            out = rearrange(
                pool(rearrange(input, "n c l -> n l c")),
                "n l c -> n c l",
                n=n,
                l=l
            )
            return out.squeeze()
        else:
            raise NotImplementedError


class AttributeTransformer2(nn.Module):
    def __init__(self, vit_backbone_model, num_attribute=28, feat_channal=197, grad_from_block=11):
        super().__init__()
        self.feature_extractor = vit_backbone(vit_backbone_model=vit_backbone_model,
                                              grad_from_block=grad_from_block)
        for m in self.feature_extractor.parameters():
            m.requires_grad = False

        self.contrastive_branch = vit_branch(vit_backbone_model=vit_backbone_model,
                                             grad_from_block=grad_from_block)
        # self.attribute_branch = vit_branch(vit_backbone_model=vit_backbone_model,
        #                                      grad_from_block=grad_from_block)

        self.attribute_branch_list = nn.ModuleList([nn.Sequential(nn.Conv1d(feat_channal, feat_channal, 1),
                                                                  nn.BatchNorm1d(feat_channal), nn.GELU(),
                                                                  ChannelMaxPoolFlat(feat_channal)) for i in
                                                    range(num_attribute)])

        del vit_backbone_model
        torch.cuda.empty_cache()

    def forward(self, x):
        z = self.feature_extractor(x, return_all_patches=True)
        head_embedding_list = []
        for att_head in self.attribute_branch_list:
            head_embedding_list.append(att_head(z))
        co_embedding = self.contrastive_branch(z, return_all_patches=False)
        if self.training:
            return co_embedding, head_embedding_list
        else:
            # return torch.cat((co_embedding, at_embedding), dim=1)
            return co_embedding


class AttributeTransformer3(nn.Module):
    def __init__(self, vit_backbone_model, num_attribute=28, feat_channal=197, grad_from_block=11):
        super().__init__()
        self.feature_extractor = vit_backbone(vit_backbone_model=vit_backbone_model,
                                              grad_from_block=grad_from_block)
        for m in self.feature_extractor.parameters():
            m.requires_grad = False

        self.contrastive_branch = vit_branch(vit_backbone_model=vit_backbone_model,
                                             grad_from_block=grad_from_block)
        # self.attribute_branch = vit_branch(vit_backbone_model=vit_backbone_model,
        #                                      grad_from_block=grad_from_block)
        self.attribute_branch_list = nn.ModuleList([nn.Sequential(nn.Conv1d(feat_channal, feat_channal, 1),
                                                                  nn.BatchNorm1d(feat_channal), nn.GELU(),
                                                                  ChannelPoolFlat(feat_channal)) for i in
                                                    range(num_attribute)])

        del vit_backbone_model
        torch.cuda.empty_cache()

    def forward(self, x):
        z = self.feature_extractor(x, return_all_patches=True)
        head_embedding_list = []

        co_embedding = self.contrastive_branch(z, return_all_patches=True)
        for att_head in self.attribute_branch_list:
            head_embedding_list.append(att_head(co_embedding))
        if self.training:
            return co_embedding[:, 0], head_embedding_list
        else:
            # return torch.cat((co_embedding, at_embedding), dim=1)
            return co_embedding[:, 0]


class AttributeTransformer4(nn.Module):
    def __init__(self, vit_backbone_model, num_attribute=28, feat_channal=197, grad_from_block=11):
        super().__init__()
        self.feature_extractor = vit_backbone(vit_backbone_model=vit_backbone_model,
                                              grad_from_block=grad_from_block)
        for m in self.feature_extractor.parameters():
            m.requires_grad = False
        backbone_feat_dim = self.feature_extractor.out_feat_dim
        self.contrastive_branch = vit_branch(vit_backbone_model=vit_backbone_model,
                                             grad_from_block=grad_from_block)
        self.attribute_embedding_branch = nn.Sequential(nn.Conv1d(backbone_feat_dim, 8, 1),
                                                        nn.BatchNorm1d(8), nn.GELU())

        self.attribute_branch_list = nn.ModuleList([nn.Sequential(nn.Conv1d(8, 8, 1),
                                                                  nn.BatchNorm1d(8), nn.GELU(),
                                                                  nn.AdaptiveMaxPool1d(1),
                                                                  nn.Flatten()) for i in
                                                    range(num_attribute)])

        del vit_backbone_model
        torch.cuda.empty_cache()

    def forward(self, x, concat=False):
        bs = x.size()[0]
        z = self.feature_extractor(x, return_all_patches=True)
        head_embedding_list = []

        co_embedding = self.contrastive_branch(z, return_all_patches=True)
        att_embedding = self.attribute_embedding_branch(torch.transpose(co_embedding, 1, 2))
        for att_head in self.attribute_branch_list:
            head_embedding_list.append(att_head(att_embedding))
        if self.training:
            return co_embedding[:, 0], head_embedding_list
        else:
            if concat:
                # return torch.cat((co_embedding, at_embedding), dim=1)
                return torch.cat((torch.nn.functional.normalize(co_embedding[:, 0]),
                                  torch.nn.functional.normalize(att_embedding.view(bs, -1))), dim=1)
            else:
                return co_embedding[:, 0]


class AttributeTransformer5(nn.Module):
    def __init__(self, vit_backbone_model, num_attribute=28, feat_channal=197, grad_from_block=11):
        super().__init__()
        self.feature_extractor = vit_backbone(vit_backbone_model=vit_backbone_model,
                                              grad_from_block=grad_from_block)
        for m in self.feature_extractor.parameters():
            m.requires_grad = False
        backbone_feat_dim = self.feature_extractor.out_feat_dim
        self.contrastive_branch = vit_branch(vit_backbone_model=vit_backbone_model,
                                             grad_from_block=grad_from_block)
        self.attribute_branch = vit_branch(vit_backbone_model=vit_backbone_model,
                                           grad_from_block=grad_from_block)

        self.attribute_embedding_branch = nn.Sequential(nn.Conv1d(backbone_feat_dim, 8, 1),
                                                        nn.BatchNorm1d(8), nn.GELU())

        self.attribute_branch_list = nn.ModuleList([nn.Sequential(nn.Conv1d(8, 8, 1),
                                                                  nn.BatchNorm1d(8), nn.GELU(),
                                                                  nn.AdaptiveMaxPool1d(1),
                                                                  nn.Flatten()) for i in
                                                    range(num_attribute)])

        del vit_backbone_model
        torch.cuda.empty_cache()

    def forward(self, x, concat=False):
        bs = x.size()[0]
        z = self.feature_extractor(x, return_all_patches=True)
        head_embedding_list = []

        co_embedding = self.contrastive_branch(z, return_all_patches=False)
        att_embedding = self.attribute_branch(z, return_all_patches=True)
        att_embedding = self.attribute_embedding_branch(torch.transpose(att_embedding, 1, 2))
        for att_head in self.attribute_branch_list:
            head_embedding_list.append(att_head(att_embedding))
        if self.training:
            return co_embedding, head_embedding_list
        else:
            if concat:
                # return torch.cat((co_embedding, at_embedding), dim=1)
                return torch.cat((torch.nn.functional.normalize(co_embedding),
                                  torch.nn.functional.normalize(att_embedding.view(bs, -1))), dim=1)
            else:
                return co_embedding


class AttributeTransformer6(nn.Module):
    def __init__(self, vit_backbone_model, num_attribute=28, attribute_feat_channal=8, grad_from_block=11):
        super().__init__()
        self.feature_extractor = vit_backbone(vit_backbone_model=vit_backbone_model,
                                              grad_from_block=grad_from_block)
        for m in self.feature_extractor.parameters():
            m.requires_grad = False
        backbone_feat_dim = vit_backbone_model.num_features
        self.contrastive_branch = vit_branch(vit_backbone_model=vit_backbone_model,
                                             grad_from_block=grad_from_block)
        self.attribute_branch = vit_branch(vit_backbone_model=vit_backbone_model,
                                           grad_from_block=grad_from_block)

        self.attribute_embedding_branch = nn.Sequential(nn.Conv1d(backbone_feat_dim, attribute_feat_channal, 1),
                                                        nn.BatchNorm1d(attribute_feat_channal), nn.GELU())

        self.attribute_branch_list = nn.ModuleList(
            [nn.Sequential(nn.Conv1d(attribute_feat_channal, attribute_feat_channal, 1),
                           nn.BatchNorm1d(attribute_feat_channal), nn.GELU(),
                           nn.AdaptiveMaxPool1d(1),
                           nn.Flatten()) for i in
             range(num_attribute)])

        self.attribute_out_pool = ChannelAvgPoolFlat(197)

        del vit_backbone_model
        torch.cuda.empty_cache()

    def forward(self, x, concat=False):
        z = self.feature_extractor(x, return_all_patches=True)
        head_embedding_list = []

        co_embedding = self.contrastive_branch(z, return_all_patches=False)
        att_embedding = self.contrastive_branch(z, return_all_patches=True)

        att_embedding_out = self.attribute_out_pool(att_embedding)
        att_embedding = self.attribute_embedding_branch(torch.transpose(att_embedding, 1, 2))
        for att_head in self.attribute_branch_list:
            head_embedding_list.append(att_head(att_embedding))
        if self.training:
            return co_embedding, head_embedding_list
        else:
            if concat:
                # return torch.cat((co_embedding, at_embedding), dim=1)
                return torch.cat(
                    (torch.nn.functional.normalize(co_embedding), torch.nn.functional.normalize(att_embedding_out)),
                    dim=1)
            else:
                return co_embedding


class AttributeTransformer7(nn.Module):
    def __init__(self, vit_backbone_model, num_attribute=28, attribute_feat_channal=8, grad_from_block=11):
        super().__init__()
        self.feature_extractor = vit_backbone(vit_backbone_model=vit_backbone_model,
                                              grad_from_block=grad_from_block)
        for m in self.feature_extractor.parameters():
            m.requires_grad = False
        backbone_feat_dim = self.feature_extractor.out_feat_dim
        self.contrastive_branch = vit_branch(vit_backbone_model=vit_backbone_model,
                                             grad_from_block=grad_from_block)

        self.attribute_embedding_branch = nn.Sequential(nn.Conv1d(backbone_feat_dim, attribute_feat_channal, 1),
                                                        nn.BatchNorm1d(attribute_feat_channal), nn.GELU())

        self.attribute_branch_list = nn.ModuleList(
            [nn.Sequential(nn.Conv1d(attribute_feat_channal, attribute_feat_channal, 1),
                           nn.BatchNorm1d(attribute_feat_channal), nn.GELU(),
                           nn.AdaptiveMaxPool1d(1),
                           nn.Flatten()) for i in
             range(num_attribute)])

        self.attribute_out_pool = ChannelAvgPoolFlat(197)

        del vit_backbone_model
        torch.cuda.empty_cache()

    def forward(self, x, concat=False):
        z = self.feature_extractor(x, return_all_patches=True)
        head_embedding_list = []

        co_embedding = self.contrastive_branch(z, return_all_patches=True)
        co_embedding_out = co_embedding[:, 0]
        # att_embedding = self.contrastive_branch(z, return_all_patches=True)

        att_embedding_out = self.attribute_out_pool(co_embedding)
        att_embedding = self.attribute_embedding_branch(torch.transpose(co_embedding[:, 1:], 1, 2))
        for att_head in self.attribute_branch_list:
            head_embedding_list.append(att_head(att_embedding))
        if self.training:
            return co_embedding_out, head_embedding_list
        else:
            if concat:
                # return torch.cat((co_embedding, at_embedding), dim=1)
                return torch.cat(
                    (torch.nn.functional.normalize(co_embedding_out), torch.nn.functional.normalize(att_embedding_out)),
                    dim=1)
            else:
                return co_embedding_out


class attribute_subnet(nn.Module):
    def __init__(self, input_feature_dim, norm_type='bn'):
        super().__init__()
        self.conv_1_1 = nn.Conv1d(input_feature_dim, input_feature_dim, 1)
        if norm_type == 'bn':
            self.norm1 = nn.BatchNorm1d(input_feature_dim)
        elif norm_type == 'ln':
            self.norm1 = nn.LayerNorm(input_feature_dim)
        elif norm_type == 'none' or norm_type is None:
            self.norm1 == nn.Identity()
        else:
            raise NotImplementedError
        self.activation = nn.GELU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.flatten = nn.Flatten()
        self.norm2 = nn.LayerNorm(input_feature_dim)

    def forward(self, x):
        out = self.flatten(self.pool(self.activation(self.norm1(self.conv_1_1(x)))))
        out = self.norm2(out)
        return out


class AttributeTransformer8(nn.Module):
    def __init__(self, vit_backbone_model, num_attribute=28, attribute_feat_channal=8, grad_from_block=11):
        super().__init__()
        self.feature_extractor = vit_backbone(vit_backbone_model=vit_backbone_model,
                                              grad_from_block=grad_from_block)
        for m in self.feature_extractor.parameters():
            m.requires_grad = False
        backbone_feat_dim = self.feature_extractor.out_feat_dim
        self.contrastive_branch = vit_branch(vit_backbone_model=vit_backbone_model,
                                             grad_from_block=grad_from_block)

        self.attribute_branch_list = nn.ModuleList([attribute_subnet(backbone_feat_dim) for i in
                                                    range(num_attribute)])

        self.attribute_out_pool = ChannelAvgPoolFlat(196)

        del vit_backbone_model
        torch.cuda.empty_cache()

    def forward(self, x, concat=False):
        z = self.feature_extractor(x, return_all_patches=True)
        head_embedding_list = []
        l2norm_head_embedding_list = []

        co_embedding = self.contrastive_branch(z, return_all_patches=True)
        co_embedding_out = co_embedding[:, 0]
        att_embedding = torch.transpose(co_embedding[:, 1:], 1, 2)
        for att_head in self.attribute_branch_list:
            _att_embedding = att_head(att_embedding)
            head_embedding_list.append(_att_embedding)
            l2norm_head_embedding_list.append(F.normalize(_att_embedding))
        a = torch.stack(l2norm_head_embedding_list, dim=1)  # 64*28*384
        b = torch.unsqueeze(co_embedding_out, dim=2)  # 64*384*1
        ab = F.softmax(torch.bmm(a, b), dim=1)  # 64*28*1
        a_t = torch.transpose(a, dim0=1, dim1=2)  # 64*384*28
        a_t_ab = torch.bmm(a_t, ab)
        a_t_ab = a_t_ab.squeeze()

        if self.training:
            return co_embedding_out, head_embedding_list
        else:
            if concat:
                return torch.cat(
                    (torch.nn.functional.normalize(co_embedding_out), torch.nn.functional.normalize(a_t_ab)), dim=1)
            else:
                return co_embedding_out


class AttributeTransformer9(nn.Module):
    def __init__(self, vit_backbone_model, num_attribute=28, attribute_feat_channal=8, grad_from_block=11):
        super().__init__()
        self.feature_extractor = vit_backbone(vit_backbone_model=vit_backbone_model,
                                              grad_from_block=grad_from_block)
        for m in self.feature_extractor.parameters():
            m.requires_grad = False
        backbone_feat_dim = self.feature_extractor.out_feat_dim
        self.contrastive_branch = vit_branch(vit_backbone_model=vit_backbone_model,
                                             grad_from_block=grad_from_block)
        self.attribute_branch_list = nn.ModuleList([attribute_subnet(backbone_feat_dim) for i in
                                                    range(num_attribute)])

        self.attribute_out_pool = ChannelAvgPoolFlat(196)

        del vit_backbone_model
        torch.cuda.empty_cache()

    def forward(self, x, concat=False):
        z = self.feature_extractor(x, return_all_patches=True)
        head_embedding_list = []
        l2norm_head_embedding_list = []

        co_embedding = self.contrastive_branch(z, return_all_patches=True)
        co_embedding_out = co_embedding[:, 0]
        att_embedding = torch.transpose(co_embedding[:, 1:], 1, 2)
        for att_head in self.attribute_branch_list:
            _att_embedding = att_head(att_embedding)
            head_embedding_list.append(_att_embedding)
            l2norm_head_embedding_list.append(F.normalize(_att_embedding))
        a = torch.stack(l2norm_head_embedding_list, dim=1)  # 64*28*384
        b = torch.unsqueeze(co_embedding_out, dim=2)  # 64*384*1
        ab = F.softmax(torch.bmm(a, b), dim=1)  # 64*28*1
        a_t = torch.transpose(a, dim0=1, dim1=2)  # 64*384*28
        a_t_ab = torch.bmm(a_t, ab)
        a_t_ab = a_t_ab.squeeze()
        # co_embedding_out += a_t_ab
        if self.training:
            return co_embedding_out + a_t_ab, head_embedding_list
        else:
            if concat:
                return torch.cat(
                    (torch.nn.functional.normalize(co_embedding_out), torch.nn.functional.normalize(a_t_ab)), dim=1)
            else:
                return co_embedding_out


class AttributeTransformer10(nn.Module):
    def __init__(self, vit_backbone_model, num_attribute=28, attribute_feat_channal=8, grad_from_block=11):
        super().__init__()
        self.feature_extractor = vit_backbone(vit_backbone_model=vit_backbone_model,
                                              grad_from_block=grad_from_block)
        for m in self.feature_extractor.parameters():
            m.requires_grad = False
        backbone_feat_dim = self.feature_extractor.out_feat_dim
        self.contrastive_branch = vit_branch(vit_backbone_model=vit_backbone_model,
                                             grad_from_block=grad_from_block)
        self.attribute_branch_list = nn.ModuleList([attribute_subnet(backbone_feat_dim) for i in
                                                    range(num_attribute)])

        del vit_backbone_model
        torch.cuda.empty_cache()

    def forward(self, x, concat=False):
        z = self.feature_extractor(x, return_all_patches=True)
        head_embedding_list = []
        l2norm_head_embedding_list = []

        co_embedding = self.contrastive_branch(z, return_all_patches=True)
        co_embedding_out = co_embedding[:, 0]
        att_embedding = torch.transpose(co_embedding[:, 1:], 1, 2)
        for att_head in self.attribute_branch_list:
            _att_embedding = att_head(att_embedding)
            head_embedding_list.append(_att_embedding)
            l2norm_head_embedding_list.append(F.normalize(_att_embedding))
        a = torch.stack(l2norm_head_embedding_list, dim=1)  # 64*28*384
        b = torch.unsqueeze(co_embedding_out, dim=2)  # 64*384*1
        ab = F.softmax(torch.bmm(a, b), dim=1)  # 64*28*1
        a_t = torch.transpose(a, dim0=1, dim1=2)  # 64*384*28
        a_t_ab = torch.bmm(a_t, ab)
        a_t_ab = a_t_ab.squeeze()
        # co_embedding_out += a_t_ab
        if self.training:
            return co_embedding_out + a_t_ab, head_embedding_list, co_embedding
        else:
            if concat:
                return torch.cat(
                    (torch.nn.functional.normalize(co_embedding_out), torch.nn.functional.normalize(a_t_ab)), dim=1)
            else:
                return co_embedding_out


class AttributeTransformer11(nn.Module):
    def __init__(self, vit_backbone_model, num_attribute=28, attribute_feat_channal=8, grad_from_block=11):
        super().__init__()
        self.feature_extractor = vit_backbone(vit_backbone_model=vit_backbone_model,
                                              grad_from_block=grad_from_block)
        for m in self.feature_extractor.parameters():
            m.requires_grad = False
        backbone_feat_dim = self.feature_extractor.out_feat_dim
        self.contrastive_branch = vit_branch(vit_backbone_model=vit_backbone_model,
                                             grad_from_block=grad_from_block)
        self.attribute_branch_list = nn.ModuleList([attribute_subnet(backbone_feat_dim) for i in
                                                    range(num_attribute)])

        del vit_backbone_model
        torch.cuda.empty_cache()

    def forward(self, x, concat=False):
        z = self.feature_extractor(x, return_all_patches=True)
        head_embedding_list = []
        l2norm_head_embedding_list = []

        co_embedding = self.contrastive_branch(z, return_all_patches=True)
        co_embedding_out = co_embedding[:, 0]
        att_embedding = torch.transpose(co_embedding[:, 1:], 1, 2)
        for att_head in self.attribute_branch_list:
            _att_embedding = att_head(att_embedding)
            head_embedding_list.append(_att_embedding)
            l2norm_head_embedding_list.append(F.normalize(_att_embedding))
        a = torch.stack(l2norm_head_embedding_list, dim=1)  # 64*28*384
        b = torch.unsqueeze(co_embedding_out, dim=2)  # 64*384*1
        ab = F.softmax(torch.bmm(a, b), dim=1)  # 64*28*1
        a_t = torch.transpose(a, dim0=1, dim1=2)  # 64*384*28
        a_t_ab = torch.bmm(a_t, ab)
        a_t_ab = a_t_ab.squeeze()
        # co_embedding_out += a_t_ab
        if self.training:
            return co_embedding_out + a_t_ab, head_embedding_list, co_embedding
        else:
            if concat:
                return z
            else:
                return co_embedding_out


class AttributeTransformer12(nn.Module):
    def __init__(self, vit_backbone_model, num_attribute=28, attribute_feat_channal=8, grad_from_block=11):
        super().__init__()
        self.feature_extractor = vit_backbone(vit_backbone_model=vit_backbone_model,
                                              grad_from_block=grad_from_block)
        for m in self.feature_extractor.parameters():
            m.requires_grad = False
        backbone_feat_dim = self.feature_extractor.out_feat_dim
        self.contrastive_branch = vit_branch(vit_backbone_model=vit_backbone_model,
                                             grad_from_block=grad_from_block)
        self.attribute_branch_list = nn.ModuleList([attribute_subnet(backbone_feat_dim) for i in
                                                    range(num_attribute)])

        del vit_backbone_model
        torch.cuda.empty_cache()

    def forward(self, x, concat=False):
        z = self.feature_extractor(x, return_all_patches=True)
        head_embedding_list = []
        l2norm_head_embedding_list = []

        co_embedding = self.contrastive_branch(z, return_all_patches=True)
        co_embedding_out = co_embedding[:, 0]
        att_embedding = torch.transpose(co_embedding[:, 1:], 1, 2)
        for att_head in self.attribute_branch_list:
            _att_embedding = att_head(att_embedding)
            head_embedding_list.append(_att_embedding)
            l2norm_head_embedding_list.append(F.normalize(_att_embedding))
        a = torch.stack(l2norm_head_embedding_list, dim=1)  # 64*28*384
        b = torch.unsqueeze(co_embedding_out, dim=2)  # 64*384*1
        ab = F.softmax(torch.bmm(a, b), dim=1)  # 64*28*1
        a_t = torch.transpose(a, dim0=1, dim1=2)  # 64*384*28
        a_t_ab = torch.bmm(a_t, ab)
        a_t_ab = a_t_ab.squeeze()
        # co_embedding_out += a_t_ab
        if self.training:
            return co_embedding_out, head_embedding_list, co_embedding
        else:
            return co_embedding_out


class AttributeTransformer13(nn.Module):
    def __init__(self, vit_backbone_model, num_attribute=28, attribute_feat_channal=8, grad_from_block=11):
        super().__init__()
        self.feature_extractor = vit_backbone(vit_backbone_model=vit_backbone_model,
                                              grad_from_block=grad_from_block)
        for m in self.feature_extractor.parameters():
            m.requires_grad = False
        backbone_feat_dim = self.feature_extractor.out_feat_dim
        self.contrastive_branch = vit_branch(vit_backbone_model=vit_backbone_model,
                                             grad_from_block=grad_from_block)
        self.num_features = backbone_feat_dim
        del vit_backbone_model
        torch.cuda.empty_cache()

    def forward(self, x, concat=False):
        z = self.feature_extractor(x, return_all_patches=True)
        head_embedding_list = []
        l2norm_head_embedding_list = []

        co_embedding = self.contrastive_branch(z, return_all_patches=True)
        co_embedding_out = co_embedding[:, 0]
        att_embedding = torch.transpose(co_embedding[:, 1:], 1, 2)

        if self.training:
            return co_embedding_out, att_embedding, co_embedding
        else:
            return co_embedding_out


class AttributeTransformer14(nn.Module):
    def __init__(self, vit_backbone_model, num_attribute=28, attribute_feat_channal=8, grad_from_block=11, device=None):
        super().__init__()
        self.device = device
        self.feature_extractor = vit_backbone(vit_backbone_model=vit_backbone_model,
                                              grad_from_block=grad_from_block)
        for m in self.feature_extractor.parameters():
            m.requires_grad = False
        backbone_feat_dim = self.feature_extractor.out_feat_dim
        self.contrastive_branch = vit_branch(vit_backbone_model=vit_backbone_model,
                                             grad_from_block=grad_from_block)
        self.attribute_branch_list = nn.ModuleList([attribute_subnet(backbone_feat_dim) for i in
                                                    range(num_attribute)])
        self.meta_graph = Meta_Graph1(hidden_dim=backbone_feat_dim, device=self.device)

        del vit_backbone_model
        torch.cuda.empty_cache()

    def forward(self, x, attribute_label=None, concat=False):
        head_embedding_list = []
        if attribute_label is not None:
            z = self.feature_extractor(x, return_all_patches=True)
            co_embedding = self.contrastive_branch(z, return_all_patches=True)
            co_embedding_out = co_embedding[:, 0]
            att_embedding = torch.transpose(co_embedding[:, 1:], 1, 2)
            for att_head in self.attribute_branch_list:
                _att_embedding = att_head(att_embedding)
                head_embedding_list.append(_att_embedding)
            att_graph_out = self.meta_graph(co_embedding_out, head_embedding_list, attribute_label)
            return co_embedding_out + att_graph_out, head_embedding_list, None

        else:
            z = self.feature_extractor(x, return_all_patches=True)
            co_embedding = self.contrastive_branch(z, return_all_patches=True)
            co_embedding_out = co_embedding[:, 0]
            if self.training:
                return co_embedding_out, None, None
            else:
                att_embedding = torch.transpose(co_embedding[:, 1:], 1, 2)
                for att_head in self.attribute_branch_list:
                    _att_embedding = att_head(att_embedding)
                    head_embedding_list.append(_att_embedding)
                att_graph_out = self.meta_graph(co_embedding_out, head_embedding_list)
                if concat:
                    return att_graph_out + F.normalize(co_embedding_out)
                else:
                    return co_embedding_out


class Meta_Graph1(nn.Module):
    def __init__(self, hidden_dim, device=torch.cuda.device('cuda')):
        super().__init__()


        self.device = device
        self.gcn = GraphConvolution(device=device,
                                    hidden_dim=hidden_dim,
                                    sparse_inputs=False,
                                    act=nn.Tanh(),
                                    bias=True, dropout=0.6).to(device=device)

        torch.cuda.empty_cache()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, attribute_feat=None, attribute_label=None):
        if attribute_label is not None:
            x_out = []
            adj = self.create_compositional_graph(attribute_label)
            attribute_feat_tensor = torch.stack(attribute_feat, dim=1)
            for _x, _att_f, _adj in zip(x, attribute_feat_tensor, adj):
                _vertex = torch.cat((_att_f, _x.unsqueeze(0)), dim=0)
                after_vertex = self.gcn(_vertex, _adj)
                x_out.append(after_vertex[-1])
            x_out = torch.stack(x_out, dim=0)
            return x_out
        else:
            l2norm_head_embedding_list = []
            for _att_f in attribute_feat:
                l2norm_head_embedding_list.append(F.normalize(_att_f))
            a = torch.stack(l2norm_head_embedding_list, dim=1)  # 64*28*384
            l2_x = F.normalize(x)
            b = torch.unsqueeze(l2_x, dim=2)  # 64*384*1
            ab = F.softmax(torch.bmm(a, b), dim=1)  # 64*28*1
            a_t = torch.transpose(a, dim0=1, dim1=2)  # 64*384*28
            a_t_ab = torch.bmm(a_t, ab)
            a_t_ab = a_t_ab.squeeze()
            return a_t_ab

    def create_compositional_graph(self, attribute_label):
        att_num = attribute_label.size(1)
        copy_attribute_label = attribute_label.detach()
        adj_list = []
        for row in copy_attribute_label:
            adj = torch.zeros((att_num + 1, att_num + 1))
            non_zero_positions = torch.nonzero(row)
            for p in non_zero_positions:
                adj[p, att_num] = 1
                adj[att_num, p] = 1
            adj_list.append(adj)
        adj_matrix = torch.stack(adj_list, dim=0).to(device=self.device)

        return adj_matrix


class Meta_Attribute_Generator1(nn.Module):
    def __init__(self, vit_backbone_model, dict_attribute, grad_from_block=11):
        super().__init__()
        backbone_feat_dim = vit_backbone_model.num_features
        self.num_attribute_class = len(dict_attribute.keys())
        self.num_attribute_all = sum([len(v) for v in dict_attribute.values()])
        self.attribute_generator_list = nn.ModuleList()

        self.meta_branch = vit_branch(vit_backbone_model=vit_backbone_model,
                                      grad_from_block=grad_from_block)

        for key in dict_attribute.keys():
            _conv = attribute_subnet(backbone_feat_dim)
            _classifier = nn.Linear(backbone_feat_dim, len(dict_attribute[key]) + 1)  # 1 for no present
            _softmax = nn.Softmax(dim=1)
            self.attribute_generator_list.append(nn.Sequential(_conv, _classifier, _softmax))

        del vit_backbone_model
        torch.cuda.empty_cache()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        fake_prob_list = []
        meta_embedding = self.meta_branch(x, return_all_patches=True)
        meta_embedding = torch.transpose(meta_embedding, 1, 2)
        for att_head in self.attribute_generator_list:
            fake_prob = att_head(meta_embedding)
            fake_prob_list.append(fake_prob)
        return fake_prob_list


class Meta_Attribute_Generator2(nn.Module):
    def __init__(self, vit_backbone_model, dict_attribute, grad_from_block=11):
        super().__init__()
        backbone_feat_dim = vit_backbone_model.num_features
        self.num_attribute_class = len(dict_attribute.keys())
        self.num_attribute_all = sum([len(v) for v in dict_attribute.values()])
        self.attribute_generator_list = nn.ModuleList()

        for key in dict_attribute.keys():
            _conv = attribute_subnet(backbone_feat_dim)
            _classifier = nn.Linear(backbone_feat_dim, len(dict_attribute[key]) + 1)  # 1 for no present
            _softmax = nn.Softmax(dim=1)
            self.attribute_generator_list.append(nn.Sequential(_conv, _classifier, _softmax))

        del vit_backbone_model
        torch.cuda.empty_cache()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        fake_prob_list = []
        meta_embedding = torch.transpose(x, 1, 2)
        for att_head in self.attribute_generator_list:
            fake_prob = att_head(meta_embedding)
            fake_prob_list.append(fake_prob)
        return fake_prob_list


def at_small(pretrain_path):
    student = vit_small()
    weight = torch.load(pretrain_path, map_location='cpu')
    msg = student.load_state_dict(weight, strict=False)
    print(msg)
    model = AttributeTransformer(student)
    return model


def at_base(pretrain_path):
    student = vit_base()
    weight = torch.load(pretrain_path, map_location='cpu')
    msg = student.load_state_dict(weight, strict=False)
    print(msg)
    model = AttributeTransformer(student)
    return model


def at2_base(pretrain_path, num_attribute=28, feat_channal=197, grad_from_block=11):
    student = vit_base()
    weight = torch.load(pretrain_path, map_location='cpu')
    msg = student.load_state_dict(weight, strict=False)
    print(msg)
    model = AttributeTransformer2(student, num_attribute, feat_channal, grad_from_block)
    return model


def at2_small(pretrain_path, num_attribute=28, feat_channal=197, grad_from_block=11):
    student = vit_small()
    weight = torch.load(pretrain_path, map_location='cpu')
    msg = student.load_state_dict(weight, strict=False)
    print(msg)
    model = AttributeTransformer2(student, num_attribute, feat_channal, grad_from_block)
    return model


def at3_small(pretrain_path, num_attribute=28, feat_channal=197, grad_from_block=11):
    student = vit_small()
    weight = torch.load(pretrain_path, map_location='cpu')
    msg = student.load_state_dict(weight, strict=False)
    print(msg)
    model = AttributeTransformer3(student, num_attribute, feat_channal, grad_from_block)
    return model


def at4_small(pretrain_path, num_attribute=28, feat_channal=197, grad_from_block=11):
    student = vit_small()
    weight = torch.load(pretrain_path, map_location='cpu')
    msg = student.load_state_dict(weight, strict=False)
    print(msg)
    model = AttributeTransformer4(student, num_attribute, feat_channal, grad_from_block)
    return model


def at4_base(pretrain_path, num_attribute=28, feat_channal=197, grad_from_block=11):
    student = vit_base()
    weight = torch.load(pretrain_path, map_location='cpu')
    msg = student.load_state_dict(weight, strict=False)
    print(msg)
    model = AttributeTransformer4(student, num_attribute, feat_channal, grad_from_block)
    return model


def at5_small(pretrain_path, num_attribute=28, feat_channal=197, grad_from_block=11):
    student = vit_small()
    weight = torch.load(pretrain_path, map_location='cpu')
    msg = student.load_state_dict(weight, strict=False)
    print(msg)
    model = AttributeTransformer5(student, num_attribute, feat_channal, grad_from_block)
    return model


def at5_base(pretrain_path, num_attribute=28, feat_channal=197, grad_from_block=11):
    student = vit_base()
    weight = torch.load(pretrain_path, map_location='cpu')
    msg = student.load_state_dict(weight, strict=False)
    print(msg)
    model = AttributeTransformer5(student, num_attribute, feat_channal, grad_from_block)
    return model


def at6_small(pretrain_path, num_attribute=28, attribute_feat_channal=8, grad_from_block=11):
    student = vit_small()
    weight = torch.load(pretrain_path, map_location='cpu')
    msg = student.load_state_dict(weight, strict=False)
    print(msg)
    model = AttributeTransformer6(student, num_attribute, attribute_feat_channal, grad_from_block)
    return model


def at6_base(pretrain_path, num_attribute=28, attribute_feat_channal=8, grad_from_block=11):
    student = vit_base()
    weight = torch.load(pretrain_path, map_location='cpu')
    msg = student.load_state_dict(weight, strict=False)
    print(msg)
    model = AttributeTransformer6(student, num_attribute, attribute_feat_channal, grad_from_block)
    return model


def at7_small(pretrain_path, num_attribute=28, attribute_feat_channal=8, grad_from_block=11):
    student = vit_small()
    weight = torch.load(pretrain_path, map_location='cpu')
    msg = student.load_state_dict(weight, strict=False)
    print(msg)
    model = AttributeTransformer7(student, num_attribute, attribute_feat_channal, grad_from_block)
    return model


def at7_base(pretrain_path, num_attribute=28, attribute_feat_channal=8, grad_from_block=11):
    student = vit_base()
    weight = torch.load(pretrain_path, map_location='cpu')
    msg = student.load_state_dict(weight, strict=False)
    print(msg)
    model = AttributeTransformer7(student, num_attribute, attribute_feat_channal, grad_from_block)
    return model


def at8_small(pretrain_path, num_attribute=28, attribute_feat_channal=8, grad_from_block=11):
    student = vit_small()
    weight = torch.load(pretrain_path, map_location='cpu')
    msg = student.load_state_dict(weight, strict=False)
    print(msg)
    model = AttributeTransformer8(student, num_attribute, attribute_feat_channal, grad_from_block)
    return model


def at8_base(pretrain_path, num_attribute=28, attribute_feat_channal=8, grad_from_block=11):
    student = vit_base()
    weight = torch.load(pretrain_path, map_location='cpu')
    msg = student.load_state_dict(weight, strict=False)
    print(msg)
    model = AttributeTransformer8(student, num_attribute, attribute_feat_channal, grad_from_block)
    return model


def at9_small(pretrain_path, num_attribute=28, attribute_feat_channal=8, grad_from_block=11):
    student = vit_small()
    weight = torch.load(pretrain_path, map_location='cpu')
    msg = student.load_state_dict(weight, strict=False)
    print(msg)
    model = AttributeTransformer9(student, num_attribute, attribute_feat_channal, grad_from_block)
    return model


def at9_base(pretrain_path, num_attribute=28, attribute_feat_channal=8, grad_from_block=11):
    student = vit_base()
    weight = torch.load(pretrain_path, map_location='cpu')
    msg = student.load_state_dict(weight, strict=False)
    print(msg)
    model = AttributeTransformer9(student, num_attribute, attribute_feat_channal, grad_from_block)
    return model


def at10_small(pretrain_path, num_attribute=28, attribute_feat_channal=8, grad_from_block=11):
    student = vit_small()
    weight = torch.load(pretrain_path, map_location='cpu')
    msg = student.load_state_dict(weight, strict=False)
    print(msg)
    model = AttributeTransformer10(student, num_attribute, attribute_feat_channal, grad_from_block)
    return model


def at10_base(pretrain_path, num_attribute=28, attribute_feat_channal=8, grad_from_block=11):
    student = vit_base()
    weight = torch.load(pretrain_path, map_location='cpu')
    msg = student.load_state_dict(weight, strict=False)
    print(msg)
    model = AttributeTransformer10(student, num_attribute, attribute_feat_channal, grad_from_block)
    return model


def at11_small(pretrain_path, num_attribute=28, attribute_feat_channal=8, grad_from_block=11):
    student = vit_small()
    weight = torch.load(pretrain_path, map_location='cpu')
    msg = student.load_state_dict(weight, strict=False)
    print(msg)
    model = AttributeTransformer11(student, num_attribute, attribute_feat_channal, grad_from_block)
    return model


def at11_base(pretrain_path, num_attribute=28, attribute_feat_channal=8, grad_from_block=11):
    student = vit_base()
    weight = torch.load(pretrain_path, map_location='cpu')
    msg = student.load_state_dict(weight, strict=False)
    print(msg)
    model = AttributeTransformer11(student, num_attribute, attribute_feat_channal, grad_from_block)
    return model


def at12_small(pretrain_path, num_attribute=28, attribute_feat_channal=8, grad_from_block=11):
    student = vit_small()
    weight = torch.load(pretrain_path, map_location='cpu')
    msg = student.load_state_dict(weight, strict=False)
    print(msg)
    model = AttributeTransformer12(student, num_attribute, attribute_feat_channal, grad_from_block)
    return model


def at12_base(pretrain_path, num_attribute=28, attribute_feat_channal=8, grad_from_block=11):
    student = vit_base()
    weight = torch.load(pretrain_path, map_location='cpu')
    msg = student.load_state_dict(weight, strict=False)
    print(msg)
    model = AttributeTransformer12(student, num_attribute, attribute_feat_channal, grad_from_block)
    return model


def at13_small(pretrain_path, num_attribute=28, attribute_feat_channal=8, grad_from_block=11):
    student = vit_small()
    weight = torch.load(pretrain_path, map_location='cpu')
    msg = student.load_state_dict(weight, strict=False)
    print(msg)
    model = AttributeTransformer13(student, num_attribute, attribute_feat_channal, grad_from_block)
    return model


def at13_base(pretrain_path, num_attribute=28, attribute_feat_channal=8, grad_from_block=11):
    student = vit_base()
    weight = torch.load(pretrain_path, map_location='cpu')
    msg = student.load_state_dict(weight, strict=False)
    print(msg)
    model = AttributeTransformer13(student, num_attribute, attribute_feat_channal, grad_from_block)
    return model

def at14_small(pretrain_path, num_attribute=28, attribute_feat_channal=8, grad_from_block=11, device=torch.cuda.device('cuda')):
    student = vit_small()
    weight = torch.load(pretrain_path, map_location='cpu')
    msg = student.load_state_dict(weight, strict=False)
    print(msg)
    model = AttributeTransformer14(student, num_attribute, attribute_feat_channal, grad_from_block, device)
    return model


def at14_base(pretrain_path, num_attribute=28, attribute_feat_channal=8, grad_from_block=11,  device=torch.cuda.device('cuda')):
    student = vit_base()
    weight = torch.load(pretrain_path, map_location='cpu')
    msg = student.load_state_dict(weight, strict=False)
    print(msg)
    model = AttributeTransformer14(student, num_attribute, attribute_feat_channal, grad_from_block, device)
    return model

def meta1_small(pretrain_path, dict_attribute, grad_from_block=11):
    student = vit_small()
    weight = torch.load(pretrain_path, map_location='cpu')
    msg = student.load_state_dict(weight, strict=False)
    print(msg)
    model = Meta_Attribute_Generator1(student, dict_attribute, grad_from_block=grad_from_block)
    return model


def meta1_base(pretrain_path, dict_attribute, grad_from_block=11):
    student = vit_base()
    weight = torch.load(pretrain_path, map_location='cpu')
    msg = student.load_state_dict(weight, strict=False)
    print(msg)
    model = Meta_Attribute_Generator1(student, dict_attribute, grad_from_block=grad_from_block)
    return model


def meta2_small(pretrain_path, dict_attribute, grad_from_block=11):
    student = vit_small()
    weight = torch.load(pretrain_path, map_location='cpu')
    msg = student.load_state_dict(weight, strict=False)
    print(msg)
    model = Meta_Attribute_Generator2(student, dict_attribute, grad_from_block=grad_from_block)
    return model


def meta2_base(pretrain_path, dict_attribute, grad_from_block=11):
    student = vit_base()
    weight = torch.load(pretrain_path, map_location='cpu')
    msg = student.load_state_dict(weight, strict=False)
    print(msg)
    model = Meta_Attribute_Generator2(student, dict_attribute, grad_from_block=grad_from_block)
    return model