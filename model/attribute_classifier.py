import math
from functools import partial
import torch
import torch.nn as nn

from torch.nn.init import trunc_normal_

class Attribute_Classifier(nn.Module):
    def __init__(self, dict_attribute, in_dim, projected_dim, use_bn=False, use_independent_projection=True):
        super().__init__()
        print(dict_attribute)
        self.num_attribute_class = len(dict_attribute.keys())
        self.num_attribute_all = sum([len(v) for v in dict_attribute.values()])
        self.classifier_list = nn.ModuleList()
        self._softmax = nn.Softmax()
        if use_independent_projection:
            self.shared_projected_layer = None
        else:
            if use_bn:
                self.shared_projected_layer = nn.Sequential(nn.Linear(in_dim, projected_dim),
                                                            nn.BatchNorm1d(projected_dim),
                                                            nn.GELU())
            else:
                self.shared_projected_layer = nn.Sequential(nn.Linear(in_dim, projected_dim),
                                                            nn.GELU())
        for key in dict_attribute.keys():
            if use_independent_projection:
                layers = [nn.Linear(in_dim, projected_dim)]
                if use_bn:
                    layers.append(nn.BatchNorm1d(projected_dim))
                layers.append(nn.GELU())
                layers.append(nn.Linear(projected_dim, len(dict_attribute[key]) + 1)) # 1 for no present
                self.classifier_list.append(nn.Sequential(*layers))
            else:
                layers = []
                if use_bn:
                    layers.append(nn.BatchNorm1d(projected_dim))
                layers.append(nn.GELU())
                layers.append(nn.Linear(projected_dim, len(dict_attribute[key]) + 1))  # 1 for no present
                self.classifier_list.append(nn.Sequential(*layers))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        probability = []
        if self.shared_projected_layer is None:
            for classifier in self.classifier_list:
                logit = classifier(x)
                probability.append(self._softmax(logit))
        else:
            x = self.shared_projected_layer(x)
            for classifier in self.classifier_list:
                logit = classifier(x)
                probability.append(self._softmax(logit))

        return probability


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

