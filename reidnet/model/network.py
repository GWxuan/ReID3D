import torch
import torch.nn as nn
import torch.nn.functional as F

import model
from model.gnn_tse import Net
from model.fusion_trans import Fusion
import parser
args = parser.parse_args()
import numpy as np
import os

class reid3d(nn.Module):
    def __init__(self, feat_dim=1024, num_class=512, stride=1):
        super(reid3d, self).__init__()
        self.features_pc = Net(k=10, emb_dims=512, output_channels=feat_dim)
        self.bn_s = nn.BatchNorm1d(feat_dim)
        self.bn_s.bias.requires_grad_(False)
        self.bn_s.apply(model.weights_init_kaiming)
        self.bn_t = nn.BatchNorm1d(feat_dim)
        self.bn_t.bias.requires_grad_(False)
        self.bn_t.apply(model.weights_init_kaiming)

        self.cls_s_ = nn.Linear(feat_dim, num_class,  bias=False)
        self.cls_s_.apply(model.weights_init_classifier)
        self.cls_t_ = nn.Linear(feat_dim, num_class,  bias=False)
        self.cls_t_.apply(model.weights_init_classifier)

        self.fusion = Fusion(d_model=feat_dim, num_heads=4, nlayers=2) #network


    def forward(self, x):

        x = x.float()
        x = x.permute(0, 1, 3, 2)
        B, S, C, N = x.size()
        x = x.reshape((B*S, )+x.shape[2:])
         
        key_t = self.features_pc(x)
        val_s_bn = self.bn_s(key_t)
        val_t = self.fusion(key_t)
        val_t_bn = self.bn_t(val_t) 

        if self.training:
            val_s_cls = self.cls_s_(val_s_bn)
            val_t_cls = self.cls_t_(val_t_bn)
            return {'val_t':val_t, 'val_t_cls':val_t_cls, 'val_s':key_t, 'val_s_cls':val_s_cls}
        else:
            return {'val_bn':val_t_bn, 'val_t': val_t}
        
        
