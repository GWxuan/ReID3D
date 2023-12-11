import torch
import math
import numpy as np
from torch import nn
from torch.nn import functional as F

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
        device = x.device

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

        idx = idx + idx_base

        idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature, idx

def get_graph_feature_only(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx0 = knn(x, k=k)   # (batch_size, num_points, k)
        device = x.device

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

        idx = idx0 + idx_base

        idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = feature.permute(0, 3, 1, 2).contiguous()

    return feature, idx0

class TSE(nn.Module):
    def __init__(self, branch, inputs, outputs):
        super(TSE, self).__init__()
        self.branch = branch
        self.block_size = 3
        self.in_channels = inputs
        self.out_channels = outputs
        self.brance_channels = inputs
        self.k = 10

        self.conv_reduce = nn.Conv2d(self.brance_channels, self.in_channels, 
                kernel_size=1, stride=1, padding=0, bias=False)

        self.conv_erase = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels,
                    kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(self.in_channels)
        )
        for m in [self.conv_reduce]:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        for m in self.conv_erase:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0)
                m.bias.data.zero_()


    def corre_binar(self, x1, x2):
        b, c, n = x2.size()
        x1 = self.conv_reduce(x1.unsqueeze(-1)) # 256 -> 512
        x1 = x1.view(b, -1) # [b, 512]

        x2 = x2.view(b, c, -1) # [b, 512, n]
        f = torch.matmul(x1.view(b, 1, c), x2) # [b, 1, n]
        f = f / np.sqrt(c)
        
        fk, idx = get_graph_feature_only(x2, k=8) #  (b, 512, n, k)
        fk = fk.view(b, c, -1)                   #  (b, 512, n*k)
        fk = torch.matmul(x1.view(b, 1, c), fk)  #  (b, 1, n*k)
        fk = fk / np.sqrt(c)
        fk = fk.view(b, 1, n, 8)    
        fk = torch.sum(fk, dim=-1)              #  (b, 1, n) 
        fk = f + fk
        
        f = f.view(b, 1, n)

        bs, t, n = f.size()

        index = torch.argmax(fk.view(bs * t, n), dim=1) 
        masks = torch.zeros(bs * t, n)
        masks = masks.cuda()
        index_b = torch.arange(0, bs * t, dtype=torch.long)
        masks[index_b, index] = 1
        index_k = idx[index_b, index]
        for i in range(bs * t):
            masks[i, index_k[i]] = 1
        
        masks = 1 - masks.view(bs, t, n)
        return masks, f


    def erase_feature(self, x, masks, soft_masks):
        b, c, n = x.size()
        soft_masks = soft_masks - (1 - masks) * 1e8
        soft_masks = F.softmax(soft_masks.view(b, n) , 1)

        inputs = x * masks.unsqueeze(1)   # (b, c, n)  512
        res = torch.bmm(x.view(b, c, n), soft_masks.unsqueeze(-1)) 
        outputs = inputs.unsqueeze(-1) + self.conv_erase(res.unsqueeze(-1))  # (b, c, 1, 1)
        return outputs.view(b, c, n)


    def forward(self, x):
        b, t, c, n = x.size()
        m = torch.ones(b*t, n)
        m = m.cuda()
        x1 = torch.cat((x[:, 1:], x[:, [-1]]), dim=1).view(b*t,c,n)
        x0 = self.erase_feature(x.view(b*t,c,n), m, m) # [b, c, h, w]
        x0, _ = get_graph_feature(x0, k=self.k)
        
        x0 = self.branch[0](x0)
        x0 = x0.max(dim=-1, keepdim=False)[0]
        x0_ = x0.max(dim=-1, keepdim=True)[0]
        x0 = torch.cat((F.adaptive_max_pool1d(x0, 1).view(x0.size()[0], -1), F.adaptive_avg_pool1d(x0, 1).view(x0.size()[0], -1)), 1)
        masks, soft_masks = self.corre_binar(x0_.detach(), x1)
        masks, soft_masks = masks[:, 0], soft_masks[:, 0] 

        x1 = self.erase_feature(x1.view(b*t,c,n), masks, soft_masks)
        x1, idx = get_graph_feature(x1, k=self.k)
        x1 = self.branch[1](x1)
        x1 = x1.max(dim=-1, keepdim=False)[0]  # [b, c, n]

        x1 = torch.cat((F.adaptive_max_pool1d(x1, 1).view(x1.size()[0], -1), F.adaptive_avg_pool1d(x1, 1).view(x1.size()[0], -1)), 1)

        return x0 + x1
