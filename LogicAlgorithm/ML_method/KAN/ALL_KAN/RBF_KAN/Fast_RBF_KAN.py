#!/usr/bin/env python3.8.10
# -*- coding: utf-8 -*-
"""
function description: 此文件是Github上ZiyaoLi于202405日版本的RBF-KAN项目
Used RBF(Radial Basis Functions) to approximate the B-spline basis,
which is the bottleneck of KAN and efficient KAN
用RBF径向基函数逼近B样条基（含3种）
author: sidhu2690
contact: https://github.com/sidhu2690/RBF-KAN
IDE: PyCharm Community Edition 2020.2.5
time: 2024/6/21 16:09
version: V1.0
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


class RBFLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_min=-2., grid_max=2., num_grids=8, spline_weight_init_scale=0.1):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.grid = nn.Parameter(torch.linspace(grid_min, grid_max, num_grids), requires_grad=False)
        self.spline_weight = nn.Parameter(torch.randn(in_features * num_grids, out_features) * spline_weight_init_scale)

        """
        description: 新增alpha，方便后面计算RBF
        author: TangKan
        contact: 785455964@qq.com
        IDE: PyCharm Community Edition 2021.2.3
        time: 2024/6/21 16:35
        version: V1.1
        """
        self.alpha = (self.grid_max - self.grid_min) / (self.num_grids - 1)

    """
    description: 修改forward，添加至三种RBF形式
    author: TangKan
    contact: 785455964@qq.com
    IDE: PyCharm Community Edition 2021.2.3
    time: 2024/6/21 16:35
    version: V1.1
    """
    # def forward(self, x):
    #     x = x.unsqueeze(-1)
    #     basis = torch.exp(-((x - self.grid) / ((self.grid_max - self.grid_min) / (self.num_grids - 1))) ** 2)
    #     return basis.view(basis.size(0), -1).matmul(self.spline_weight)
    def forward(self, x, rbf_mode='gaussian_rbf'):
        x = x.unsqueeze(-1)
        distances = x - self.grid

        if rbf_mode == 'gaussian_rbf':
            basis = torch.exp(-(distances / self.alpha) ** 2)
        elif rbf_mode == 'multiquadratic_rbf':
            basis = (1 + (self.alpha * distances) ** 2) ** 0.5
        elif rbf_mode == 'thin_plate_spline_rbf':
            basis = distances ** 2 * torch.log(self.alpha * distances)
        elif rbf_mode == 'inverse_quadric':
            basis = 1 / (1 + (self.alpha * distances) ** 2)
        else:
            basis = torch.exp(-(distances / self.alpha) ** 2)

        """
        description: 改变前向传播中x的维度，否则在某些运算场景会报错
        author: TangKan
        contact: 785455964@qq.com
        IDE: PyCharm Community Edition 2021.2.3
        time: 2024/6/21 15:35
        version: V1.1
        """
        output = basis.view(*x.shape[:-2], -1).matmul(self.spline_weight)

        return output


class RBFKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, grid_min=-2., grid_max=2., num_grids=8, use_base_update=True,
                 base_activation=nn.SiLU(), spline_weight_init_scale=0.1, rbf_mode='gaussian_rbf'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_base_update = use_base_update
        self.base_activation = base_activation
        self.spline_weight_init_scale = spline_weight_init_scale
        self.rbf_linear = RBFLinear(input_dim, output_dim, grid_min, grid_max, num_grids, spline_weight_init_scale)
        self.base_linear = nn.Linear(input_dim, output_dim) if use_base_update else None

        """
        description: 修改forward，添加至三种RBF形式
        author: TangKan
        contact: 785455964@qq.com
        IDE: PyCharm Community Edition 2021.2.3
        time: 2024/6/21 16:35
        version: V1.1
        """
        self.rbf_mode = rbf_mode

    """
    description: 修改forward，添加至三种RBF形式
    author: TangKan
    contact: 785455964@qq.com
    IDE: PyCharm Community Edition 2021.2.3
    time: 2024/6/21 16:35
    version: V1.1
    """
    # def forward(self, x):
    #     ret = self.rbf_linear(x)
    #     if self.use_base_update:
    #         base = self.base_linear(self.base_activation(x))
    #         ret = ret + base
    #     return ret
    def forward(self, x):
        ret = self.rbf_linear(x, rbf_mode=self.rbf_mode)
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret


class RBFKAN(nn.Module):
    def __init__(self, layers_hidden, grid_min=-2., grid_max=2., num_grids=8, use_base_update=True,
                 base_activation=nn.SiLU(), spline_weight_init_scale=0.1, rbf_mode='gaussian_rbf'):
        super().__init__()

        """
        description: 修改forward，添加至三种RBF形式
        author: TangKan
        contact: 785455964@qq.com
        IDE: PyCharm Community Edition 2021.2.3
        time: 2024/6/21 16:35
        version: V1.1
        """
        self.rbf_mode = rbf_mode

        self.layers = nn.ModuleList(
            [RBFKANLayer(in_dim, out_dim, grid_min, grid_max, num_grids, use_base_update,
                         base_activation, spline_weight_init_scale, self.rbf_mode)
             for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

