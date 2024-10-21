#!/usr/bin/env python3.8.10
# -*- coding: utf-8 -*-
"""
function description: 此文件是Github上SynodicMonth于202405版本的ChebyKAN项目下的ChebyKANLayer.py
B-splines are poor in performance and not very intuitive to use.
I'm trying to replace B-splines with Chebyshev polynomials
B样条性能较差，使用起来不是很直观。我正在尝试用切比雪夫多项式替换B样条曲线
author: SynodicMonth
contact: https://github.com/SynodicMonth/ChebyKAN
IDE: PyCharm Community Edition 2020.2.5
time: 2024/6/1 14:09
version: V1.0
"""

import torch
import torch.nn as nn

"""
Example
Construct a ChebyKAN for MNIST:
class MNISTChebyKAN(nn.Module):
    def __init__(self):
        super(MNISTChebyKAN, self).__init__()
        self.chebykan1 = ChebyKANLayer(28*28, 32, 4)
        self.ln1 = nn.LayerNorm(32) # To avoid gradient vanishing caused by tanh
        self.chebykan2 = ChebyKANLayer(32, 16, 4)
        self.ln2 = nn.LayerNorm(16)
        self.chebykan3 = ChebyKANLayer(16, 10, 4)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the images
        x = self.chebykan1(x)
        x = self.ln1(x)
        x = self.chebykan2(x)
        x = self.ln2(x)
        x = self.chebykan3(x)
        return x
"""


class Chebyshev_KAN(torch.nn.Module):
    """
    description: 创建Chebyshev KAN
    author: TangKan
    contact: 785455964@qq.com
    IDE: PyCharm Community Edition 2021.2.3
    time: 2024/6/18 15:35
    version: V1.1
    """
    def __init__(self, layer_list, degree_list, with_LayerNorm_flag=False, **kwargs):
        """
        创建Chebyshev KAN
        :param layer_list: KAN网络结构，[4, 4, 1]
        :param degree_list: 创建Chebyshev多项式的degree list，[2, 3]
        degree为Chebyshev polynomials的度数，即多项式中最高次幂的指数
        :param with_LayerNorm_flag: 是否在每层KAN加入归一化层防止tanh函数引起的梯度消失
        :param kwargs: 其他参数
        """
        super(Chebyshev_KAN, self).__init__()

        self.ChebyKAN_Layer = nn.ModuleList()
        for i in range(len(degree_list)):
            self.ChebyKAN_Layer.append(ChebyKANLayer(layer_list[i], layer_list[i+1], degree_list[i]))
            if with_LayerNorm_flag:
                LayerNorm_normalized_shape = kwargs.get("LayerNorm_normalized_shape", layer_list[i+1])
                self.ChebyKAN_Layer.append(nn.LayerNorm(LayerNorm_normalized_shape))
            else:
                pass

    def forward(self, x):
        for layer in self.ChebyKAN_Layer:
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.ChebyKAN_Layer
        )


# This is inspired by Kolmogorov-Arnold Networks but using Chebyshev polynomials instead of splines coefficients
class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree  # degree为Chebyshev polynomials的度数，即多项式中最高次幂的指数

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

        # # 将base_weight和spline_weight加入到可训练中的参数中
        # self.base_weight = torch.nn.Parameter(torch.Tensor(output_dim, input_dim))
        # self.base_activation = nn.Identity()

    def forward(self, x):
        """
        description: 添加shape
        author: TangKan
        contact: 785455964@qq.com
        IDE: PyCharm Community Edition 2021.2.3
        time: 2024/6/18 21:30
        version: V1.1
        """
        original_shape = x.shape

        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)

        # import torch.nn.functional as F
        # a = self.base_activation(x)
        # b = self.base_weight
        # base_output = F.linear(self.base_activation(x), self.base_weight)  # (batch, output_size)
        # # (batch, input_size) + (output_size, input_size) / (batch, output_size)

        # View and repeat input degree + 1 times
        x = x.view((-1, self.inputdim, 1)).expand(
            -1, -1, self.degree + 1
        )  # shape = (batch_size, inputdim, self.degree + 1)
        # Apply acos
        x = x.acos()
        # Multiply by arange [0 .. degree]
        # x *= self.arange
        x = x * self.arange
        # Apply cos
        x = x.cos()
        # Compute the Chebyshev interpolation
        y = torch.einsum(
            "bid,iod->bo", x, self.cheby_coeffs
        )  # shape = (batch_size, outdim)

        # spline_output = F.linear(self.b_splines(x).view(x.size(0), -1),
        #                          self.scaled_spline_weight.view(self.out_features, -1), )  # (batch, output_size)
        # output = base_output + spline_output  # (batch, output_size)

        """
        description: 根据shape输出y
        author: TangKan
        contact: 785455964@qq.com
        IDE: PyCharm Community Edition 2021.2.3
        time: 2024/6/18 21:16
        version: V1.1
        """
        # y = y.view(-1, self.outdim)
        y = y.view(*original_shape[:-1], self.outdim)

        return y
