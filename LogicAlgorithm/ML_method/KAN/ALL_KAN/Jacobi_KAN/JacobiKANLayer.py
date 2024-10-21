#!/usr/bin/env python3.8.10
# -*- coding: utf-8 -*-
"""
function description: 此文件是Github上SpaceLearner于202405版本的JacobiKAN项目
Jacobi polynomials are orthogonal polynomials defined on the interval [-1, 1].
They are very good at approximating functions and can be calculated recursively.
author: SpaceLearner
contact: https://github.com/SpaceLearner/JacobiKAN
IDE: PyCharm Community Edition 2020.2.5
time: 2024/6/22 20:36
version: V1.0
"""

import torch
import torch.nn as nn
import numpy as np


class Jacobi_KAN(torch.nn.Module):
    """
    description: 创建Jacobi KAN
    author: TangKan
    contact: 785455964@qq.com
    IDE: PyCharm Community Edition 2021.2.3
    time: 2024/6/18 15:35
    version: V1.1
    """
    def __init__(self, layer_list, degree_list, with_LayerNorm_flag=False, **kwargs):
        """
        创建Jacobi KAN
        :param layer_list: KAN网络结构，[4, 4, 1]
        :param degree_list: 创建Jacobi多项式的degree list，[2, 3]
        degree为Jacobi polynomials的度数，即多项式中最高次幂的指数
        :param with_LayerNorm_flag: 是否在每层KAN加入归一化层防止tanh函数引起的梯度消失
        :param kwargs: 其他参数
        """
        super(Jacobi_KAN, self).__init__()

        self.JacobiKAN_Layer = nn.ModuleList()
        for i in range(len(degree_list)):
            self.JacobiKAN_Layer.append(JacobiKANLayer(layer_list[i], layer_list[i+1], degree_list[i]))
            if with_LayerNorm_flag:
                LayerNorm_normalized_shape = kwargs.get("LayerNorm_normalized_shape", layer_list[i+1])
                self.JacobiKAN_Layer.append(nn.LayerNorm(LayerNorm_normalized_shape))
            else:
                pass

    def forward(self, x):
        for layer in self.JacobiKAN_Layer:
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.JacobiKAN_Layer
        )


# This is inspired by Kolmogorov-Arnold Networks but using Jacobian polynomials instead of splines coefficients
class JacobiKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, a=1.0, b=1.0):
        super(JacobiKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.a = a
        self.b = b
        self.degree = degree

        self.jacobi_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))

        nn.init.normal_(self.jacobi_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

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

        x = torch.reshape(x, (-1, self.inputdim))  # shape = (batch_size, inputdim)

        """
        description: 使用归一化代替tanh函数将数据转化至[-1, 1]
        否则出现梯度消失问题——tanh在输入值较大或较小时会饱和，导致梯度几乎为零
        author: TangKan
        contact: 785455964@qq.com
        IDE: PyCharm Community Edition 2021.2.3
        time: 2024/7/5 12:16
        version: V1.1
        """
        # Since Jacobian polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        # x = torch.tanh(x)
        # x = 2 * (x - x.min()) / (x.max() - x.min()) - 1
        x = 2 * (x - x.min(dim=0, keepdim=True)[0]) / (
                x.max(dim=0, keepdim=True)[0] - x.min(dim=0, keepdim=True)[0]) - 1

        # Initialize Jacobian polynomial tensors
        jacobi = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device)
        if self.degree > 0:  ## degree = 0: jacobi[:, :, 0] = 1 (already initialized) ; degree = 1: jacobi[:, :, 1] = x ; d
            jacobi[:, :, 1] = ((self.a - self.b) + (self.a + self.b + 2) * x) / 2
        for i in range(2, self.degree + 1):
            theta_k = (2 * i + self.a + self.b) * (2 * i + self.a + self.b - 1) / (2 * i * (i + self.a + self.b))
            theta_k1 = (2 * i + self.a + self.b - 1) * (self.a * self.a - self.b * self.b) / (
                        2 * i * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
            theta_k2 = (i + self.a - 1) * (i + self.b - 1) * (2 * i + self.a + self.b) / (
                        i * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
            jacobi[:, :, i] = (theta_k * x + theta_k1) * jacobi[:, :, i - 1].clone() - theta_k2 * jacobi[:, :,
                                                                                                  i - 2].clone()  # 2 * x * jacobi[:, :, i - 1].clone() - jacobi[:, :, i - 2].clone()
        # Compute the Jacobian interpolation
        y = torch.einsum('bid,iod->bo', jacobi, self.jacobi_coeffs)  # shape = (batch_size, outdim)

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


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt

    # Define target function
    def target_function(x):
        y = np.zeros_like(x)
        mask1 = x < 0.5
        y[mask1] = np.sin(20 * np.pi * x[mask1]) + x[mask1] ** 2
        mask2 = (0.5 <= x) & (x < 1.5)
        y[mask2] = 0.5 * x[mask2] * np.exp(-x[mask2]) + np.abs(np.sin(5 * np.pi * x[mask2]))
        mask3 = x >= 1.5
        y[mask3] = np.log(x[mask3] - 1) / np.log(2) - np.cos(2 * np.pi * x[mask3])

        # add noise
        noise = np.random.normal(0, 0.2, y.shape)
        y += noise

        return y

    class JacobiKAN(nn.Module):
        def __init__(self):
            super(JacobiKAN, self).__init__()
            self.Jacobikan1 = JacobiKANLayer(1, 8, 8)
            self.Jacobikan2 = JacobiKANLayer(8, 1, 8)

        def forward(self, x):
            x = self.Jacobikan1(x)
            x = self.Jacobikan2(x)
            return x
    # Generate sample data
    x_train = torch.linspace(0, 2, steps=500).unsqueeze(1)
    y_train = torch.tensor(target_function(x_train))

    # Instantiate models
    Jacobi_model = JacobiKAN()

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer_Jacobi = torch.optim.Adam(Jacobi_model.parameters(), lr=0.01)

    Jacobi_losses = []

    # Train the models
    epochs = 2000
    for epoch in range(epochs):
        optimizer_Jacobi.zero_grad()
        outputs_Jacobi = Jacobi_model(x_train)
        loss_Jacobi = criterion(outputs_Jacobi, y_train)
        loss_Jacobi.backward()
        optimizer_Jacobi.step()

        if epoch % 100 == 0:
            Jacobi_losses.append(loss_Jacobi.item())
            print(f'Epoch {epoch + 1}/{epochs}, JacobiKAN Loss: {loss_Jacobi.item():.4f}')
            for name, param in Jacobi_model.named_parameters():
                if param.grad is not None:
                    print(f"Gradient for {name}: {param.grad.abs().mean().item()}")

    # Test the models
    x_test = torch.linspace(0, 2, steps=400).unsqueeze(1)
    y_pred_Jacobi = Jacobi_model(x_test).detach()
    plt.figure(figsize=(10, 5))
    plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original Data')
    plt.plot(x_test.numpy(), y_pred_Jacobi.numpy(), 'b-', label='JacobiKAN')
    plt.title('Comparison of JacobiKAN and MLP Interpolations f(x)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.show()
