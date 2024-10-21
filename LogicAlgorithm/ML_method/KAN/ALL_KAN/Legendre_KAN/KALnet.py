#!/usr/bin/env python3.8.10
# -*- coding: utf-8 -*-
"""
function description: 此文件是Github上 1ssb于202405日版本的torchkan项目
KAC-Net: Utilizing Legendre Polynomials
author: 1ssb
contact: https://github.com/1ssb/torchkan
IDE: PyCharm Community Edition 2020.2.5
time: 2024/6/22 19:40
version: V1.0
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache


class KAL_Net(nn.Module):  # Kolmogorov Arnold Legendre Network (KAL-Net)
    def __init__(self, layers_hidden, polynomial_order=3, base_activation=nn.SiLU):
        super(KAL_Net, self).__init__()  # Initialize the parent nn.Module class

        # layers_hidden: A list of integers specifying the number of neurons in each layer
        self.layers_hidden = layers_hidden
        # polynomial_order: Order up to which Legendre polynomials are calculated
        self.polynomial_order = polynomial_order
        # base_activation: Activation function used after each layer's computation
        self.base_activation = base_activation()

        # ParameterList for the base weights of each layer
        self.base_weights = nn.ParameterList()
        # ParameterList for the polynomial weights for Legendre expansion
        self.poly_weights = nn.ParameterList()
        # ModuleList for layer normalization for each layer's output
        self.layer_norms = nn.ModuleList()

        # Initialize network parameters
        for i, (in_features, out_features) in enumerate(zip(layers_hidden, layers_hidden[1:])):
            # Base weight for linear transformation in each layer
            self.base_weights.append(nn.Parameter(torch.randn(out_features, in_features)))
            # Polynomial weight for handling Legendre polynomial expansions
            self.poly_weights.append(nn.Parameter(torch.randn(out_features, in_features * (polynomial_order + 1))))
            # Layer normalization to stabilize learning and outputs
            self.layer_norms.append(nn.LayerNorm(out_features))

        # Initialize weights using Kaiming uniform distribution for better training start
        for weight in self.base_weights:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')
        for weight in self.poly_weights:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')

    @lru_cache(maxsize=128)  # Cache to avoid recomputation of Legendre polynomials
    def compute_legendre_polynomials(self, x, order):
        # Base case polynomials P0 and P1
        P0 = x.new_ones(x.shape)  # P0 = 1 for all x
        if order == 0:
            return P0.unsqueeze(-1)
        P1 = x  # P1 = x
        legendre_polys = [P0, P1]

        # Compute higher order polynomials using recurrence
        for n in range(1, order):
            Pn = ((2.0 * n + 1.0) * x * legendre_polys[-1] - n * legendre_polys[-2]) / (n + 1.0)
            legendre_polys.append(Pn)

        return torch.stack(legendre_polys, dim=-1)

    def forward(self, x):
        # Ensure x is on the right device from the start, matching the model parameters
        x = x.to(self.base_weights[0].device)

        for i, (base_weight, poly_weight, layer_norm) in enumerate(
                zip(self.base_weights, self.poly_weights, self.layer_norms)):
            # Apply base activation to input and then linear transform with base weights
            base_output = F.linear(self.base_activation(x), base_weight)

            base_output = 2 * (base_output - base_output.min()) / (base_output.max() - base_output.min()) - 1

            # Normalize x to the range [-1, 1] for stable Legendre polynomial computation
            x_normalized = 2 * (x - x.min()) / (x.max() - x.min()) - 1
            # Compute Legendre polynomials for the normalized x
            legendre_basis = self.compute_legendre_polynomials(x_normalized, self.polynomial_order)

            """
            description: 改变前向传播中x的维度，否则在某些运算场景会报错
            author: TangKan
            contact: 785455964@qq.com
            IDE: PyCharm Community Edition 2021.2.3
            time: 2024/6/22 19:45
            version: V1.1
            """
            # Reshape legendre_basis to match the expected input dimensions for linear transformation
            # legendre_basis = legendre_basis.view(x.size(0), -1)
            legendre_basis = legendre_basis.view(*legendre_basis.shape[:-2], -1)

            # Compute polynomial output using polynomial weights
            poly_output = F.linear(legendre_basis, poly_weight)
            # Combine base and polynomial outputs, normalize, and activate
            x = self.base_activation(layer_norm(base_output + poly_output))

        return x


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

    class legendreKAN(nn.Module):
        def __init__(self):
            super(legendreKAN, self).__init__()
            self.legendrekan = KAL_Net([1, 8, 1], polynomial_order=8)

        def forward(self, x):
            x = self.legendrekan(x)
            return x
    # Generate sample data
    x_train = torch.linspace(0, 2, steps=500).unsqueeze(1)
    y_train = torch.tensor(target_function(x_train))

    # Instantiate models
    legendre_model = legendreKAN()

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer_legendre = torch.optim.Adam(legendre_model.parameters(), lr=0.01)

    legendre_losses = []

    # Train the models
    epochs = 2000
    for epoch in range(epochs):
        optimizer_legendre.zero_grad()
        outputs_legendre = legendre_model(x_train)
        loss_legendre = criterion(outputs_legendre, y_train)
        loss_legendre.backward()
        optimizer_legendre.step()

        if epoch % 100 == 0:
            legendre_losses.append(loss_legendre.item())
            print(f'Epoch {epoch + 1}/{epochs}, JacobiKAN Loss: {loss_legendre.item():.4f}')
            for name, param in legendre_model.named_parameters():
                if param.grad is not None:
                    print(f"Gradient for {name}: {param.grad.abs().mean().item()}")

    # Test the models
    x_test = torch.linspace(0, 2, steps=400).unsqueeze(1)
    y_pred_legendre = legendre_model(x_test).detach()
    plt.figure(figsize=(10, 5))
    plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original Data')
    plt.plot(x_test.numpy(), y_pred_legendre.numpy(), 'b-', label='legendreKAN')
    plt.title('Comparison of legendreKAN and MLP Interpolations f(x)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.show()
