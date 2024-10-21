'''

#!/usr/bin/env python3.8.10
# -*- coding: utf-8 -*-
"""
function description: 此文件是Github上ZiyaoLi于202405日版本的Wav-KAN项目
author: zavareh1
contact: https://github.com/zavareh1/Wav-KAN
IDE: PyCharm Community Edition 2020.2.5
time: 2024/6/21 16:09
version: V1.0
"""

This is a sample code for the simulations of the paper:
Bozorgasl, Zavareh and Chen, Hao, Wav-KAN: Wavelet Kolmogorov-Arnold Networks (May, 2024)

https://arxiv.org/abs/2405.12832
and also available at:
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4835325
We used efficient KAN notation and some part of the code:https://github.com/Blealtan/efficient-kan

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import math


class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, wavelet_type='mexican_hat'):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type

        # Parameters for wavelet transformation
        self.scale = nn.Parameter(torch.ones(out_features, in_features))
        self.translation = nn.Parameter(torch.zeros(out_features, in_features))

        # Linear weights for combining outputs
        # self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight1 = nn.Parameter(torch.Tensor(out_features,
                                                 in_features))  # not used; you may like to use it for wieghting base activation and adding it like Spl-KAN paper
        self.wavelet_weights = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.kaiming_uniform_(self.wavelet_weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))

        # Base activation function #not used for this experiment
        self.base_activation = nn.SiLU()

        # Batch normalization
        self.bn = nn.BatchNorm1d(out_features)

    def wavelet_transform(self, x):
        if x.dim() == 2:
            x_expanded = x.unsqueeze(1)
        else:
            x_expanded = x

        translation_expanded = self.translation.unsqueeze(0).expand(x.size(0), -1, -1)
        scale_expanded = self.scale.unsqueeze(0).expand(x.size(0), -1, -1)
        x_scaled = (x_expanded - translation_expanded) / scale_expanded

        # Implementation of different wavelet types
        if self.wavelet_type == 'mexican_hat':
            term1 = ((x_scaled ** 2) - 1)
            term2 = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = (2 / (math.sqrt(3) * math.pi ** 0.25)) * term1 * term2
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'morlet':
            omega0 = 5.0  # Central frequency
            real = torch.cos(omega0 * x_scaled)
            envelope = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = envelope * real
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)

        elif self.wavelet_type == 'dog':
            # Implementing Derivative of Gaussian Wavelet 
            dog = -x_scaled * torch.exp(-0.5 * x_scaled ** 2)
            wavelet = dog
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'meyer':
            # Implement Meyer Wavelet here
            # Constants for the Meyer wavelet transition boundaries
            v = torch.abs(x_scaled)
            pi = math.pi

            def meyer_aux(v):
                return torch.where(v <= 1 / 2, torch.ones_like(v),
                                   torch.where(v >= 1, torch.zeros_like(v), torch.cos(pi / 2 * nu(2 * v - 1))))

            def nu(t):
                return t ** 4 * (35 - 84 * t + 70 * t ** 2 - 20 * t ** 3)

            # Meyer wavelet calculation using the auxiliary function
            wavelet = torch.sin(pi * v) * meyer_aux(v)
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'shannon':
            # Windowing the sinc function to limit its support
            pi = math.pi
            sinc = torch.sinc(x_scaled / pi)  # sinc(x) = sin(pi*x) / (pi*x)

            # Applying a Hamming window to limit the infinite support of the sinc function
            window = torch.hamming_window(x_scaled.size(-1), periodic=False, dtype=x_scaled.dtype,
                                          device=x_scaled.device)
            # Shannon wavelet is the product of the sinc function and the window
            wavelet = sinc * window
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
            # You can try many more wavelet types ...
        else:
            raise ValueError("Unsupported wavelet type")

        return wavelet_output

    def forward(self, x):
        """
        description: 添加shape
        author: TangKan
        contact: 785455964@qq.com
        IDE: PyCharm Community Edition 2021.2.3
        time: 2024/6/22 21:30
        version: V1.1
        """
        original_shape = x.shape

        wavelet_output = self.wavelet_transform(x)
        # You may like test the cases like Spl-KAN
        # wav_output = F.linear(wavelet_output, self.weight)
        # base_output = F.linear(self.base_activation(x), self.weight1)

        base_output = F.linear(x, self.weight1)
        combined_output = wavelet_output  # + base_output

        """
        description: 根据shape输出y
        author: TangKan
        contact: 785455964@qq.com
        IDE: PyCharm Community Edition 2021.2.3
        time: 2024/6/22 21:16
        version: V1.1
        """
        # Apply batch normalization
        return self.bn(combined_output)
        # return self.bn(combined_output).view(*original_shape[:-1], self.out_features)


class KAN(nn.Module):
    def __init__(self, layers_hidden, wavelet_type='mexican_hat'):
        super(KAN, self).__init__()
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.layers.append(KANLinear(in_features, out_features, wavelet_type))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    # demo()
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


    class WavKAN(nn.Module):
        def __init__(self):
            super(WavKAN, self).__init__()
            # mexican_hat/morlet/dog/meyer/shannon
            self.Wavkan1 = KANLinear(1, 8, wavelet_type='morlet')
            self.Wavkan2 = KANLinear(8, 1, wavelet_type='morlet')

        def forward(self, x):
            x = self.Wavkan1(x)
            x = self.Wavkan2(x)
            return x


    # Generate sample data
    x_train = torch.linspace(0, 2, steps=500).unsqueeze(1)
    y_train = torch.tensor(target_function(x_train))

    # Instantiate models
    Wav_model = WavKAN()

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer_Wav = torch.optim.Adam(Wav_model.parameters(), lr=0.01)

    Wav_losses = []

    # Train the models
    epochs = 2000
    for epoch in range(epochs):
        optimizer_Wav.zero_grad()
        outputs_Wav = Wav_model(x_train)
        loss_Wav = criterion(outputs_Wav, y_train)
        loss_Wav.backward()
        optimizer_Wav.step()

        if epoch % 100 == 0:
            Wav_losses.append(loss_Wav.item())
            print(f'Epoch {epoch + 1}/{epochs}, WavKAN Loss: {loss_Wav.item():.4f}')
            for name, param in Wav_model.named_parameters():
                if param.grad is not None:
                    print(f"Gradient for {name}: {param.grad.abs().mean().item()}")

    # Test the models
    x_test = torch.linspace(0, 2, steps=400).unsqueeze(1)
    y_pred_Wav = Wav_model(x_test).detach()
    plt.figure(figsize=(10, 5))
    plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original Data')
    plt.plot(x_test.numpy(), y_pred_Wav.numpy(), 'b-', label='WavKAN')
    plt.title('Comparison of WavKAN and MLP Interpolations f(x)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.show()
