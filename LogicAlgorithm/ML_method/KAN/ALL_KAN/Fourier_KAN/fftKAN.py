#!/usr/bin/env python3.8.10
# -*- coding: utf-8 -*-
"""
function description: 此文件是Github上GistNoesis于202405版本的FourierKAN项目
Pytorch Layer for FourierKAN
It is a layer intended to be a substitution for Linear + non-linear activation
author: GistNoesis
contact: https://github.com/GistNoesis/FourierKAN
IDE: PyCharm Community Edition 2020.2.5
time: 2024/6/22 20:36
version: V1.0
"""

import torch as th
import numpy as np


# This is inspired by Kolmogorov-Arnold Networks but using 1d fourier coefficients instead of splines coefficients
# It should be easier to optimize as fourier are more dense than spline (global vs local)
# Once convergence is reached you can replace the 1d function with spline approximation for faster evaluation giving almost the same result
# The other advantage of using fourier over spline is that the function are periodic, and therefore more numerically bounded
# Avoiding the issues of going out of grid


class Fourier_KAN(th.nn.Module):
    """
    description: 创建Fourier KAN
    author: TangKan
    contact: 785455964@qq.com
    IDE: PyCharm Community Edition 2021.2.3
    time: 2024/6/22 21:35
    version: V1.1
    """
    def __init__(self, layer_list, gridsize=5, **kwargs):
        """
        创建Fourier KAN
        :param layer_list: KAN网络结构，[4, 4, 1]
        :param gridsize: gridsize，默认5
        :param kwargs: 其他参数
        """
        super(Fourier_KAN, self).__init__()
        self.Fourier_Layer = th.nn.ModuleList()
        for i in range(len(layer_list) - 1):
            self.Fourier_Layer.append(NaiveFourierKANLayer(layer_list[i], layer_list[i+1], gridsize=gridsize))

    def forward(self, x):
        for layer in self.Fourier_Layer:
            x = layer(x)
        return x


class NaiveFourierKANLayer(th.nn.Module):
    def __init__(self, inputdim, outdim, gridsize, addbias=True, smooth_initialization=False):
        super(NaiveFourierKANLayer, self).__init__()
        self.gridsize = gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim

        # With smooth_initialization, fourier coefficients are attenuated by the square of their frequency.
        # This makes KAN's scalar functions smooth at initialization.
        # Without smooth_initialization, high gridsizes will lead to high-frequency scalar functions,
        # with high derivatives and low correlation between similar inputs.
        grid_norm_factor = (th.arange(gridsize) + 1) ** 2 if smooth_initialization else np.sqrt(gridsize)

        # The normalization has been chosen so that if given inputs where each coordinate is of unit variance,
        # then each coordinates of the output is of unit variance
        # independently of the various sizes
        self.fouriercoeffs = th.nn.Parameter(th.randn(2, outdim, inputdim, gridsize) /
                                             (np.sqrt(inputdim) * grid_norm_factor))
        if (self.addbias):
            self.bias = th.nn.Parameter(th.zeros(1, outdim))

    # x.shape ( ... , indim )
    # out.shape ( ..., outdim)
    def forward(self, x):
        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        x = th.reshape(x, (-1, self.inputdim))
        # Starting at 1 because constant terms are in the bias
        k = th.reshape(th.arange(1, self.gridsize + 1, device=x.device), (1, 1, 1, self.gridsize))
        xrshp = th.reshape(x, (x.shape[0], 1, x.shape[1], 1))
        # This should be fused to avoid materializing memory
        c = th.cos(k * xrshp)
        s = th.sin(k * xrshp)
        # We compute the interpolation of the various functions defined by their fourier coefficient for each input coordinates and we sum them
        y = th.sum(c * self.fouriercoeffs[0:1], (-2, -1))
        y += th.sum(s * self.fouriercoeffs[1:2], (-2, -1))
        if (self.addbias):
            y += self.bias
        # End fuse
        '''
        #You can use einsum instead to reduce memory usage
        #It stills not as good as fully fused but it should help
        #einsum is usually slower though
        c = th.reshape(c,(1,x.shape[0],x.shape[1],self.gridsize))
        s = th.reshape(s,(1,x.shape[0],x.shape[1],self.gridsize))
        y2 = th.einsum( "dbik,djik->bj", th.concat([c,s],axis=0) ,self.fouriercoeffs )
        if( self.addbias):
            y2 += self.bias
        diff = th.sum((y2-y)**2)
        print("diff")
        print(diff) #should be ~0
        '''
        # y = th.reshape(y, outshape)
        y = y.view(*xshp[:-1], self.outdim)
        return y


def demo():
    bs = 10
    L = 3  # Not necessary just to show that additional dimensions are batched like Linear
    inputdim = 50
    hidden = 200
    outdim = 100
    gridsize = 300

    device = "cpu"  # "cuda"

    fkan1 = NaiveFourierKANLayer(inputdim, hidden, gridsize).to(device)
    fkan2 = NaiveFourierKANLayer(hidden, outdim, gridsize).to(device)

    x0 = th.randn(bs, inputdim).to(device)

    h = fkan1(x0)
    y = fkan2(h)
    print("x0.shape")
    print(x0.shape)
    print("h.shape")
    print(h.shape)
    print("th.mean( h )")
    print(th.mean(h))
    print("th.mean( th.var(h,-1) )")
    print(th.mean(th.var(h, -1)))

    print("y.shape")
    print(y.shape)
    print("th.mean( y)")
    print(th.mean(y))
    print("th.mean( th.var(y,-1) )")
    print(th.mean(th.var(y, -1)))

    print(" ")
    print(" ")
    print("Sequence example")
    print(" ")
    print(" ")
    xseq = th.randn(bs, L, inputdim).to(device)

    h = fkan1(xseq)
    y = fkan2(h)
    print("xseq.shape")
    print(xseq.shape)
    print("h.shape")
    print(h.shape)
    print("th.mean( h )")
    print(th.mean(h))
    print("th.mean( th.var(h,-1) )")
    print(th.mean(th.var(h, -1)))

    print("y.shape")
    print(y.shape)
    print("th.mean( y)")
    print(th.mean(y))
    print("th.mean( th.var(y,-1) )")
    print(th.mean(th.var(y, -1)))


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


    class FourierKAN(nn.Module):
        def __init__(self):
            super(FourierKAN, self).__init__()
            self.Fourierkan1 = NaiveFourierKANLayer(1, 8, 8)
            self.Fourierkan2 = NaiveFourierKANLayer(8, 1, 8)

        def forward(self, x):
            x = self.Fourierkan1(x)
            x = self.Fourierkan2(x)
            return x


    # Generate sample data
    x_train = torch.linspace(0, 2, steps=500).unsqueeze(1)
    y_train = torch.tensor(target_function(x_train))

    # Instantiate models
    Fourier_model = FourierKAN()

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer_Fourier = torch.optim.Adam(Fourier_model.parameters(), lr=0.01)

    Fourier_losses = []

    # Train the models
    epochs = 2000
    for epoch in range(epochs):
        optimizer_Fourier.zero_grad()
        outputs_Fourier = Fourier_model(x_train)
        loss_Fourier = criterion(outputs_Fourier, y_train)
        loss_Fourier.backward()
        optimizer_Fourier.step()

        if epoch % 100 == 0:
            Fourier_losses.append(loss_Fourier.item())
            print(f'Epoch {epoch + 1}/{epochs}, FourierKAN Loss: {loss_Fourier.item():.4f}')
            for name, param in Fourier_model.named_parameters():
                if param.grad is not None:
                    print(f"Gradient for {name}: {param.grad.abs().mean().item()}")

    # Test the models
    x_test = torch.linspace(0, 2, steps=400).unsqueeze(1)
    y_pred_Fourier = Fourier_model(x_test).detach()
    plt.figure(figsize=(10, 5))
    plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original Data')
    plt.plot(x_test.numpy(), y_pred_Fourier.numpy(), 'b-', label='FourierKAN')
    plt.title('Comparison of FourierKAN and MLP Interpolations f(x)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.show()
