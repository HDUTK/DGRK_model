#!/usr/bin/env python3.8.10
# -*- coding: utf-8 -*-
"""
function description: 此文件是Github上 Boris-73-TA于202405日版本的OrthogPolyKANs项目
Kolmogorov-Arnold Networks (KAN) using orthogonal polynomials instead of B-splines.
! Based heavily on ChebyKAN implementation by SynodicMonth !
https://github.com/SynodicMonth/ChebyKAN
There are many orthogonal polynomials: https://en.wikipedia.org/wiki/Classical_orthogonal_polynomials
https://mathworld.wolfram.com/OrthogonalPolynomials.html
Polynomials: Legendre, generalized Laguerre, Chebyshev 2nd kind, Gegenbauer, Hermite, Fibonacci, Bessel, Lucas, and Jacobi
Working on:
More polynomials: Romanovski, Bernstein, Newton, Bernoulli, Euler, Zernike, Kravchuk, and Lucas Polynomials
Alternative to tanh for normalizing to [-1, 1] using MinMax...
Optimize by implementing polynomials with explicit formulas instead of recursive definitions...
NB: This is a very rough implementation, and there is a lot to improve.
author: Boris-73-TA
contact: https://github.com/1ssb/torchkan
IDE: PyCharm Community Edition 2020.2.5
time: 2024/6/22 19:40
version: V1.0
"""

import torch
import torch.nn as nn
from BesselKANLayer import BesselKANLayer
from Cheby2KANLayer import Cheby2KANLayer
from FibonacciKANLayer import FibonacciKANLayer
from LucasKANLayer import LucasKANLayer
from LegendreKANLayer import LegendreKANLayer
from HermiteKANLayer import HermiteKANLayer
from GegenbauerKANLayer import GegenbauerKANLayer
from JacobiKANLayer import JacobiKANLayer
from LaguerreKANLayer import LaguerreKANLayer

# Bessel, Cheby2, Fibonacci, Lucas, Legendre, Laguerre, Hermite, Gegenbauer,  Jacobi

class OrthogPolyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, poly_type = 'legendre', alpha=3, beta=3):
        super(OrthogPolyKANLayer, self).__init__()
        self.poly_type = poly_type.lower()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.alpha = alpha
        self.beta = beta

        # Initialize the appropriate layer based on the polynomial type
        if self.poly_type == 'jacobi':
            self.layer = JacobiKANLayer(input_dim, output_dim, degree, alpha, beta)
        elif self.poly_type == 'gegenbauer':
            self.layer = GegenbauerKANLayer(input_dim, output_dim, degree, alpha)
        elif self.poly_type == 'hermite':
            self.layer = HermiteKANLayer(input_dim, output_dim, degree)
        elif self.poly_type == 'lucas':
            self.layer = LucasKANLayer(input_dim, output_dim, degree)
        elif self.poly_type == 'fibonacci':
            self.layer = FibonacciKANLayer(input_dim, output_dim, degree)
        elif self.poly_type == 'legendre':
            self.layer = LegendreKANLayer(input_dim, output_dim, degree)
        elif self.poly_type == 'bessel':
            self.layer = BesselKANLayer(input_dim, output_dim, degree)
        elif self.poly_type == 'cheby2':
            self.layer = Cheby2KANLayer(input_dim, output_dim, degree)
        elif self.poly_type == 'laguerre':
            self.layer = LaguerreKANLayer(input_dim, output_dim, degree, alpha)  
        else:
            raise ValueError(f"Unsupported polynomial type: {self.poly_type}, please select one of the following: Bessel, Cheby2, Fibonacci, Lucas, Legendre, Laguerre, Hermite, Gegenbauer,  Jacobi. ")

    def forward(self, x):
        return self.layer(x)
