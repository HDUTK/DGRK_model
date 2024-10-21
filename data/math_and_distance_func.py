#!/usr/bin/env python3.6.8
# -*- coding: utf-8 -*-
"""
function description: 此文件用于数学函数 包括距离函数
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2021.2.3
time: 2022/3/24 23:52
version: V1.0
"""
import numpy as np


class my_math_func(object):
    """
    数学函数
    """
    def __init__(self):
        pass

    @staticmethod
    def my_sigmoid(x):
        """
        sigmoid函数 y=1.0/(1.0+exp(-x))
        :param x:输入
        :return:返回
        """
        z = 1.0 / (1.0 + np.exp((-1) * x))
        return z

    @staticmethod
    def my_sgn(x):
        """
        阶跃函数 y=sgn(x)
        :param x:输入
        :return: 输出
        """
        if x < 0:
            return -1
        elif x == 0:
            return 0
        else:
            return 1


class my_distance_func(object):
    """
    距离函数
    """
    def __init__(self):
        pass

    @staticmethod
    def manhattan_distance(x, y):
        """
        曼哈顿距离 L1范数
        :param x:输入x
        :param y:输入y
        :return:返回曼哈顿距离
        """
        distance = np.sum(np.abs(x - y))
        return distance

    @staticmethod
    def euclidean_distance(x, y):
        """
        欧氏距离 L2范数
        :param x:输入x
        :param y:输入y
        :return:返回欧式距离
        """
        distance = np.sqrt(np.sum((x - y)**2))
        return distance

    @staticmethod
    def minkowski_distance(x, y, p):
        """
        闵可夫斯基距离
        :param x:输入x
        :param y:输入y
        :param p:闵氏距离各种不同的距离参数
        :return:返回欧式距离
        """
        distance = np.sum(np.abs(x - y) ** p) ** (1 / p)
        return distance

    @staticmethod
    def chebyshev_distance(x, y):
        """
        切比雪夫距离
        :param x:输入x
        :param y:输入y
        :return:返回切比雪夫距离
        """
        distance = np.abs(x-y).max()
        return distance

    @staticmethod
    def cosine_similarity(x, y):
        """
        余弦相似度
        余弦夹角一般用来测量两个样本之间的相似性
        常用于图像特征向量之间的相似度比对
        :param x:输入x
        :param y:输入y
        :return:返回欧式距离
        """
        similarity = np.dot(x, y)/(np.sqrt(x, x)*np.sqrt(y, y))
        return similarity

