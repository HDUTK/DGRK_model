#!/usr/bin/env python3.8.10
# -*- coding: utf-8 -*-
"""
function description: 此文件用于灰色关联性分析
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2020.2.5
time: 2024/5/17 14:41
version: V1.0
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from common.plot_func import my_plot_func


def GRA_ONE(data, m=0, p=0.5):
    """
    求参考列（默认第一列）(影响因素)和其它所有列(影响因素)的灰色关联值
    :param data: 标准化后的数据，Dataframe格式
    :param m: 参考列数，默认第一列
    :param p: 灰色分辨系数，默认0.5
    :return: 参考列和其他列的灰色相关系数，pd.Series格式
    """
    # 参考数列
    std = data.iloc[:, m]
    # 比较数列
    ce = data.copy()

    n = ce.shape[0]
    m = ce.shape[1]

    # 与参考数列比较，相减
    gray = np.zeros([n, m])
    for i in range(m):
        for j in range(n):
            gray[j, i] = abs(ce.iloc[j, i] - std[j])

    # 取出矩阵中的最大值和最小值
    m_max = np.amax(gray)
    m_min = np.amin(gray)

    # 计算值
    gray = pd.DataFrame(gray).applymap(lambda x: (m_min + p * m_max) / (x + p * m_max))

    # 求均值，得到灰色关联值
    RT = gray.mean(axis=0)

    return pd.Series(RT)


def GRA(data, p=0.5, plot_flag=False):
    """
    调用GRA_ONE，可以求得所有因素之间的灰色关联值
    :param data: 标准化后的数据，Dataframe格式
    :param p: 灰色分辨系数，默认0.5
    :param plot_flag: 是否画图，默认否
    :return: 灰色关联矩阵
    """
    list_columns = np.arange(data.shape[1])
    df_local = pd.DataFrame(columns=list_columns)
    for i in tqdm(np.arange(data.shape[1])):
        df_local.iloc[:, i] = GRA_ONE(data, m=i, p=p)

    if plot_flag:
        my_plot_func.dataframe_to_heat_map(df_local, plot_flag=plot_flag)
    else:
        pass

    return df_local
