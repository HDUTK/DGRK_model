#!/usr/bin/env python3.6.8
# -*- coding: utf-8 -*-
"""
function description: 此文件用于画图时的配置
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2021.2.3
time: 2022/3/26 17:15
version: V1.0
"""

my_plot_config = dict()

# 画图时的颜色
my_plot_config['color'] = {
    '黑': 'k',
    '白': 'w',
    '红': 'r',
    '黄': 'y',
    '绿': 'g',
    '蓝': 'b',
    '品红': 'm',
    '青': 'c'
}

my_plot_config_list = ['gold', 'darkkhaki', 'y', 'darkolivegreen', 'forestgreen',
                       'lightseagreen', 'cyan', 'deepskyblue',
                       'steelblue', 'royalblue', 'b', 'darkviolet', 'violet', 'pink',
                       'k', 'grey', 'rosybrown', 'r', 'saddlebrown', 'peru', 'orange']

# 画图时的字符及坐标等配置
my_plot_config['figure_config'] = {
    'figure_size': (9, 9),  # 图片大小
    'font_size': 20,  # 字体大小
    'label_size': 12,  # 坐标轴大小
    'grid_flag': False,  # 是否显示网格
    'rotation': 0,  # x轴字体的旋转角度
    'legend_font_size': 20,  # 注解的字体大小
    'title_size': 25,  # 标题字体大小
    'boxplot_title_size': 20,  # 箱线图标题字体大小
    'boxplot_font_scale': 1,  # 箱线图字体大小放大倍数
    'boxplot_labelsize': 15,  # 箱线图字体大小
    'histogram_xlabelsize': 10,  # 直方图x轴标签大小
    'histogram_ylabelsize': 10,  # 直方图y轴标签大小
    'histogram_titlesize': 15  # 直方图标题大小
}


