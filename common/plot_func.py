#!/usr/bin/env python3.6.8
# -*- coding: utf-8 -*-
"""
function description: 此文件用于画图
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2021.2.3
time: 2022/3/26 0:54
version: V1.0
"""
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from config.plot_config import my_plot_config, my_plot_config_list
from common.common_func import dir_path_exist_or_makedir


class my_plot_func(object):
    """
    画图函数
    """

    def __init__(self):
        pass

    @staticmethod
    def time_curve(*curve_data_param, x_time=None, title_name=None, xlabelname=None, ylabelname=None,
                   show_flag=True, dpi=None, save_dir_path='', save_figure_name='', **kwargs):
        """
        画图，时间与输出曲线关系图
        :param curve_data_param: 曲线y轴数据、参数
        若有多个y、必须保证多个y的x是一致的
        字典形式为
        {'y': list/...,  # y数据
        'color_curve': 'b-',  # 默认蓝色实线
        'curvename': ''}  # 曲线图例
        :param x_time: 时间
        :param title_name: 图片标题名称
        :param xlabelname: x轴名称
        :param ylabelname: y轴名称
        :param show_flag: 是否展示，默认True即展示
        :param dpi: 保存图像的清晰度
        :param save_dir_path: 若和save_figure_name同时不为空，则保存至该路径下
        :param save_figure_name: 若和save_dir_path同时不为空，则保存至该路径下时使用该名字保存成图片格式
        :param kwargs: 其他参数
        :return:
        """

        # 防止内存不够，pycharm一直在生成图形，虽然没用plt.show()但是还是占内存
        if not show_flag:
            matplotlib.use("Agg")
        else:
            pass

        # 设置默认样式风格
        plt.style.use('default')

        # 新建颜色列表，若有多个y会依次使用列表中颜色，7种颜色
        color_list = [my_plot_config['color']['蓝'], my_plot_config['color']['绿'],
                      my_plot_config['color']['红'], my_plot_config['color']['黄'],
                      my_plot_config['color']['黑'], my_plot_config['color']['青'],
                      my_plot_config['color']['品红']]

        # 若x轴为空，则以第一个y的计数代替
        if x_time is None:
            # m为列表长度
            m = len(curve_data_param[0]['y'])
            x_time = list(range(1, m + 1))
        else:
            pass

        # 如果kwargs中有图片的大小，则进行赋值，否则为设置的默认值
        figure_size = kwargs.get("figure_size", my_plot_config['figure_config']['figure_size'])
        # 图片大小
        plt.figure(figsize=figure_size)

        # 解决中文显示问题
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        # Times New Roman
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

        # 如果kwargs中有旋转轴标识角度，则旋转
        rotation = kwargs.get("rotation", my_plot_config['figure_config']['rotation'])
        # 旋转标签rotation度
        plt.xticks(rotation=rotation)

        # 根据参数数目画图
        for i in range(len(curve_data_param)):
            # 创建临时y数据参数，防止参数为空
            curve_data_param_temp = {'color_curve': str(color_list[i % 7]) + '-', 'curvename': ''}
            # 根据代入的字典更新参数
            curve_data_param_temp.update(curve_data_param[i])
            # 画图
            plt.plot(x_time, curve_data_param_temp['y'],
                     curve_data_param_temp['color_curve'],
                     label=curve_data_param_temp['curvename'])

        # 如果kwargs中有注解字体的大小，则进行赋值，否则默认20
        legend_font_size = kwargs.get("legend_font_size", my_plot_config['figure_config']['legend_font_size'])
        # 显示图例
        plt.legend(fontsize=legend_font_size, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

        # 如果kwargs中有各种字体的大小，则进行赋值，否则为设置的默认值
        xlabel_font_size = kwargs.get("xlabel_font_size", my_plot_config['figure_config']['font_size'])
        ylabel_font_size = kwargs.get("ylabel_font_size", my_plot_config['figure_config']['font_size'])
        title_font_size = kwargs.get("title_font_size", my_plot_config['figure_config']['title_size'])
        label_font_size = kwargs.get("label_font_size", my_plot_config['figure_config']['label_size'])
        # x轴、y轴及标题名称等
        plt.xlabel(xlabelname, fontsize=xlabel_font_size)
        plt.ylabel(ylabelname, fontsize=ylabel_font_size)
        plt.title(title_name, fontsize=title_font_size)
        # 坐标轴大小
        plt.tick_params(labelsize=label_font_size)

        # 是否显示网格
        if my_plot_config['figure_config']['grid_flag'] is True:
            plt.grid()
        else:
            pass

        # 调整布局
        plt.tight_layout()

        # 是否存储图片
        if (save_dir_path != '') and (save_figure_name != ''):
            dir_path_exist_or_makedir(save_dir_path)
            plt.savefig(save_dir_path + '/' + save_figure_name, dpi=dpi)
            print('+' * 20)
            print('Save picture to ' + save_dir_path + '/' + save_figure_name)
            print('+' * 20)
        else:
            pass

        # 是否展示
        if show_flag is True:
            plt.show()
        else:
            pass

        # # 清空当年画布
        # plt.clf()
        # 关闭所有图片
        plt.close('all')

        return

    @staticmethod
    def dataframe_to_time_curve(my_dataframe, title_name=None, xlabelname=None, ylabelname=None,
                                show_flag=True, dpi=None, save_dir_path='', save_figure_name='',
                                **kwargs):
        """
        使用Dataframe画图，每一列都要画在一张图里，时间与输出曲线关系图，第一列必须是时间['time']
        :param my_dataframe: 输入的Dataframe
        :param title_name: 图片标题名称
        :param xlabelname: x轴名称
        :param ylabelname: y轴名称
        :param show_flag: 是否展示，默认True即展示
        :param dpi: 保存图像的清晰度
        :param save_dir_path: 若和save_figure_name同时不为空，则保存至该路径下
        :param save_figure_name: 若和save_dir_path同时不为空，则保存至该路径下时使用该名字保存成图片格式
        :param kwargs: 其他参数
        :return:
        """

        # 防止内存不够，若pycharm一直在生成图形，虽然没用plt.show()但是还是占内存
        if not show_flag:
            matplotlib.use("Agg")
        else:
            pass

        # 设置默认样式风格
        plt.style.use('default')

        # 如果kwargs中有图片的大小，则进行赋值，否则为设置的默认值
        figure_size = kwargs.get("figure_size", my_plot_config['figure_config']['figure_size'])
        # 图片大小
        plt.figure(figsize=figure_size)

        # 解决中文显示问题
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        # Times New Roman
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

        # 如果kwargs中有旋转轴标识角度，则旋转
        rotation = kwargs.get("rotation", my_plot_config['figure_config']['rotation'])
        # 旋转标签rotation度
        plt.xticks(rotation=rotation)

        # 绘制多条曲线图，若曲线数超过10，为了不颜色重复，根据配置文件里的颜色my_plot_config_list自定义曲线颜色
        if len(my_dataframe.columns) > 10:
            i = 0
            for column in my_dataframe.columns[1:]:
                plt.plot(my_dataframe['time'], my_dataframe[column], label=column[:-7],
                         color=my_plot_config_list[i])
                i = i + 1
        else:
            for column in my_dataframe.columns[1:]:
                plt.plot(my_dataframe['time'], my_dataframe[column], label=column[:-7])

        # 如果kwargs中有注解字体的大小，则进行赋值，否则默认20
        legend_font_size = kwargs.get("legend_font_size", my_plot_config['figure_config']['legend_font_size'])
        # 显示图例
        plt.legend(fontsize=legend_font_size)

        # 如果kwargs中有各种字体的大小，则进行赋值，否则为设置的默认值
        xlabel_font_size = kwargs.get("xlabel_font_size", my_plot_config['figure_config']['font_size'])
        ylabel_font_size = kwargs.get("ylabel_font_size", my_plot_config['figure_config']['font_size'])
        title_font_size = kwargs.get("title_font_size", my_plot_config['figure_config']['title_size'])
        label_font_size = kwargs.get("label_font_size", my_plot_config['figure_config']['label_size'])
        # x轴、y轴及标题名称等
        plt.xlabel(xlabelname, fontsize=xlabel_font_size)
        plt.ylabel(ylabelname, fontsize=ylabel_font_size)
        plt.title(title_name, fontsize=title_font_size)
        # 坐标轴大小
        plt.tick_params(labelsize=label_font_size)

        # 是否显示网格
        if my_plot_config['figure_config']['grid_flag'] is True:
            plt.grid()
        else:
            pass

        # 是否存储图片
        if (save_dir_path != '') and (save_figure_name != ''):
            dir_path_exist_or_makedir(save_dir_path)
            plt.savefig(save_dir_path + '/' + save_figure_name, dpi=dpi)
            print('+' * 20)
            print('Save picture to ' + save_dir_path + '/' + save_figure_name)
            print('+' * 20)
        else:
            pass

        # 是否展示
        if show_flag is True:
            plt.show()
        else:
            pass

        # # 清空当年画布
        # plt.clf()
        # 关闭所有图片
        plt.close('all')

        return

    @staticmethod
    def plot_gradient_descend(cost_res, plot_param):
        """
        梯度下降 画图
        :param cost_res: 误差列表
        :param plot_param: 画图相关参数
        :return:
        """
        try:
            # 梯度下降损失函数 画图的参数初始化 默认使用蓝色实线
            curve_data_param_temp = {'y': cost_res,
                                     'color_curve': 'b-',
                                     'curvename': 'loss of SGD',
                                     'xlabelname': 'Number of Iterations',
                                     'ylabelname': 'Loss',
                                     'title_name': 'Loss in Training'}
            # 若传入的画图参数不为空 则更新
            if plot_param is not None:
                curve_data_param_temp.update(plot_param)
            else:
                pass
            curve_data_param = {'y': curve_data_param_temp['y'],
                                'color_curve': curve_data_param_temp['color_curve'],
                                'curvename': curve_data_param_temp['curvename']}
            # 画图
            my_plot_func.time_curve(curve_data_param,
                                    xlabelname=curve_data_param_temp['xlabelname'],
                                    ylabelname=curve_data_param_temp['ylabelname'],
                                    title_name=curve_data_param_temp['title_name'])
        except Exception as e:
            # 画图失败
            print("画图失败，失败原因是：", e.args)

    @staticmethod
    def dataframe_to_histogram(df, bins=10, column=None, by=None, grid=True, fig_size=None,
                               layout=None, show_flag=True, dpi=None, save_dir_path='',
                               save_figure_name='', **kwargs):
        """
        由Dataframe格式变换到直方图
        :param df: DataFrame
        :param bins: 一个图中直方的数量
        :param column: 用于将数据限制为列的子集，['column_name1','column_name2']
        :param by: 用于为不同的组形成直方图
        :param grid: 是否显示网格
        :param fig_size: 图形大小（以英寸为单位），（宽，高）
        :param layout: 布局，（行，列）
        :param show_flag: 是否展示，默认True即展示
        :param dpi: 保存图像的清晰度
        :param save_dir_path: 若和save_figure_name同时不为空，则保存至该路径下
        :param save_figure_name: 若和save_dir_path同时不为空，则保存至该路径下时使用该名字保存成图片格式
        :param kwargs: 其他参数
        :return:
        """

        # 防止内存不够，pycharm一直在生成图形，虽然没用plt.show()但是还是占内存
        if not show_flag:
            matplotlib.use("Agg")
        else:
            pass

        # 设置默认样式风格
        plt.style.use('default')

        # Times New Roman
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

        # 如果kwargs中有x轴或者y轴标签大小，则赋值
        xlabelsize = kwargs.get("xlabelsize", my_plot_config['figure_config']['histogram_xlabelsize'])
        ylabelsize = kwargs.get("ylabelsize", my_plot_config['figure_config']['histogram_ylabelsize'])

        # 如果kwargs中有标题字符的大小，则赋值
        titlesize = kwargs.get("titlesize", my_plot_config['figure_config']['histogram_titlesize'])
        matplotlib.rcParams.update({'axes.titlesize': titlesize})

        df.hist(bins=bins, column=column, by=by, grid=grid, figsize=fig_size, layout=layout,
                xlabelsize=xlabelsize, ylabelsize=ylabelsize)

        # 是否存储图片
        if (save_dir_path != '') and (save_figure_name != ''):
            dir_path_exist_or_makedir(save_dir_path)
            plt.savefig(save_dir_path + '/' + save_figure_name, dpi=dpi)
            print('+' * 20)
            print('Save picture to ' + save_dir_path + '/' + save_figure_name)
            print('+' * 20)
        else:
            pass

        # 是否展示
        if show_flag:
            plt.show()
        else:
            pass

        # # 清空当年画布
        # plt.clf()
        # 关闭所有图片
        plt.close('all')

        return

    @staticmethod
    def dataframe_column_to_boxplot(df, hue=None, orient='v', palette=None, width=0.8,
                                    flier_size=5, whis=1.5, show_flag=True, dpi=None,
                                    save_dir_path='', save_figure_name='', **kwargs):
        """
        由Dataframe格式变换到箱型图
        :param df: DataFrame中某一行column
        :param hue: dataframe的列名，按照列名中的值分类形成分类的条形图
        :param orient: 图像水平还是竖直显示（此参数一般当不传入x、y，只传入df的时候使用）,"v"|"h"
        :param palette: 调色板，控制图像的色调
        :param width: float，控制箱型图的宽度
        :param flier_size: float，用于指示离群值观察的标记大小
        :param whis: 确定离群值的上下界（IQR超过低和高四分位数的比例），此范围之外的点将被识别为异常值。IQR指的是上下四分位的差值。
        :param show_flag: 是否展示，默认True即展示
        :param dpi: 保存图像的清晰度
        :param save_dir_path: 若和save_figure_name同时不为空，则保存至该路径下
        :param save_figure_name: 若和save_dir_path同时不为空，则保存至该路径下时使用该名字保存成图片格式
        :param kwargs: 其他参数
        :return:
        """

        # 防止内存不够，pycharm一直在生成图形，虽然没用plt.show()但是还是占内存
        if not show_flag:
            matplotlib.use("Agg")
        else:
            pass

        # 设置默认样式风格
        plt.style.use('default')

        # Times New Roman
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

        # 如果kwargs中有标题参数，则添加标题
        title = kwargs.get("title", '')
        plt.title(title, fontsize=my_plot_config['figure_config']['boxplot_title_size'])
        # 如果kwargs中有设置字体大小参数，则设置字体大小
        font_scale = kwargs.get("font_scale", my_plot_config['figure_config']['boxplot_font_scale'])
        sns.set(font_scale=font_scale)

        # # 如果kwargs中有标题字符的大小，则赋值
        # labelsize = kwargs.get("labelsize", my_plot_config['figure_config']['boxplot_labelsize'])
        # matplotlib.rcParams.update({'axes.labelsize': labelsize})

        boxplot_res = sns.boxplot(data=df, hue=hue, orient=orient, palette=palette, width=width,
                                  fliersize=flier_size, whis=whis)

        # 是否存储图片
        if (save_dir_path != '') and (save_figure_name != ''):
            dir_path_exist_or_makedir(save_dir_path)
            plt.savefig(save_dir_path + '/' + save_figure_name, dpi=dpi)
            print('+' * 20)
            print('Save picture to ' + save_dir_path + '/' + save_figure_name)
            print('+' * 20)
        else:
            pass

        # 是否展示
        if show_flag:
            plt.show()
        else:
            pass

        # # 清空当前画布
        # plt.clf()
        # 关闭所有图片
        plt.close('all')

        return boxplot_res

    @staticmethod
    def dataframe_to_heat_map(data, fig_size=(14, 12), show_flag=True, dpi=None, save_dir_path='',
                              save_figure_name='', **kwargs):
        """
        热力图
        :param data: Dataframe格式
        :param fig_size: 图片尺寸
        :param show_flag: 是否展示，默认True即展示
        :param dpi: 保存图像的清晰度
        :param save_dir_path: 若和save_figure_name同时不为空，则保存至该路径下
        :param save_figure_name: 若和save_dir_path同时不为空，则保存至该路径下时使用该名字保存成图片格式
        :param kwargs: 其他参数
        :return:
        """

        # 防止内存不够，pycharm一直在生成图形，虽然没用plt.show()但是还是占内存
        if not show_flag:
            matplotlib.use("Agg")
        else:
            pass

        colormap = plt.cm.RdBu
        plt.figure(figsize=fig_size)
        # 如果kwargs中有标题参数，则添加标题
        title = kwargs.get("title", '')
        plt.title(title, fontsize=my_plot_config['figure_config']['boxplot_title_size'], y=1.05, size=15)

        sns.heatmap(data.astype(float), linewidths=0.1, vmax=1.0, square=True, cmap=colormap,
                    linecolor='white', annot=True)

        # 是否存储图片
        if (save_dir_path != '') and (save_figure_name != ''):
            dir_path_exist_or_makedir(save_dir_path)
            plt.savefig(save_dir_path + '/' + save_figure_name, dpi=dpi)
            print('+' * 20)
            print('Save picture to ' + save_dir_path + '/' + save_figure_name)
            print('+' * 20)
        else:
            pass

        # 是否展示
        if show_flag:
            plt.show()
        else:
            pass

        # # 清空当前画布
        # plt.clf()
        # 关闭所有图片
        plt.close('all')

        return
