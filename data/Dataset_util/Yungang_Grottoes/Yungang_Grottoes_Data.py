#!/usr/bin/env python3.8.10
# -*- coding: utf-8 -*-
"""
function description: 此文件用于云冈石窟数据的处理
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2020.2.5
time: 2024/3/25 16:57
version: V1.0
"""
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tsfresh import extract_features
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, \
    load_robot_execution_failures
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction.feature_calculators import abs_energy
from tsfresh.feature_extraction import ComprehensiveFCParameters

from common.common_func import dir_path_exist_or_makedir
from dao.table_process import table_process
from dao.local_database.YunGang_Grottoes_data import table_info, create_table
from common.plot_func import my_plot_func
from config.Yungang_Grottoes_config import *


class grottoes_data_process(object):
    """
    云冈数据处理
    """
    def __init__(self, table_list, time_between_list, now_location_str, resample_interval_str,
                 column_str, Data_path, interpolated_data_path, database_name,
                 interpolation_database_name):
        """
        初始化
        :param table_list: SJD多个传感器联表查询某指标
        :param time_between_list: 取的时间
        :param now_location_str: 现在传感器位置
        :param resample_interval_str: 采样间隔：5min/30min/1h
        :param column_str: 指标：air_temperature/air_humidity/wall_temperature
        :param Data_path: Data的path
        :param interpolated_data_path: 插值后的Data的path
        :param database_name: 原始数据库名字
        :param interpolation_database_name: 插值后的数据库名字
        """

        # SJD多个传感器联表查询某指标
        self.table_list = table_list
        # 取的时间
        self.time_between_list = time_between_list
        # 现在传感器位置
        self.now_location_str = now_location_str
        # 采样间隔：5min/30min/1h
        self.resample_interval_str = resample_interval_str
        # 指标：air_temperature/air_humidity/wall_temperature
        self.column_str = column_str
        # Data的path
        self.Data_path = Data_path
        # 插值后的Data的path
        self.interpolated_data_path = interpolated_data_path
        # 原始数据库名字
        self.database_name = database_name
        # 插值后的数据库名字
        self.interpolation_database_name = interpolation_database_name

    # def update_class_attribute_resample_interval_str(self, new_resample_interval_str, **kwargs):
    #     """
    #     改变resample_interval_str后需要更新类内的属性
    #     :param new_resample_interval_str: 新的采样间隔：5min/30min/1h
    #     :param kwargs: 其他参数
    #     :return:
    #     """
    #     self.resample_interval_str = new_resample_interval_str
    #
    #     # Data的path
    #     self.Data_path = r"D:/PythonProject/MachineLearning/My_Dataset/YunGang_Grottoes_Data" \
    #                      r"/SJD_23.0626-24.0606/after_resample_" + self.resample_interval_str + "/"
    #     # 插值后的Data的path
    #     self.interpolated_data_path = self.Data_path[:self.Data_path.find('after_resample_')] + 'after_interpolate_' \
    #                                   + self.resample_interval_str + '/'
    #     # 原始数据库名字: 'yungang_grottoes_database'
    #     self.database_name = 'yungang_grottoes_resample_20240725_' + self.resample_interval_str
    #     # 插值后的数据库名字
    #     self.interpolation_database_name = 'yungang_grottoes_interpolation_20240725_' + self.resample_interval_str
    #
    #     return

    def load_dataset_from_csv(self, csv_name, my_data_path=None, print_flag=True):
        """
        导入云冈石窟数据
        :param csv_name: csv文件名字
        :param my_data_path: csv文件路径
        :param print_flag: 是否打印，默认True
        :return: 返回云冈石窟数据的Dataframe格式
        """
        # Load Data
        if my_data_path:
            Dataset_dataframe = (pd.read_csv(my_data_path + csv_name + '.CSV', header=0,
                                             parse_dates=['time'])).iloc[:, 1:]
        else:
            Dataset_dataframe = (pd.read_csv(self.Data_path + csv_name + '.CSV', header=0,
                                             parse_dates=['time'])).iloc[:, 1:]
        if print_flag:
            print('*****************************')
            print('Selected Dataset:')
            print(Dataset_dataframe)
            print('*****************************')
        else:
            pass

        return Dataset_dataframe

    @staticmethod
    def sql_join_select_concat(select_column_str: str, from_table_list: list,
                               between_time_list: list):
        """
        多联表查询的sql
        :param select_column_str: sql语句中需要选择的参数，例如air_temperature
        :param from_table_list: sql语句中需要联表的表名列表
        例如['a01_sensor', 'a02_sensor', 'a03_sensor']
        :param between_time_list: sql语句中选择的时间范围
        例如['2023-06-07 00:00:00', '2024-03-07 00:00:00']
        :return: sql
        """
        sql = "SELECT " + from_table_list[0] + ".time, "
        for i in range(len(from_table_list)):
            sql = sql + from_table_list[i] + "." + select_column_str + " " \
                  + from_table_list[i] + ", "

        sql = sql[:-2] + " FROM " + from_table_list[0] + " "

        if len(from_table_list) == 1:
            pass
        else:
            for j in range(len(from_table_list[1:])):
                sql = sql + "INNER JOIN " + (from_table_list[1:])[j] + " ON " \
                      + from_table_list[0] + ".time=" + (from_table_list[1:])[j] + ".time "
        sql = sql + "WHERE " + from_table_list[0] + ".time BETWEEN " + "'" \
              + between_time_list[0] + "' AND '" + between_time_list[1] \
              + "' ORDER BY " + from_table_list[0] + ".time;"

        return sql

    def load_dataset_from_sql(self, select_column_str='', from_table_list=None, between_time_list=None,
                              select_data_sql=None, result_to_dataframe=False, sql_print=True,
                              to_csv=False, to_csv_path='', database_name=None, **kwargs):
        """
        使用sql中的select语句选择需要的数据
        :param select_column_str: sql语句中需要选择的参数，例如air_temperature
        :param from_table_list: sql语句中需要联表的表名列表，例如['a01_sensor', 'a02_sensor', 'a03_sensor']
        :param between_time_list: sql语句中选择的时间范围，例如['2023-06-07 00:00:00', '2024-03-07 00:00:00']
        :param select_data_sql: 直接输入sql语句
        :param result_to_dataframe: 若使用select，是否将结果输出为dataframe
        :param sql_print: 是否打印运行的sql
        :param to_csv: 是否要输出到csv文件保存
        :param to_csv_path: 若要输出到csv文件保存，则保存的path
        :param database_name: 使用的数据库
        :param kwargs: 其他参数
        :return: 使用sql的查询结果
        """
        database_name_temp = self.database_name if not database_name else database_name

        table_operate = table_info()
        if select_data_sql is not None:
            select_result = table_operate.execute_select_sql(sql=select_data_sql,
                                                             database_name=database_name_temp,
                                                             result_to_dataframe=result_to_dataframe,
                                                             sql_print=sql_print)
        else:
            sql = self.sql_join_select_concat(select_column_str, from_table_list, between_time_list)

            select_result = table_operate.execute_select_sql(sql=sql, database_name=database_name_temp,
                                                             result_to_dataframe=result_to_dataframe,
                                                             sql_print=sql_print)

        # 若Dataframe格式则打印
        if isinstance(select_result, pd.DataFrame):
            print('-' * 30)
            print('The result: ')
            print(select_result)
        else:
            pass

        # 输出至CSV
        if to_csv:
            # 若不是Dataframe格式则输出不了CSV 直接返回
            if not result_to_dataframe:
                print("The format is not Dataframe and can't be output to CSV !")
                return select_result
            else:
                pass

            # 若不存在Path则创建
            dir_path_exist_or_makedir(to_csv_path)

            # 输出至CSV
            try:
                print('+' * 15)

                if select_data_sql:
                    # 如果kwargs中有to_csv的文件名，则进行赋值，否则为设置的默认值temp
                    to_csv_name = kwargs.get("to_csv_name", 'temp')

                    print('Dataframe to CSV START ! The Path: ' + to_csv_path + to_csv_name + '.csv')

                    select_result.to_csv(to_csv_path + to_csv_name + ".csv")

                    print('Dataframe to CSV in OVER ! The Path: ' + to_csv_path + to_csv_name + '.csv')
                else:
                    print('Dataframe to CSV START ! The Path: ' + to_csv_path + between_time_list[0][:10]
                          + '——' + between_time_list[1][:10] + '_' + select_column_str + '.csv')

                    select_result.to_csv(to_csv_path + between_time_list[0][:10] + "——"
                                         + between_time_list[1][:10] + "_" + select_column_str + ".csv")

                    print('Dataframe to CSV in OVER ! The Path: ' + to_csv_path + between_time_list[0][:10]
                          + '——' + between_time_list[1][:10] + '_' + select_column_str + '.csv')
                print('+' * 15)
            except Exception as e:
                print('Some errors received during Dataframe to CSV. Concrete Problem: ', e)

        else:
            pass

        return select_result

    def linked_table_query(self, time_list=week_list, plot_flag=True, save_path=None, **kwargs):
        """
        使用联表查询查找table_list中某一指标，时间段week_list/week_lost_list/week_lost_week_list
        :param time_list: 时间列表：
        [['2023-06-07 00:00:00', '2023-06-13 23:59:59'],
        ['2023-06-14 00:00:00', '2023-06-20 23:59:59']]
        :param plot_flag: 是否画图，默认是
        :param save_path: 存储位置
        :param kwargs: 其他参数
        :return:
        """
        save_path = (self.Data_path + "every_week/") if not save_path else save_path

        for z in time_list:
            time_between_list = z
            # to_csv_path中的every_week/可改为every_week/_lost_week/或者every_week/_all_week/
            res = self.load_dataset_from_sql(select_column_str=self.column_str,
                                             from_table_list=self.table_list,
                                             between_time_list=time_between_list,
                                             result_to_dataframe=True,
                                             sql_print=True, to_csv=True,
                                             to_csv_path=save_path)
            ylabelname_dict = {'air_temperature': 'air temperature(℃)', 'air_humidity': 'air humidity(%)',
                               'wall_temperature': 'wall temperature(℃)'}
            if plot_flag:
                # save_dir_path中的every_week/可改为every_week/_lost_week/或者every_week/_all_week/
                my_plot_func.dataframe_to_time_curve(res, title_name=None, xlabelname='time',
                                                     ylabelname=ylabelname_dict[self.column_str],
                                                     show_flag=False, dpi=300, legend_font_size=12,
                                                     xlabel_font_size=25, ylabel_font_size=25,
                                                     label_font_size=20, figure_size=(20, 10),
                                                     save_dir_path=save_path,
                                                     save_figure_name=time_between_list[0][:10] + '——'
                                                     + time_between_list[1][:10] + '_' + self.column_str
                                                     + '.png')
            else:
                pass

        return

    def indicator_chart_from_all_time(self, database_name=None, save_path=None, **kwargs):
        """
        画出每一个传感器在所有时间范围内的3个指标图，传感器默认指标为空气温度、空气湿度、壁面温度
        :param database_name: 使用的数据库
        :param save_path: 图片保存的路径
        :param kwargs: 其他参数
        :return:
        """
        database_name_temp = self.database_name if not database_name else database_name
        save_path = (self.Data_path + "graph/") if not save_path else save_path

        for z in self.table_list:
            sql_new = "select time,CAST(air_temperature AS FLOAT) as air_temperature," \
                      "CAST(air_humidity AS FLOAT) as air_humidity," \
                      "CAST(wall_temperature AS FLOAT) as wall_temperature from " + z + ";"

            res = self.load_dataset_from_sql(select_data_sql=sql_new, database_name=database_name_temp,
                                             result_to_dataframe=True, sql_print=True, to_csv=False,
                                             to_csv_path=r"C:/Users/78545/Desktop/")
            # ylabelname_dict = {'air_temperature': 'air temperature(℃)', 'air_humidity': 'air humidity(%)',
            #                    'wall_temperature': 'wall temperature(℃)'}
            my_plot_func.time_curve({'y': res['air_temperature'], 'curvename': 'air temperature'},
                                    {'y': res['air_humidity'], 'curvename': 'air humidity'},
                                    {'y': res['wall_temperature'], 'curvename': 'wall temperature'},
                                    x_time=res['time'], title_name=None, xlabelname='Time',
                                    ylabelname='temperature/humidity(℃/%)', show_flag=False, dpi=300,
                                    legend_font_size=18, xlabel_font_size=25, ylabel_font_size=25,
                                    label_font_size=20, figure_size=(20, 10),
                                    save_dir_path=save_path,
                                    save_figure_name=z.upper()[:-7] + '.png')
        return

    # def plot_from_sql(self, res_from_sql, **kwargs):
    #     """
    #     根据sql返回的Dataframe画图（多个Dataframe，每一个Dataframe只取一列画/一个Dataframe每一列分别列出来画）
    #     :param res_from_sql: 由sql返回的Dataframe，第一列为time，后面三列为空气温度、空气湿度、壁面温度
    #     :param kwargs: 其他参数
    #     :return:
    #     """
    #     my_plot_func.time_curve({'y': res_from_sql['air_temperature'], 'curvename': 'air temperature'},
    #                             {'y': res_from_sql['air_humidity'], 'curvename': 'air humidity'},
    #                             {'y': res_from_sql['wall_temperature'], 'curvename': 'wall temperature'},
    #                             x_time=res_from_sql['time'], title_name=None, xlabelname='Time',
    #                             ylabelname='temperature/humidity(℃/%)', show_flag=True, dpi=300,
    #                             legend_font_size=18, xlabel_font_size=25, ylabel_font_size=25,
    #                             label_font_size=20, figure_size=(20, 10),
    #                             save_dir_path=self.Data_path + "every_week/_graph/",
    #                             save_figure_name=self.time_between_list[0][:10] + '——'
    #                             + self.time_between_list[1][:10] + '_' + self.column_str + '.png')
    #     return

    def dataframe_to_curve(self, res, save_path=None, **kwargs):
        """
        根据sql返回的Dataframe/csv导入的Dataframe画图（第一列为时间，后续每一列都要画在同一张图里）
        :param res: 由sql返回的Dataframe 或 load的csv得到的Dataframe，第一列为时间，后续每一列都要画在同一张图里
        :param save_path: 存储位置
        :param kwargs: 其他参数
        :return:
        """
        save_path = (self.Data_path + "every_week/graph/") if not save_path else save_path

        ylabelname_dict = {'air_temperature': 'air temperature(℃)', 'air_humidity': 'air humidity(%)',
                           'wall_temperature': 'wall temperature(℃)'}
        my_plot_func.dataframe_to_time_curve(res, title_name=None, xlabelname='time',
                                             ylabelname=ylabelname_dict[self.column_str], show_flag=True,
                                             dpi=300, legend_font_size=12, xlabel_font_size=25,
                                             ylabel_font_size=25, label_font_size=20, figure_size=(20, 10),
                                             save_dir_path=save_path,
                                             save_figure_name=self.time_between_list[0][:10] + '——'
                                             + self.time_between_list[1][:10] + '_' + self.column_str
                                             + '.png')
        return

    def table_list_to_histogram(self, data_path=None, save_path=None, **kwargs):
        """
        根据table_list画出其中每一个传感器的每一个指标在所有时间段的直方图
        :param data_path: 数据源位置
        :param save_path: 存储位置
        :param kwargs: 其他参数
        :return:
        """
        data_path = self.Data_path if not data_path else data_path
        save_path = (self.Data_path + "histogram/") if not save_path else save_path

        for k in self.table_list:
            res = self.load_dataset_from_csv((k.replace('_sensor', '')).upper(), my_data_path=data_path)

            my_plot_func.dataframe_to_histogram(res.iloc[:, 1:], bins=30, grid=True, fig_size=(10, 10),
                                                layout=None, show_flag=False,
                                                save_dir_path=save_path,
                                                save_figure_name=(k.replace('_sensor', '')).upper()
                                                + '_histogram.png', dpi=300)
        return

    def table_list_to_boxplot(self, data_path=None, save_path=None, **kwargs):
        """
        根据table_list画出其中每一个传感器的每一个指标在所有时间段的箱型图
        :param data_path: 数据源位置
        :param save_path: 存储位置
        :param kwargs: 其他参数
        :return:
        """
        data_path = self.Data_path if not data_path else data_path
        save_path = (self.Data_path + "boxplot/") if not save_path else save_path

        for k in self.table_list:
            res = self.load_dataset_from_csv((k.replace('_sensor', '')).upper(), my_data_path=data_path)

            for m in config_dict[(k.replace('_sensor', '')).upper()]:
                my_plot_func.dataframe_column_to_boxplot(res[sensor_type_dict[str(m)][1]], orient='v',
                                                         width=0.8, flier_size=5, whis=1.5,
                                                         show_flag=False, dpi=300, title='Boxplot',
                                                         save_dir_path=save_path,
                                                         save_figure_name=(k.replace('_sensor', '')).upper()
                                                         + '_' + sensor_type_dict[str(m)][1]
                                                         + '_boxplot.png', font_scale=1.5)
        return

    def split_by_week(self, data_path=None, split_week_list=None, save_path=None,
                      plot_flag=True, **kwargs):
        """
        将数据按周分割，画出其中每一个传感器的按周的直方图、箱型图，时间段week_list/week_lost_list/week_lost_week_list
        :param data_path: 数据源位置
        :param split_week_list: 按此week_list切割数据，若此week_list为None，则按
        :param save_path: 保存的路径
        :param plot_flag: 是否画图，默认是
        :param kwargs: 其他参数
        :return:
        """
        data_path = self.Data_path if not data_path else data_path
        save_path = self.Data_path if not save_path else save_path

        # 按不同的采样频率每周共有多少个采样点
        time_interval_temp = {'5min': 288 * 7, '30min': 48 * 7, '1h': 24 * 7}

        for k in self.table_list:
            res = self.load_dataset_from_csv((k.replace('_sensor', '')).upper(), my_data_path=data_path,
                                             print_flag=False)

            dir_path_exist_or_makedir(save_path + "every_week/" + k[:-7].upper())

            if split_week_list:
                # 从week_list中算每周
                for zz in split_week_list:
                    res_temp = res.loc[(res['time'] >= zz[0][:10]) & (res['time'] <= zz[1][:10])]

                    save_path_temp = save_path + "every_week/" + k[:-7].upper() + "/"
                    res_temp.to_csv(save_path_temp + str(res_temp.iloc[0, 0])[:10] + "——"
                                    + str(res_temp.iloc[-1, 0])[:10] + ".csv")

                    if plot_flag:
                        # 画出该周的曲线图，目前写了空气温度、空气湿度、壁面温度三种指标，若有其他指标，则需要修改
                        my_plot_func.time_curve({'y': res_temp['air_temperature'], 'curvename': 'air temperature'},
                                                {'y': res_temp['air_humidity'], 'curvename': 'air humidity'},
                                                {'y': res_temp['wall_temperature'], 'curvename': 'wall temperature'},
                                                x_time=res_temp['time'], title_name=k[:-7].upper() + ' '
                                                + str(res_temp.iloc[0, 0])[:10] + "——" + str(
                                res_temp.iloc[-1, 0])[:10],
                                                dpi=300, xlabelname='time', ylabelname='℃/%', show_flag=False,
                                                save_dir_path=save_path_temp + 'graph/',
                                                save_figure_name=str(res_temp.iloc[0, 0])[:10] + "——"
                                                + str(res_temp.iloc[-1, 0])[:10] + '.png', rotation=30,
                                                legend_font_size=15,
                                                figure_size=(20, 10))

                        # 画该周的直方图
                        my_plot_func.dataframe_to_histogram(res_temp.iloc[:, 1:], bins=15, grid=True, fig_size=(10, 10),
                                                            layout=None, show_flag=False,
                                                            save_dir_path=save_path_temp + "histogram/",
                                                            dpi=300,
                                                            save_figure_name=str(res_temp.iloc[0, 0])[:10] + "——"
                                                            + str(res_temp.iloc[-1, 0])[:10] + '_histogram.png',
                                                            xlabelsize=13, ylabelsize=15, titlesize=25)

                        # 画该周的箱型图
                        whis_dict = {'air_temperature': 1.5, 'air_humidity': 2.7, 'wall_temperature': 1.5}
                        for m in config_dict[(k.replace('_sensor', '')).upper()]:
                            my_plot_func.dataframe_column_to_boxplot(res_temp[sensor_type_dict[str(m)][1]], orient='v',
                                                                     width=0.8, flier_size=10,
                                                                     whis=whis_dict[sensor_type_dict[str(m)][1]],
                                                                     show_flag=False, dpi=300, title='Boxplot',
                                                                     save_dir_path=save_path_temp + "boxplot/",
                                                                     save_figure_name=str(res_temp.iloc[0, 0])[:10]
                                                                     + "——" + str(res_temp.iloc[-1, 0])[:10] + '_'
                                                                     + sensor_type_dict[str(m)][1] + '_boxplot.png')

            else:
                # 从data_path中的csv第一行开始算每周，算到最后一行为止
                for n in range(int(len(res) / time_interval_temp[self.resample_interval_str])):
                    res_temp = res.iloc[n*time_interval_temp[self.resample_interval_str]:
                                        n*time_interval_temp[self.resample_interval_str] +
                                        time_interval_temp[self.resample_interval_str], :]

                    save_path_temp = save_path + "every_week/" + k[:-7].upper() + "/"
                    res_temp.to_csv(save_path_temp + str(res_temp.iloc[0, 0])[:10] + "——"
                                    + str(res_temp.iloc[-1, 0])[:10] + ".csv")

                    if plot_flag:
                        # 画出该周的曲线图，目前写了空气温度、空气湿度、壁面温度三种指标，若有其他指标，则需要修改
                        my_plot_func.time_curve({'y': res_temp['air_temperature'], 'curvename': 'air temperature'},
                                                {'y': res_temp['air_humidity'], 'curvename': 'air humidity'},
                                                {'y': res_temp['wall_temperature'], 'curvename': 'wall temperature'},
                                                x_time=res_temp['time'], title_name=k[:-7].upper() + ' ' +
                                                str(res_temp.iloc[0, 0])[:10] + "——" + str(res_temp.iloc[-1, 0])[:10],
                                                dpi=300, xlabelname='time', ylabelname='℃/%', show_flag=False,
                                                save_dir_path=save_path_temp + 'graph/',
                                                save_figure_name=str(res_temp.iloc[0, 0])[:10] + "——"
                                                + str(res_temp.iloc[-1, 0])[:10] + '.png', rotation=30,
                                                legend_font_size=15, figure_size=(20, 10))

                        # 画该周的直方图
                        my_plot_func.dataframe_to_histogram(res_temp.iloc[:, 1:], bins=15, grid=True, fig_size=(10, 10),
                                                            layout=None, show_flag=False,
                                                            save_dir_path=save_path_temp+"histogram/",
                                                            dpi=300, save_figure_name=str(res_temp.iloc[0, 0])[:10]
                                                            + "——" + str(res_temp.iloc[-1, 0])[:10] + '_histogram.png',
                                                            xlabelsize=13, ylabelsize=15, titlesize=25)

                        # 画该周的箱型图
                        whis_dict = {'air_temperature': 1.5, 'air_humidity': 2.7, 'wall_temperature': 1.5}
                        for m in config_dict[(k.replace('_sensor', '')).upper()]:
                            my_plot_func.dataframe_column_to_boxplot(res_temp[sensor_type_dict[str(m)][1]], orient='v',
                                                                     width=0.8, flier_size=10,
                                                                     whis=whis_dict[sensor_type_dict[str(m)][1]],
                                                                     show_flag=False, dpi=300, title='Boxplot',
                                                                     save_dir_path=save_path_temp+"boxplot/",
                                                                     save_figure_name=str(res_temp.iloc[0, 0])[:10]
                                                                     + "——" + str(res_temp.iloc[-1, 0])[:10] + '_'
                                                                     + sensor_type_dict[str(m)][1] + '_boxplot.png')
                    else:
                        pass

        return

    def interpolate_to_5min_data(self, time_list=None, database_insert=False, **kwargs):
        """
        对5min的数据进行插值，若没有时间段则按self.time_between_list时间段进行插值
        否则按时间段week_list进行插值
        :param time_list: 时间列表：
        [['2023-06-07 00:00:00', '2023-06-13 23:59:59'],
        ['2023-06-14 00:00:00', '2023-06-20 23:59:59']]
        :param database_insert: 是否在插值完后执行数据库插入，默认否
        :param kwargs: 其他参数
        :return:
        """
        if not time_list:
            for k in self.table_list:
                sql_new = "select time,CAST(air_temperature AS FLOAT) as air_temperature," \
                          "CAST(air_humidity AS FLOAT) as air_humidity," \
                          "CAST(wall_temperature AS FLOAT) as wall_temperature from " + k + \
                          " WHERE " + k + ".time BETWEEN " + "'" + self.time_between_list[0] +\
                          "' AND '" + self.time_between_list[1] + "' ORDER BY " + k + ".time;"

                res = self.load_dataset_from_sql(select_data_sql=sql_new, database_name=self.database_name,
                                                 result_to_dataframe=True, sql_print=True, to_csv=False,
                                                 to_csv_path=r"C:/Users/78545/Desktop/")

                # 将时间轴设置为索引并进行插值
                res = res.set_index('time')
                res = res.interpolate(method='time')
                # 将索引改为列
                res = res.reset_index()

                # 若不存在Path则创建
                dir_path_exist_or_makedir(self.interpolated_data_path + 'select_time/'
                                          + (k.replace('_sensor', '')).upper() + '/')

                save_path = self.interpolated_data_path + 'select_time/' + (k.replace('_sensor', '')).upper() + '/'
                # to_csv
                res.to_csv(save_path + self.time_between_list[0][:10] + '——' + self.time_between_list[1][:10] + ".csv")

                if database_insert:
                    # 若不存在则创建数据表
                    create_table((k.replace('_sensor', '')).upper(), database_name=self.interpolation_database_name)

                    # 插入数据
                    table = table_info()
                    table.table = (k.replace('_sensor', '')).upper() + '_Sensor'
                    table_list = table_process.make_multi_insert_list(res)

                    print('-' * 50)
                    print('Inserting data, table name:  ' + (k.replace('_sensor', '')).upper() + '_Sensor')

                    table.insert_data(table_list, now_location_str=(k.replace('_sensor', '')).upper(),
                                      database_name=self.interpolation_database_name, sql_print=False)

                    print('Finish insert, table name:  ' + (k.replace('_sensor', '')).upper() + '_Sensor')
                    print('-' * 50)
                else:
                    pass

                # 画出该周的曲线图，目前写了空气温度、空气湿度、壁面温度三种指标，若有其他指标，则需要修改
                my_plot_func.time_curve({'y': res['air_temperature'], 'curvename': 'air temperature'},
                                        {'y': res['air_humidity'], 'curvename': 'air humidity'},
                                        {'y': res['wall_temperature'], 'curvename': 'wall temperature'},
                                        x_time=res['time'], title_name=k[:-7].upper() + ' : '
                                        + self.time_between_list[0][:10] + '——' + self.time_between_list[1][:10],
                                        dpi=300, xlabelname='time', ylabelname='temperature/humidity(℃/%)',
                                        show_flag=False, legend_font_size=20, xlabel_font_size=25,
                                        ylabel_font_size=25, label_font_size=20, figure_size=(20, 10),
                                        title_font_size=30, save_dir_path=save_path + 'graph/',
                                        save_figure_name=self.time_between_list[0][:10] + '——'
                                        + self.time_between_list[1][:10] + '.png')
        else:
            for k in self.table_list:
                for n in time_list:
                    res = self.load_dataset_from_csv(n[0][:10] + '——' + n[1][:10], my_data_path=self.Data_path
                                                     + 'every_week/' + (k.replace('_sensor', '')).upper() + '/')

                    # 将时间轴设置为索引并进行插值
                    res = res.set_index('time')
                    res = res.interpolate(method='time')
                    # 将索引改为列
                    res = res.reset_index()

                    # 若不存在Path则创建
                    dir_path_exist_or_makedir(self.interpolated_data_path + 'every_week/'
                                              + (k.replace('_sensor', '')).upper() + '/')

                    # to_csv
                    res.to_csv(self.interpolated_data_path + 'every_week/' + (k.replace('_sensor', '')).upper()
                               + '/' + n[0][:10] + '——' + n[1][:10] + ".csv")

                    # 画出该周的曲线图，目前写了空气温度、空气湿度、壁面温度三种指标，若有其他指标，则需要修改
                    my_plot_func.time_curve({'y': res['air_temperature'], 'curvename': 'air temperature'},
                                            {'y': res['air_humidity'], 'curvename': 'air humidity'},
                                            {'y': res['wall_temperature'], 'curvename': 'wall temperature'},
                                            x_time=res['time'], title_name=k[:-7].upper() + ' : ' + n[0][:10]
                                            + '——' + n[1][:10], dpi=300, xlabelname='time',
                                            ylabelname='temperature/humidity(℃/%)', show_flag=True,
                                            legend_font_size=20, xlabel_font_size=25, ylabel_font_size=25,
                                            label_font_size=20, figure_size=(20, 10), title_font_size=30,
                                            save_dir_path=self.interpolated_data_path + 'every_week/'
                                            + (k.replace('_sensor', '')).upper() + '/graph/',
                                            save_figure_name=n[0][:10] + '——' + n[1][:10] + '_'
                                            + self.column_str + '.png')

        return

    @staticmethod
    def extract_feature_func(df_temp, **kwargs):
        """
        抽取特征函数
        :param df_temp: 传入需要提取特征的Dataframe，首列为id列，第二列为time列，后续为按时间变化的变量列
        :param kwargs: 其他参数
        :return: 抽取完的特征，Dataframe格式，shape为1*特征数
        """

        # 特征自定义设置
        kind_to_fc_parameters = extract_feature_dict
        extracted_features = extract_features(df_temp, column_id="id", column_sort="time",
                                              kind_to_fc_parameters=kind_to_fc_parameters)

        return extracted_features

    def extract_feature_every_week(self, data_path=None, split_week_list=None, print_flag=False,
                                   set_option_flag=False, to_csv=False, save_path=None, **kwargs):
        """
        获取所有传感器的时间序列特征，格式为Dataframe，具体特征见config配置文件中的字典配置
        输出至CSV文件，形如A63_features.csv
        :param data_path: 数据源位置
        :param split_week_list: 按此week_list切割数据，若此week_list为None，则按
        :param print_flag: 是否将传感器的特征的Dataframe打印，默认False
        :param set_option_flag: 是否显示所有列的结果，默认False
        :param to_csv: 是否输出至CSV文件，默认False
        :param save_path: 若输出至CSV文件，则需要将路径传入
        :param kwargs: 其他参数
        :return:
        """
        split_week_list = week_list if not split_week_list else split_week_list

        data_path = self.Data_path if not data_path else data_path
        save_path = self.Data_path if not save_path else save_path

        for k in self.table_list:
            extract_feature = None
            first_flag = True

            print('+' * 75)
            print('Extract features of ' + (k.replace('_sensor', '')).upper() + ' sensor Start!')

            if set_option_flag:
                # 展示所有列的结果
                pd.set_option('display.max_columns', 50)
            else:
                pass

            for z in split_week_list:
                if z in week_delete_list:
                    continue
                else:
                    print('Extract features in ' + z[0] + '——' + z[1] + '.')
                    res = self.load_dataset_from_csv(z[0][:10] + '——' + z[1][:10],
                                                     my_data_path=data_path + 'every_week/'
                                                     + (k.replace('_sensor', '')).upper() + '/',
                                                     print_flag=False)
                    # 将id插入第一列且全为1，在抽取特征时需要使用
                    res.insert(0, 'id', 1)

                    # 抽取特征
                    extract_feature_temp = self.extract_feature_func(res)

                    # 将时间插入第一列
                    extract_feature_temp.insert(0, 'time', z[0][:10] + '——' + z[1][:10])

                    # 若为第一次，则直接赋给extract_feature，否则进行合并
                    if first_flag:
                        extract_feature = extract_feature_temp
                        first_flag = False
                    else:
                        extract_feature = pd.concat([extract_feature, extract_feature_temp])

            if print_flag:
                print(extract_feature)
            else:
                pass

            if to_csv:
                print('/' * 30)
                print('Feature Dataframe to CSV ! The Path: ' + save_path
                      + (k.replace('_sensor', '')).upper() + '_features.csv')
                extract_feature.to_csv(save_path + (k.replace('_sensor', '')).upper() + '_features.csv')
                print('/' * 30)
            else:
                pass

            print('Extract features of ' + (k.replace('_sensor', '')).upper() + ' sensor Over!')
            print('+' * 75)

        return

    def simple_process_feature(self, columns_name_fliter=None, season_attribute='all',
                               data_path=None, to_csv=False, save_path=None, **kwargs):
        """
        对数据特征进行简单操作，并进行零均值标准化
        :param data_path: 数据源位置
        :param columns_name_fliter: 按列名筛选
        :param season_attribute: 季节属性，默认'all'，还有'summer'/'summer'
        :param to_csv: 是否输出至CSV文件，默认False
        :param save_path: 若输出至CSV文件，则需要将路径传入
        :param kwargs: 其他参数
        :return: 计算得到的特征矩阵、零均值标准化后的数据
        """
        data_path = self.Data_path if not data_path else data_path
        save_path = self.Data_path if not save_path else save_path

        res = None
        # 忽略UserWarning警告
        warnings.filterwarnings('ignore', category=UserWarning)
        for k in self.table_list:
            res_temp = self.load_dataset_from_csv((k.replace('_sensor', '')).upper() + '_features',
                                                  my_data_path=data_path, print_flag=False)

            if season_attribute == 'all':
                pass
            elif season_attribute == 'summer':
                start = (summer_week_list[0][0])[:10] + '——' + (summer_week_list[0][1])[:10]
                end = (summer_week_list[-1][0])[:10] + '——' + (summer_week_list[-1][1])[:10]
                res_temp = res_temp.loc[(res_temp['time'] >= start) & (res_temp['time'] <= end)]
            elif season_attribute == 'winter':
                start = (winter_week_list[0][0])[:10] + '——' + (winter_week_list[0][1])[:10]
                end = (winter_week_list[-1][0])[:10] + '——' + (winter_week_list[-1][1])[:10]
                res_temp = res_temp.loc[(res_temp['time'] >= start) & (res_temp['time'] <= end)]
            else:
                pass

            # 按列名筛选
            res_temp = res_temp.filter(regex=columns_name_fliter).round(3)
            # reshape
            res_temp = pd.DataFrame(np.reshape(res_temp.to_numpy(), (1, np.shape(res_temp)[0] * np.shape(res_temp)[1])))

            # 将传感器标识插入第一列
            res_temp.insert(0, 'sensor_name', (k.replace('_sensor', '')).upper())

            if k == self.table_list[0]:
                res = res_temp
            else:
                res = pd.concat([res, res_temp])

        # 创建一个 StandardScaler 对象
        scaler = StandardScaler()
        # 对数据进行 Z-score 标准化
        res_scaled_data = pd.DataFrame(scaler.fit_transform(res.iloc[:, 1:])).round(3)
        # 将传感器标识插入第一列
        res_scaled_data.insert(0, 'sensor_name', res['sensor_name'].values)

        if to_csv:
            print('/' * 30)
            print('Feature Dataframe of All Sensors to CSV ! The Path: ' + save_path
                  + '_' + season_attribute + '_' + columns_name_fliter + '_features.csv')
            res.to_csv(save_path + '_' + season_attribute + '_' + columns_name_fliter
                       + '_original_features.csv')
            res_scaled_data.to_csv(save_path + '_' + season_attribute + '_'
                                   + columns_name_fliter + '_features.csv')
            print('/' * 30)
        else:
            pass

        return res, res_scaled_data

    @staticmethod
    def read_class_sensor_data(class_sensor_list: list, data_path: str,
                               columns_name_fliter=None, season_attribute='all',
                               standard_flag=False, **kwargs):
        """
        得到想要的训练用train_X值，将类列表内的传感器的值读取、合并
        :param class_sensor_list: 类列表内的传感器，['A63', 'A64', 'A65']
        :param data_path: 数据地址
        :param columns_name_fliter: 按列名筛选，['air_temperature','air_humidity','wall_temperature']
        :param season_attribute: 季节属性，默认'all'，还有'summer'/'winter
        :param standard_flag: 是否标准化flag
        :param kwargs: 其他参数
        :return: 合并后的Dataframe (+ 标准化后的Dataframe)
        """

        res = None
        for k in class_sensor_list:
            res_temp = (pd.read_csv(data_path + k + '.CSV', header=0, parse_dates=['time'])).iloc[:, 1:]

            if season_attribute == 'all':
                pass
            elif season_attribute == 'summer':
                res_temp = res_temp.loc[(res_temp['time'] >= summer_week_list[0][0]) &
                                        (res_temp['time'] <= summer_week_list[-1][1])]
            elif season_attribute == 'winter':
                res_temp = res_temp.loc[(res_temp['time'] >= winter_week_list[0][0]) &
                                        (res_temp['time'] <= winter_week_list[-1][1])]
            else:
                pass

            # 删去不要的周
            for m in week_delete_list:
                res_temp = res_temp.drop(res_temp[(res_temp['time'] >= m[0]) & (res_temp['time'] <= m[1])].index)

            # 按列名筛选
            res_temp_after_del = res_temp.filter(regex=columns_name_fliter).round(3)
            # 更改列名
            res_temp_after_del.rename(columns={columns_name_fliter: k + '_' + columns_name_fliter}, inplace=True)

            # 将时间列插入第一列
            res_temp_after_del.insert(0, 'time', res_temp['time'])

            # 合并
            if k == class_sensor_list[0]:
                res = res_temp_after_del
            else:
                res = pd.merge(res, res_temp_after_del, on='time')

        if standard_flag:
            # 创建一个 StandardScaler/MinMaxScaler 对象
            scaler = StandardScaler()
            # 对数据进行 Z-score 标准化
            res_scaled_data = pd.DataFrame(scaler.fit_transform(res.iloc[:, 1:]))
            # 将时间列插入第一列
            res_scaled_data.insert(0, 'time', res['time'])

            return res, res_scaled_data

        else:
            pass

        return res

    @staticmethod
    def overall_value_data(overall_value_table_name: str, data_path: str, columns_name_fliter=None,
                           season_attribute='all', **kwargs):
        """
        得到想要的训练用train_Y值，取室内均值表格内的数据
        :param overall_value_table_name: 室内均值表格名称，不带开头的'_'
        :param data_path: 数据地址
        :param columns_name_fliter: 按列名筛选，'air_temperature'/'air_humidity'/'wall_temperature'
        :param season_attribute: 季节属性，默认'all'，还有'summer'/'winter
        :param kwargs: 其他参数
        :return: 室内均值表格内的数据，Dataframe格式
        """

        # _a63-b68/_a01-b06
        res_temp = (pd.read_csv(data_path + '_' + overall_value_table_name
                                + '.CSV', header=0,
                                parse_dates=['time'])).iloc[:, 1:]

        if season_attribute == 'all':
            pass
        elif season_attribute == 'summer':
            res_temp = res_temp.loc[(res_temp['time'] >= summer_week_list[0][0]) &
                                    (res_temp['time'] <= summer_week_list[-1][1])]
        elif season_attribute == 'winter':
            res_temp = res_temp.loc[(res_temp['time'] >= winter_week_list[0][0]) &
                                    (res_temp['time'] <= winter_week_list[-1][1])]
        else:
            pass

        # 删去不要的周
        for m in week_delete_list:
            res_temp = res_temp.drop(res_temp[(res_temp['time'] >= m[0]) & (res_temp['time'] <= m[1])].index)

        # 按列名筛选
        res_temp_after_del = res_temp.filter(regex=columns_name_fliter + '_average').round(3)

        # 将时间列插入第一列
        res_temp_after_del.insert(0, 'time', res_temp['time'])

        return res_temp_after_del

    def merge_feature(self, columns_name_list=None, season_attribute='all',
                      data_path=None, to_csv=False, save_path=None, **kwargs):
        """
        对数据特征进行合并操作
        :param data_path: 数据源位置
        :param columns_name_list: 按列名筛选
        :param season_attribute: 季节属性，默认'all'，还有'summer'/'summer'
        :param to_csv: 是否输出至CSV文件，默认False
        :param save_path: 若输出至CSV文件，则需要将路径传入
        :param kwargs: 其他参数
        :return: 得到的特征矩阵
        """
        data_path = self.Data_path if not data_path else data_path
        save_path = self.Data_path if not save_path else save_path

        if len(columns_name_list) == 1:
            print('The Features needed to be merged are only One-Dimensional!')
            return
        else:
            pass

        res_temp = (pd.read_csv(data_path + '_' + season_attribute + '_'
                   + columns_name_list[0] + '_features' + '.CSV',
                   header=0)).iloc[:, 1:]

        # 开始合并
        for zz in columns_name_list[1:]:
            res_temp_temp = (pd.read_csv(data_path + '_' + season_attribute + '_'
                                         + zz + '_features' + '.CSV',
                                         header=0)).iloc[:, 1:]
            res_temp = pd.merge(res_temp, res_temp_temp, on='sensor_name')

        if to_csv:
            print('/' * 30)
            print('Feature Dataframe of All Sensors to CSV ! The Path: ' + save_path
                  + '_' + season_attribute + '_merge_features.csv')
            res_temp.to_csv(save_path + '_' + season_attribute + '_merge_features.csv')
            print('/' * 30)
        else:
            pass

        return res_temp  # _A63-B68


if __name__ == "__main__":

    import argparse

    # 采样间隔：5min/30min/1h
    resample_interval_str = '5min'
    # 指标：air_temperature/air_humidity/wall_temperature
    column_str = 'air_temperature'

    parser = argparse.ArgumentParser()
    # SJD多个传感器联表查询某指标
    parser.add_argument('--table_list', default=SJD_table_list, type=list)
    # 取的时间
    parser.add_argument('--time_between_list',
                        default=['2023-07-03 00:00:00', '2023-07-18 23:59:59'], type=list)
    # 分割的时间段
    parser.add_argument('--split_week_list', default=summer_week_list, type=list)
    # 现在传感器位置
    parser.add_argument('--now_location_str', default='A63', type=str)
    # 采样间隔：5min/30min/1h
    parser.add_argument('--resample_interval_str', default=resample_interval_str, type=str)
    # 指标：air_temperature/air_humidity/wall_temperature
    parser.add_argument('--column_str', default=column_str, type=str)
    # Data的path
    parser.add_argument('--Data_path', default=r"D:/PythonProject/MachineLearning/My_Dataset"
                                               r"/YunGang_Grottoes_Data/SJD_23.0626-24.0606"
                                               r"/after_resample_" + resample_interval_str
                                               + "/", type=str)
    # 插值后的Data的path
    parser.add_argument('--interpolated_data_path',
                        default=r"D:/PythonProject/MachineLearning/My_Dataset"
                                r"/YunGang_Grottoes_Data/SJD_23.0626-24.0606/after_interpolate_"
                                + resample_interval_str + "/", type=str)
    # 原始数据库名字: 'yungang_grottoes_database'
    parser.add_argument('--database_name',
                        default='yungang_grottoes_resample_20240725_' + resample_interval_str, type=str)
    # 插值后的数据库名字
    parser.add_argument('--interpolation_database_name',
                        default='yungang_grottoes_interpolation_20240725_' + resample_interval_str, type=str)
    args = parser.parse_args()

    # 实例化
    my_process = grottoes_data_process(
        table_list=args.table_list, time_between_list=args.time_between_list,
        now_location_str=args.now_location_str, resample_interval_str=args.resample_interval_str,
        column_str=args.column_str, Data_path=args.Data_path,
        interpolated_data_path=args.interpolated_data_path, database_name=args.database_name,
        interpolation_database_name=args.interpolation_database_name)

    # """
    # 使用sql中的select语句选择需要的数据
    # """
    # # 多联表查询的sql
    # my_res = my_process.load_dataset_from_sql('air_temperature', from_table_list=my_process.table_list,
    #                                           between_time_list=my_process.time_between_list,
    #                                           result_to_dataframe=True, sql_print=False, to_csv=False,
    #                                           to_csv_path=r'C:/Users/78545/Desktop/',
    #                                           database_name=args.interpolation_database_name)
    # # 直接使用sql查询
    # name_list = ['a01', 'a02', 'a03', 'a04', 'a05', 'a06',
    #              'ab01', 'ab02', 'ab03', 'ab04', 'ab05', 'ab06',
    #              'b01', 'b02', 'b03', 'b04', 'b05', 'b06']
    # for i in name_list:
    #     my_res = my_process.load_dataset_from_sql(select_data_sql="SELECT time,air_temperature,air_humidity,"
    #                                               "wall_temperature FROM " + i + "_sensor WHERE " + i
    #                                               + "_sensor.time BETWEEN " + "'2023-07-28 00:00:00'" + "AND"
    #                                               + "'2023-11-01 00:00:00'" + ";",
    #                                               result_to_dataframe=True, sql_print=True, to_csv=True,
    #                                               to_csv_path=r'C:/Users/78545/Desktop/temp/', to_csv_name=i,
    #                                               database_name='yungang_grottoes_resample_20240725_5min')

    # my_res = my_process.load_dataset_from_sql(select_data_sql="SELECT time,air_temperature,air_humidity,"
    #                                           "wall_temperature FROM " + i + "_sensor WHERE " + i
    #                                           + "_sensor.time BETWEEN " + "'2023-07-28 00:00:00'" + "AND"
    #                                           + "'2023-11-01 00:00:00'" + ";",
    #                                           result_to_dataframe=True, sql_print=True, to_csv=True,
    #                                           to_csv_path=r'C:/Users/78545/Desktop/temp/', to_csv_name=i,
    #                                           database_name='yungang_grottoes_20240725')

    """
    根据sql返回的Dataframe/csv导入的Dataframe画图（第一列为时间，后续每一列都要画在同一张图里）
    默认存储至Data_path/every_week/graph/下
    """
    # my_process.dataframe_to_curve(my_res, save_path=args.interpolated_data_path+"every_week/graph/")

    """
    使用联表查询查找table_list中某一指标，时间段固定，为week_list/week_lost_list/week_lost_week_list
    默认存储至Data_path/every_week/下
    """
    # # my_process.linked_table_query(time_list=week_list)
    # my_process.linked_table_query(
    #     time_list=[['2023-09-20 00:00:00', '2023-09-26 23:59:59'],
    #                ['2023-09-27 00:00:00', '2023-10-03 23:59:59']],
    #     save_path=args.interpolated_data_path+"every_week/")

    """
    画出每一个传感器在所有时间范围内的3个指标图，传感器默认指标为空气温度、空气湿度、壁面温度
    默认存储至Data_path/graph/下
    """
    # my_process.indicator_chart_from_all_time(database_name=args.interpolation_database_name,
    #                                          save_path=args.interpolated_data_path+"graph/")

    """
    根据table_list画出其中每一个传感器在所有时间段的直方图
    默认存储至Data_path/histogram/下
    """
    # my_process.table_list_to_histogram(time_list=summer_week_list,
    #                                    data_path=args.interpolated_data_path,
    #                                    save_path=args.interpolated_data_path+"histogram/")

    """
    根据table_list画出其中每一个传感器的每一个指标在所有时间段的箱型图
    默认存储至Data_path/boxplot/下
    """
    # my_process.table_list_to_boxplot(time_list=summer_week_list,
    #                                  data_path=args.interpolated_data_path,
    #                                  save_path=args.interpolated_data_path+"boxplot/")

    """
    将数据按table_list内的周分割，画出其中每一个传感器的按周的原图、直方图、箱型图
    时间段week_list/week_lost_list/week_lost_week_list
    默认存储至every_week/graph/和every_week/histogram/和every_week/boxplot/下
    """
    # my_process.split_by_week(data_path=args.interpolated_data_path,
    #                          split_week_list=args.split_week_list,
    #                          save_path=args.interpolated_data_path)

    # """
    # 对5min的数据进行插值，若没有时间段则按self.time_between_list时间段进行插值
    # 否则按时间段week_list进行插值
    # 若按self.time_between_list时间段进行插值则存储至after_interpolate_/select_time/下
    # 否则存储至every_week/下
    # """
    # # my_process.time_between_list = ['2023-06-07 00:00:00', '2024-04-06 23:59:59']
    # # my_process.interpolate_to_5min_data(database_insert=True)

    """
    获取table_list内所有传感器的时间序列特征，格式为Dataframe，具体特征见config配置文件中的字典配置，输出至CSV文件
    """
    # my_process.extract_feature_every_week(data_path=args.interpolated_data_path,
    #                                       split_week_list=args.split_week_list,
    #                                       to_csv=True,
    #                                       save_path=args.interpolated_data_path)

    """
    将所有特征reshape、并将所有传感器特征合并、归一化
    （air_temperature/air_humidity/wall_temperature）（all/summer/winter）
    """
    # my_res, feature = my_process.simple_process_feature(columns_name_fliter='air_humidity',
    #                                                     season_attribute='summer',
    #                                                     data_path=args.interpolated_data_path,
    #                                                     to_csv=True,
    #                                                     save_path=args.interpolated_data_path)

    """
    将得到的特征结合
    （air_temperature/air_humidity/wall_temperature）
    """
    # my_feature_res = my_process.merge_feature(columns_name_list=['air_temperature', 'air_humidity'],
    #                                           season_attribute='summer',
    #                                           data_path=args.interpolated_data_path,
    #                                           to_csv=True,
    #                                           save_path=args.interpolated_data_path)
