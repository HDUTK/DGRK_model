#!/usr/bin/env python3.8.10
# -*- coding: utf-8 -*-
"""
function description: 此文件用于云冈石窟聚类后数据，每一类内数据的灰色关联度分析
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2020.2.5
time: 2024/5/17 14:30
version: V1.0
"""
import pandas as pd
from config.Yungang_Grottoes_config import *
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
from LogicAlgorithm.ML_method.Gray_Relation_Analysis.gray_relation_analysis import GRA


# 5min/30min/1h
Data_path = r'D:/PythonProject/MachineLearning/My_Dataset/YunGang_Grottoes_Data/SJD_23.0626-24.0606/' \
            r'after_interpolate_' + '5min' + '/'

class_sensor_list = 'AB01,AB02,AB03,AB04,AB05,AB06'.split(',')

# air_temperature/air_humidity/wall_temperature
my_columns_name_fliter = 'air_temperature'

# all/summer/winter
my_season_attribute = 'summer'


def read_class_sensor_data(class_sensor: list, data_path=Data_path, columns_name_fliter=None,
                           season_attribute='all', standard_flag=False, **kwargs):
    """
    将类列表内的传感器的值读取、合并
    :param class_sensor: 类列表内的传感器
    :param data_path: 数据地址
    :param columns_name_fliter: 按列名筛选
    :param season_attribute: 季节属性，默认'all'，还有'summer'/'summer'
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
        # 创建一个 StandardScaler 对象
        scaler = StandardScaler()
        # 对数据进行 Z-score 标准化
        res_scaled_data = pd.DataFrame(scaler.fit_transform(res.iloc[:, 1:]))
        # 将时间列插入第一列
        res_scaled_data.insert(0, 'time', res['time'])

        return res, res_scaled_data

    else:
        pass

    return res


my_res, my_res_scaled = read_class_sensor_data(class_sensor_list,
                                               columns_name_fliter=my_columns_name_fliter,
                                               season_attribute=my_season_attribute,
                                               standard_flag=True)

print(my_res)
print(my_res_scaled)
my_res_pearson = (my_res.iloc[:, 1:]).corr(method='pearson')
my_res_spearman = (my_res.iloc[:, 1:]).corr(method='spearman')
# 更改行列名
my_res_pearson.index = class_sensor_list
my_res_pearson.columns = class_sensor_list
my_res_spearman.index = class_sensor_list
my_res_spearman.columns = class_sensor_list
# 对行求和
my_res_pearson.loc['sum'] = my_res_pearson.apply(lambda x: x.sum(), axis=0)
my_res_spearman.loc['sum'] = my_res_spearman.apply(lambda x: x.sum(), axis=0)

# 展示所有的列
# pd.set_option('display.max_columns', None)
print('*' * 50)
print('my_res_pearson: \n', my_res_pearson)
print('*' * 50)
print('my_res_spearman: \n', my_res_spearman)
print('*' * 50)
# my_res_pearson.to_csv(Data_path + '_' + '_'.join(class_sensor_list) + '_pearson.CSV')
# my_res_spearman.to_csv(Data_path + '_' + '_'.join(class_sensor_list) + '_spearman.CSV')


GRA_res = GRA(my_res_scaled.iloc[:, 1:], plot_flag=False)
# 更改行列名
GRA_res.index = class_sensor_list
GRA_res.columns = class_sensor_list
# 对行求和
GRA_res.loc['mean'] = GRA_res.apply(lambda x: x.mean(), axis=0)
print('*' * 50)
print('GRA: \n', GRA_res)
print('*' * 50)
GRA_res.to_csv(Data_path + '_' + '_'.join(class_sensor_list) + '_GRA.CSV')

# 计算每一列的信息熵
entropy_list = []
for col in (my_res.iloc[:, 1:]).columns:
    col_entropy = entropy((my_res.iloc[:, 1:])[col].value_counts(normalize=True), base=2)
    entropy_list.append(col_entropy)

print('*' * 50)
print('Entropy: \n', entropy_list)
