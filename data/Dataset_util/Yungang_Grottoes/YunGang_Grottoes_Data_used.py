#!/usr/bin/env python3.8.10
# -*- coding: utf-8 -*-
"""
function description: 此文件用于云冈石窟数据的中间处理，处理完后使用
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2020.2.5
time: 2024/6/19 13:09
version: V1.0
"""
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from data.Dataset_util.Yungang_Grottoes.Yungang_Grottoes_Data import grottoes_data_process


def Yungang_Grottoes_Data_preprocessing(Data_path, class_sensor_list, my_columns_name_fliter,
                                        my_season_attribute, validate_week_list, test_week_list,
                                        std_flag=False, my_batch_size=32, train_test_split_flag=False,
                                        average_table_name='overall_value_sensor', **kwargs):
    """
    云冈石窟数据的中间处理，处理完后使用
    :param Data_path: 数据地址
    :param class_sensor_list: 使用的传感器list：[B68, A65, AB11, B66]
    :param my_columns_name_fliter: 要取的属性，air_temperature/air_humidity/wall_temperature
    :param my_season_attribute: 要取的季节，按config层理的列表来取
    :param my_batch_size: batch_size，默认32
    :param validate_week_list: 验证数据集周列表，例如
    [['2023-07-26 00:00:00', '2023-08-01 23:59:59'],
    ['2023-09-20 00:00:00', '2023-09-26 23:59:59']]
    :param test_week_list: 测试数据集周列表，例如
    [['2023-07-26 00:00:00', '2023-08-01 23:59:59'],
    ['2023-09-20 00:00:00', '2023-09-26 23:59:59']]
    :param std_flag: 是否需要标准化，默认否
    :param train_test_split_flag: 是否使用sklearn中的train_test_split函数分出测试集和验证集/测试集，默认否
    :param average_table_name: 取数据集的target时需要的文件名，包含Train和Test
    :param kwargs: 其他参数
    :return: 训练、验证、测试数据，以dict格式显示，数据格式分别为Dataframe格式的res_data_dict、
    TensorDataset格式的res_TensorDataset_dict、DataLoader格式的res_DataLoader_dict、
    Tensor格式的res_data_tensor_dict

    Example
    -------
    >>> my_res_data_dict, my_res_TensorDataset_dict, my_res_DataLoader_dict, my_res_data_tensor_dict = \
    Yungang_Grottoes_Data_preprocessing(Data_path='./my/', class_sensor_list=[B68, A65, AB11, B66],
    my_columns_name_fliter='air_temperature', my_season_attribute='summer',
    validate_week_list=[['2023-07-26 00:00:00', '2023-08-01 23:59:59'],
    ['2023-09-20 00:00:00', '2023-09-26 23:59:59']],
    test_week_list=[['2023-07-26 00:00:00', '2023-08-01 23:59:59'],
    ['2023-09-20 00:00:00', '2023-09-26 23:59:59']], my_batch_size=32, train_test_split_flag=False,
    average_table_name='overall_value_sensor')
    -------
    """

    # 若可以，使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 所有数据集，包含Train和Test
    my_res_X, my_res_X_scaled = \
        grottoes_data_process.read_class_sensor_data(class_sensor_list=class_sensor_list,
                                                     data_path=Data_path,
                                                     columns_name_fliter=my_columns_name_fliter,
                                                     season_attribute=my_season_attribute,
                                                     standard_flag=True)
    # 所有数据集的target，包含Train和Test
    my_res_Y = grottoes_data_process.overall_value_data(overall_value_table_name=average_table_name,
                                                        data_path=Data_path,
                                                        columns_name_fliter=my_columns_name_fliter,
                                                        season_attribute=my_season_attribute)

    # 0均值归一化
    if std_flag:
        my_res_X = my_res_X_scaled
    else:
        pass

    # pd.set_option('display.max_columns', None)
    print(my_res_X)
    print(my_res_X_scaled)
    print(my_res_Y)

    # 按配置中测试的周来划分训练集、验证集和测试集
    res_train_X = my_res_X
    res_train_Y = my_res_Y
    res_validate_X = None
    res_validate_Y = None
    res_test_X = None
    res_test_Y = None

    # 划分出测试集
    for z in test_week_list:
        # 划分训练集
        res_train_X = res_train_X.drop(res_train_X[(res_train_X['time'] >= z[0]) & (res_train_X['time'] <= z[1])].index)
        res_train_Y = res_train_Y.drop(res_train_Y[(res_train_Y['time'] >= z[0]) & (res_train_Y['time'] <= z[1])].index)
        # 划分测试集
        res_test_X_temp = my_res_X.loc[(my_res_X['time'] >= z[0]) & (my_res_X['time'] <= z[1])]
        res_test_Y_temp = my_res_Y.loc[(my_res_Y['time'] >= z[0]) & (my_res_Y['time'] <= z[1])]
        # 合并
        if z == test_week_list[0]:
            res_test_X = res_test_X_temp
            res_test_Y = res_test_Y_temp
        else:
            res_test_X = pd.concat([res_test_X, res_test_X_temp], axis=0)
            res_test_Y = pd.concat([res_test_Y, res_test_Y_temp], axis=0)

    # 划分出验证集
    if train_test_split_flag:
        res_train_X, res_validate_X, res_train_Y, res_validate_Y = train_test_split(
            res_train_X, res_train_Y, test_size=0.222, random_state=20, shuffle=True)
    else:
        for z in validate_week_list:
            # 划分训练集
            res_train_X = res_train_X.drop(
                res_train_X[(res_train_X['time'] >= z[0]) & (res_train_X['time'] <= z[1])].index)
            res_train_Y = res_train_Y.drop(
                res_train_Y[(res_train_Y['time'] >= z[0]) & (res_train_Y['time'] <= z[1])].index)

            # 划分验证集
            res_validate_X_temp = my_res_X.loc[(my_res_X['time'] >= z[0]) & (my_res_X['time'] <= z[1])]
            res_validate_Y_temp = my_res_Y.loc[(my_res_Y['time'] >= z[0]) & (my_res_Y['time'] <= z[1])]

            # 合并
            if z == validate_week_list[0]:
                res_validate_X = res_validate_X_temp
                res_validate_Y = res_validate_Y_temp
            else:
                res_validate_X = pd.concat([res_validate_X, res_validate_X_temp], axis=0)
                res_validate_Y = pd.concat([res_validate_Y, res_validate_Y_temp], axis=0)

    print('After dividing: ')
    print('X_train: ')
    print(res_train_X)
    print('*' * 25)
    print('Y_train: ')
    print(res_train_Y)
    print('-' * 50)
    print('X_validate: ')
    print(res_validate_X)
    print('*' * 25)
    print('Y_validate: ')
    print(res_validate_Y)
    print('-' * 50)
    print('X_test: ')
    print(res_test_X)
    print('*' * 25)
    print('Y_test: ')
    print(res_test_Y)
    print('-' * 50)

    # 转换为Tensor格式
    res_train_X_tensor = torch.tensor(res_train_X.iloc[:, 1:].values, dtype=torch.float32, device=device)
    res_train_Y_tensor = torch.tensor(res_train_Y.iloc[:, 1:].values, dtype=torch.float32, device=device)
    if validate_week_list:
        res_validate_X_tensor = torch.tensor(res_validate_X.iloc[:, 1:].values, dtype=torch.float32, device=device)
        res_validate_Y_tensor = torch.tensor(res_validate_Y.iloc[:, 1:].values, dtype=torch.float32, device=device)
    else:
        res_validate_X_tensor = torch.Tensor()
        res_validate_Y_tensor = torch.Tensor()
    res_test_X_tensor = torch.tensor(res_test_X.iloc[:, 1:].values, dtype=torch.float32, device=device)
    res_test_Y_tensor = torch.tensor(res_test_Y.iloc[:, 1:].values, dtype=torch.float32, device=device)

    # 构建打包成张量数据集
    train_dataset = TensorDataset(res_train_X_tensor, res_train_Y_tensor)
    trainDataLoader = DataLoader(dataset=train_dataset, batch_size=my_batch_size, shuffle=True, drop_last=True)
    if validate_week_list:
        validate_dataset = TensorDataset(res_validate_X_tensor, res_validate_Y_tensor)
        validateDataLoader = DataLoader(dataset=validate_dataset, batch_size=my_batch_size, shuffle=True,
                                        drop_last=True)
    else:
        validate_dataset = None
        validateDataLoader = None
    test_dataset = TensorDataset(res_test_X_tensor, res_test_Y_tensor)
    testDataLoader = DataLoader(dataset=test_dataset, batch_size=my_batch_size, shuffle=True, drop_last=True)

    # 全部打包成dict
    res_data_dict = {'train_input': res_train_X,
                     'train_label': res_train_Y,
                     'validate_input': res_validate_X,
                     'validate_label': res_validate_Y,
                     'test_input': res_test_X,
                     'test_label': res_test_Y}

    res_TensorDataset_dict = {'train_tensor_dataset': train_dataset,
                              'validate_tensor_dataset': validate_dataset,
                              'test_tensor_dataset': test_dataset}

    res_DataLoader_dict = {'train_DataLoader': trainDataLoader,
                           'validate_DataLoader': validateDataLoader,
                           'test_DataLoader': testDataLoader}

    res_data_tensor_dict = {'train_input': res_train_X_tensor.to(device=device),
                            'train_label': res_train_Y_tensor.to(device=device),
                            'validate_input': res_validate_X_tensor.to(device=device),
                            'validate_label': res_validate_Y_tensor.to(device=device),
                            'test_input': res_test_X_tensor.to(device=device),
                            'test_label': res_test_Y_tensor.to(device=device)}

    return res_data_dict, res_TensorDataset_dict, res_DataLoader_dict, res_data_tensor_dict
