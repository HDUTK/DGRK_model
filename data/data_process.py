#!/usr/bin/env python3.6.8
# -*- coding: utf-8 -*-
"""
function description: 此文件用于数据处理
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2021.2.3
time: 2022/3/22 22:58
version: V1.0
"""
import os
import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import decimal
from common.common_func import path_and_name_exist


class data_process_func(object):
    """
    数据处理
    """
    def __init__(self):
        pass

    @staticmethod
    def load_csv_dataset(local_address, header=None, sep=',', **kwargs):
        """
        数据导入csv
        :param local_address:本地dataset地址，csv格式，最好使用/ 不使用\
        :param header:用作列名的行，默认为None，若有一般为第0行
        :param sep:对各行进行拆分的字符序列或正则表达式
        :return:返回dataset，dataframe格式
        :param kwargs: 其他参数，参数parse_dates指定需要解析为datetime类型的列
        pd.read_csv('data.csv', parse_dates=['time_column'])
        """
        f = open(local_address, encoding="utf-8")
        parse_dates = kwargs.get("parse_dates", False)
        content = pd.read_csv(f, header=header, sep=sep, parse_dates=parse_dates)
        return content

    @staticmethod
    def load_txt_dataset(local_address, header=None, sep='\s+'):
        """
        数据导入txt
        :param local_address:本地dataset地址，txt格式，最好使用/ 不使用\
        :param header:用作列名的行，默认为None，若有一般为第0行
        :param sep:对各行进行拆分的字符序列或正则表达式 默认同时匹配任何空白字符
        :return:返回dataset，dataframe格式
        """
        f = open(local_address, encoding="utf-8")
        content = pd.read_table(f, header=header, sep=sep)

        return content

    @staticmethod
    def from_database_to_dataframe(database_data):
        """
        将database取出的数据转化为Dataframe
        并将decimal类型的数据，化为numpy.float64
        其中database取出数据格式为[{},{},{}]，一个字典一个数据
        :param database_data:database取出的数据
        :return:返回Dataframe格式的数据
        """
        database_res = pd.DataFrame(database_data)

        # 处理decimal类型的数据，化为numpy.float64
        database_data_0 = database_res.iloc[0, :]
        for j in range(len(database_data_0)):
            if isinstance(database_data_0[j], decimal.Decimal):
                database_res.iloc[:, j] = database_res.iloc[:, j].astype(np.float64)

        return database_res

    @staticmethod
    def normalize_data(x):
        """
        数据归一化处理
        :param x:待归一化的特征矩阵，一行一样本，列为特征列，m*n
        :return:归一化后的特征矩阵，一行一样本，列为特征列，m*n
        """
        # 均值和标准差计算
        xmean = np.mean(x, axis=0)
        xstd = np.std(x, axis=0)
        # 归一化
        x = (x - xmean) / xstd

        return x

    @staticmethod
    def data_cov(x):
        """
        求x的协方差矩阵
        :param x:特征矩阵，一行一样本，列为特征列，m*n
        :return:x的协方差矩阵
        """
        # 按列取均值
        xmean = np.mean(x, axis=0)
        # 数据中心化
        zeroCentred_data = x - xmean

        # 计算协方差矩阵 rowvar=False表示数据的每一列代表一个feature
        x_cov = np.cov(zeroCentred_data, rowvar=False)

        return x_cov

    @staticmethod
    def calculate_entropy(y):
        """
        函数功能：计算熵
        :param y: 数据集的标签
        :return:x的协方差矩阵
        """
        num = y.shape[0]
        # 统计y中不同label值的个数，并用字典labelCounts存储
        labelCounts = {}
        for label in y:
            if label not in labelCounts.keys():
                labelCounts[label] = 0
            labelCounts[label] = labelCounts[label] + 1
        # 计算熵
        entropy = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key])/num
            entropy = entropy - prob * np.log2(prob)
        return entropy

    @staticmethod
    def add_col1_to_array(x):
        """
        将偏置列全1 加入第一列
        :param x:x行样本，一行一样本，列为特征列，m*n
        :return:x行样本，一行一样本，列为特征列，m*(n+1)
        """
        # m行个样本、n列个特征
        m, n = x.shape
        # 在第一列 将全为1加入
        const = np.array([1] * m).reshape(m, 1)
        x_add_col1 = np.append(const, x, axis=1)

        return x_add_col1

    @staticmethod
    def read_txt(read_path, read_all=True, read_mode='r', read_lines=-1, split_str='\t', right_strip_str='\n'):
        """
        读取txt文件
        :param read_path: txt文件目录
        :param read_all: 是否读取全部数据标志位
        :param read_mode: 读取模式
        :param read_lines: 前几行数据，当不是读取全部数据时必输
        :param split_str: 每一行以split_str字符分割，默认\t
        :param right_strip_str: 每行末尾都去掉right_strip_str字符，默认\n
        :return: 返回读取结果，为List
        """
        if not os.path.exists(read_path):
            print('The directory is not exist!')
            exit(1)
        else:
            pass

        if (read_all is False) and read_lines == -1:
            print('Please enter the number of rows!')
            exit(1)
        else:
            pass

        with open(read_path, mode=read_mode, encoding='utf-8') as f:
            data = f.readlines()
            res = []
            cnt = 0
            for line in data:
                cnt = cnt + 1
                temp = (line.rstrip(right_strip_str)).split(split_str)  # 去尾的right_strip_str字符，以split_str字符分割
                res.append(temp)
                if (read_all is False) and cnt == read_lines:
                    break
                else:
                    continue
            return res

    @staticmethod
    def list_transform_to_dataframe(list_data, column_name=None):
        """
        将List转换为Dataframe格式
        :param list_data: 代入List
        :param column_name: Dataframe的列名
        :return: 返回创建的Dataframe
        """
        if (column_name is not None) and (not isinstance(column_name, list)):
            print('Please assign list to column_name!')
            exit(1)
        elif (column_name is not None) and (len(column_name) != len(list_data[0])):
            print('The length of the column_name isn\'t equal to the list_data!')
            exit(1)
        else:
            pass
        res = pd.DataFrame(list_data, columns=column_name)

        return res

    @staticmethod
    def std_EEG_data_row(Dataframe_data, rows_columns_flag=True):
        """
        将Dataframe数据按行标准化
        :param Dataframe_data: 输入数据，格式为Dataframe
        :param rows_columns_flag: 行列标志，默认对行操作
        :return: 返回Dataframe格式的标准化后的数据
        """
        if rows_columns_flag:
            Dataframe_data = Dataframe_data.T
        else:
            pass
        std_data = StandardScaler()
        Dataframe_data = pd.DataFrame((std_data.fit_transform(Dataframe_data)).T)
        return Dataframe_data

    @staticmethod
    def std_dataframe(Dataframe_data, rows_columns_flag=True):
        """
        将Dataframe数据标准化  待测
        :param Dataframe_data: 输入数据，格式为Dataframe
        :param rows_columns_flag: 行列标志，默认对行操作
        :return: 返回Dataframe格式的标准化后的数据
        """
        if rows_columns_flag:
            Dataframe_data = Dataframe_data.T
        else:
            pass
        std_data = StandardScaler()
        if rows_columns_flag:
            Dataframe_data = pd.DataFrame((std_data.fit_transform(Dataframe_data)).T)
        else:
            Dataframe_data = pd.DataFrame((std_data.fit_transform(Dataframe_data)))

        return Dataframe_data

    @staticmethod
    def str_label_to_OneHot_code_label(str_label: str):
        """
        将字符形式的label转换为onehot编码格式的label
        :param str_label: 字符形式的label，例如'是否'、'ABC'
        :return: 返回Dataframe格式的label，按列排列，取的时候按列名取
           是  否
        0  1  0
        1  0  1
        """
        s = pd.Series(list(str_label))
        OneHot_code = pd.get_dummies(s, sparse=True)
        return OneHot_code

    @staticmethod
    def print_1dim_list_to_txt(one_dim_list: list, path_and_txt_name, style_type='1', split_str='\n', header=''):
        """
        1维列表输出至txt
        注意：txt文件的行数限定2000行，列数最多1400列
        :param one_dim_list: 1维列表
        :param path_and_txt_name: 路径+文件名
        :param style_type: 样式类型，默认为样式1
        （样式1：['A', 'B', 'C', 'D', 1, 2, 3]
         样式2（若分隔符是是\n）：A
                            B
                            C
                            D）
        :param split_str: 若样式类型为2时，写入txt时的行内分隔符，默认'\n'
        :param header: 标题行，默认空，例如以'name\tgender\tstatus\tage\n'为格式
        :return:
        """
        # 确认绝对路径下文件夹是否存在，若不存在则创建文件夹（路径的斜杠必须用/）
        path_and_name_exist(path_and_txt_name)

        with open(path_and_txt_name, 'w') as f:
            if style_type == '1':
                f.write(str(one_dim_list))
            elif style_type == '2':
                f.write(split_str.join([str(i) for i in one_dim_list]))
            else:
                f.write('Some Errors Have Occurred!')
            f.close()

        return

    @staticmethod
    def print_2dim_list_to_xls(two_dim_list: list, path_and_xls_name, header=''):
        """
        2维列表输出至xls
        注意：xls文件的行数限定65536行，列数最多256列
        :param two_dim_list: 2维列表
        :param path_and_xls_name: 路径+文件名
        :param header: 标题行，默认空，例如以'name\tgender\tstatus\tage\n'为格式
        :return: 输出xls文件
        """
        # 确认绝对路径下文件夹是否存在，若不存在则创建文件夹（路径的斜杠必须用/）
        path_and_name_exist(path_and_xls_name)

        with open(path_and_xls_name, 'w', encoding='gbk') as output:
            # 若有标题行
            if header != '':
                output.write(header)
            else:
                pass

            for i in range(len(two_dim_list)):
                for j in range(len(two_dim_list[i])):
                    output.write(str(two_dim_list[i][j]))
                    output.write('\t')
                output.write('\n')
            output.close()
            pass
        return

    @staticmethod
    def print_2dim_list_to_txt(two_dim_list: list, path_and_txt_name, header='', split_str=' '):
        """
        2维列表输出至txt
        注意：txt文件的行数限定2000行，列数最多1400列
        :param two_dim_list: 2维列表
        :param path_and_txt_name: 路径+文件名
        :param header: 标题行，默认空，例如以'name\tgender\tstatus\tage\n'为格式
        :param split_str: 写入txt时，行内的分隔符，默认' '
        :return:
        """
        # 确认绝对路径下文件夹是否存在，若不存在则创建文件夹（路径的斜杠必须用/）
        path_and_name_exist(path_and_txt_name)

        with open(path_and_txt_name, 'w') as f:
            # 若有标题行
            if header != '':
                f.write(header)
            else:
                pass

            for i in two_dim_list:
                for j in i:
                    f.write(str(j))
                    f.write(split_str)
                f.write('\n')
            f.close()

        pass

    @staticmethod
    def print_2dim_list_to_csv(two_dim_list: list, path_and_csv_name, header=None):
        """
        2维列表输出至csv
        csv文件没有最大行数和列数的限制
        :param two_dim_list: 2维列表
        :param path_and_csv_name: 路径+文件名
        :param header: 标题行，默认None，例如以('标题', '租金', '付款方式')或['标题', '租金', '付款方式']为格式
        :return:
        """
        # 确认绝对路径下文件夹是否存在，若不存在则创建文件夹（路径的斜杠必须用/）
        path_and_name_exist(path_and_csv_name)

        with open(path_and_csv_name, 'w', newline='') as f:

            writer = csv.writer(f)
            # 若有标题行
            if header is not None:
                writer.writerow(header)
            else:
                pass

            for i in two_dim_list:
                writer.writerow(i)
            f.close()

        pass

