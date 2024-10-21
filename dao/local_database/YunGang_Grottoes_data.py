#!/usr/bin/env python3.8.10
# -*- coding: utf-8 -*-
"""
function description: 此文件用于云冈石窟的数据集
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2020.2.5
time: 2024/3/26 15:12
version: V1.0
"""
import pandas as pd
import datetime
import time
from tqdm import tqdm
import numpy as np

from dao.database_util import DatabaseUtil
from common.DButil import DButil
from common.common_func import dir_path_exist_or_makedir
from config.system_config import SYS_CONFIG
from config.Yungang_Grottoes_config import *
from dao.table_process import table_process
from data.data_process import data_process_func


class table_info(object):
    """
    功能
    """

    def __init__(self):
        """
        初始化表名并连接数据库
        """
        self.table = ""  # 需要赋值
        self.database = "dataset"
        self.database_name = SYS_CONFIG["local_database"]["默认使用的数据库名"]
        self.db_session = DButil.get_db_session(SYS_CONFIG["local_database"]["数据库总名"])

    def __del__(self):
        """
        释放数据库链接
        """
        self.db_session.close()

    def use_database(self, database_name, sql_print=True):
        """
        选择使用的数据库
        :param database_name: 使用的数据库名
        :param sql_print: 是否打印运行的sql
        :return:
        """
        # 若database_name不为空，则传入数据库名
        if database_name != '':
            self.database_name = database_name

        try:
            sql_use_database = 'use %s' % self.database_name
            DatabaseUtil.exec_sql(self.db_session, sql_use_database, "use", sql_print=sql_print)
        except Exception as e:
            # 画图失败
            print("选择使用数据库失败，失败原因是：", e.args)
            return

    def execute_select_sql(self, sql, database_name='', result_to_dataframe=False, sql_print=True):
        """
        执行select的复杂sql（可能涉及联表查询） 返回数据
        :param sql: 需要执行的select的复杂sql
        :param database_name: 使用的数据库名
        默认为SYS_CONFIG["local_database"]["默认使用的数据库名"]
        :param result_to_dataframe: 若使用select，是否将结果输出为dataframe
        :param sql_print: 是否打印运行的sql
        :return:
        """
        # 选择使用的数据库
        self.use_database(database_name, sql_print=sql_print)

        execute_select_result = DatabaseUtil.exec_sql(self.db_session, sql, "select",
                                                      result_to_dataframe=result_to_dataframe,
                                                      sql_print=sql_print)

        return execute_select_result

    def insert_data(self, data_list: list, now_location_str=None, insert_str='',
                    database_name='', sql_print=True):
        """
        功能 插入数据
        :param data_list: 形式为[[],[],[]]，行数据，列特征
        :param now_location_str: 目前的传感器位置，若不为None，则表示将该传感器数据一次性全部insert
        :param insert_str: 待插入的字段，首列默认时间time，格式为'temperature,humidity'
        :param database_name: 使用的数据库名
        默认为SYS_CONFIG["local_database"]['"默认使用的数据库名"']
        :param sql_print: 是否打印运行的sql
        :return: 记录
        """
        # 选择使用的数据库
        self.use_database(database_name=database_name, sql_print=sql_print)

        # 以2500行来逐行插入
        for k in range(int(len(data_list) / 2500) + 1):
            data_list_temp = data_list[2500 * k: min(2500 * (k + 1), len(data_list))]

            if not data_list_temp:
                continue
            else:
                pass

            if now_location_str is None:
                sql = ("INSERT INTO %s " % self.table) + "(time, %s) VALUES (" % insert_str
            else:
                sql = ("INSERT INTO %s " % self.table) + "(time, "
                for p in (config_dict[now_location_str]):
                    sql = sql + (sensor_type_dict[str(p)])[1] + ", "
                sql = sql[:-2] + ") VALUES ("

            for i in range(len(data_list_temp)):
                for j in range(len(data_list_temp[i])):
                    if isinstance(data_list_temp[i][j], str) or isinstance(data_list_temp[i][j], datetime.datetime):
                        sql = sql + "'" + str(data_list_temp[i][j]) + "',"
                    else:
                        sql = sql + str(data_list_temp[i][j]) + ','
                sql = sql[:-1] + '),('
            sql = sql[:-2] + ';'

            DatabaseUtil.exec_sql(self.db_session, sql.replace("None", "NULL"), "insert", sql_print=sql_print)

        return

    def update_data(self, data_list: list, update_str: str, database_name='', sql_print=True):
        """
        功能 使用时间更新数据
        :param data_list: 形式为[[时间,值,...],[时间,值,...],[时间,值,...]]，行数据，列特征
        :param update_str: 待更新的字段
        :param database_name: 使用的数据库名
        默认为SYS_CONFIG["local_database"]['"默认使用的数据库名"']
        :param sql_print: 是否打印运行的sql
        :return: 记录
        """
        # 选择使用的数据库
        self.use_database(database_name=database_name, sql_print=sql_print)

        # 以2500行来逐行更新
        for k in range(int(len(data_list) / 2500) + 1):
            data_list_temp = data_list[2500 * k: min(2500 * (k + 1), len(data_list))]

            sql = ("UPDATE %s SET " % self.table) + update_str + " = CASE "

            # data_list_temp = [[时间,值,...],[时间,值,...],[时间,值,...]]*2500
            for i in range(len(data_list_temp)):  # data_list_temp[i] = [时间,值,...]
                if isinstance(data_list_temp[i][0], str):
                    sql = sql + "WHEN time = " + "'" + str(data_list_temp[i][0]) + "' "
                else:
                    sql = sql + "WHEN time = " + str(data_list_temp[i][0]) + " "

                if isinstance(data_list_temp[i][1], str) or isinstance(data_list_temp[i][1], datetime.datetime):
                    sql = sql + "THEN " + "'" + str(data_list_temp[i][1]) + "' "
                else:
                    sql = sql + "THEN " + str(data_list_temp[i][1]) + " "

            sql = sql + "ELSE " + update_str + " END;"

            DatabaseUtil.exec_sql(self.db_session, sql.replace("None", "NULL"), "update", sql_print=sql_print)

        return

    def select_list_all(self, database_name=''):
        """
        功能 查询全部数据
        :param database_name: 使用的数据库名
        默认为SYS_CONFIG["local_database"]['"默认使用的数据库名"']
        :return: 记录
        """
        # 选择使用的数据库
        self.use_database(database_name=database_name)

        sql = "SELECT * FROM %s" % self.table
        return DatabaseUtil.exec_sql(self.db_session, sql, "select")

    def select_table_name_list(self, database_name=''):
        """
        功能 查询全部表名
        :param database_name: 使用的数据库名
        默认为SYS_CONFIG["local_database"]['"默认使用的数据库名"']
        :return: 记录
        """
        # 选择使用的数据库
        self.use_database(database_name=database_name)

        sql = "SELECT table_name FROM information_schema.tables where TABLE_SCHEMA = '%s'" % self.database_name
        return DatabaseUtil.exec_sql(self.db_session, sql, "select")

    def select_time_last_col(self, now_location_str, database_name='', sql_print=True):
        """
        功能 按时间查询最后第一行
        :param now_location_str: 现在要操作的传感器位置
        :param database_name: 使用的数据库名
        默认为SYS_CONFIG["local_database"]['"默认使用的数据库名"']
        :param sql_print: 是否打印运行的sql
        :return: 记录
        """
        # 选择使用的数据库
        self.use_database(database_name=database_name, sql_print=sql_print)

        sql = "SELECT * FROM %s ORDER BY time DESC LIMIT 1" % (now_location_str + '_sensor')

        return DatabaseUtil.exec_sql(self.db_session, sql, "select", sql_print=sql_print)

    def delete_time_last_col(self, now_location_str, database_name='', sql_print=True):
        """
        功能 删除最后一行数据
        :param now_location_str: 现在要操作的传感器位置
        :param database_name: 使用的数据库名
        默认为SYS_CONFIG["local_database"]['"默认使用的数据库名"']
        :param sql_print: 是否打印运行的sql
        :return: 记录
        """
        # 选择使用的数据库
        self.use_database(database_name=database_name, sql_print=sql_print)

        sql = "DELETE FROM %s WHERE 1 ORDER BY time DESC LIMIT 1" % (now_location_str + '_sensor')

        return DatabaseUtil.exec_sql(self.db_session, sql, "delete", sql_print=sql_print)

    def reset_id(self, now_location_str: str, database_name='', sql_print=True):
        """
        重置数据表的自增id
        :param now_location_str: 需要重置自增id的传感器位置
        :param database_name: 使用的数据库名
        默认为SYS_CONFIG["local_database"]["默认使用的数据库名"]
        :param sql_print: 是否打印运行的sql
        :return:
        """
        # 选择使用的数据库
        self.use_database(database_name, sql_print=sql_print)

        sql = 'ALTER TABLE %s AUTO_INCREMENT = 1;' % (now_location_str + '_sensor')
        DatabaseUtil.exec_sql(self.db_session, sql, "alter", sql_print=sql_print)

        return

    # def select_list_by_nbr(self, nbr: str, database_name=''):
    #     """
    #     功能 查询
    #     :param nbr: 流水号
    #     :param database_name: 使用的数据库名
    #     默认为SYS_CONFIG["local_database"]['"默认使用的数据库名"']
    #     :return: 记录
    #     """
    #     # 选择使用的数据库
    #     self.use_database(database_name=database_name)
    #
    #     sql = "SELECT * FROM %s WHERE nbr='%s' ORDER BY create_time DESC" % (self.table, nbr)
    #     return DatabaseUtil.exec_sql(self.db_session, sql, "select")


def create_table(now_location_str, database_name):
    """
    创建数据表
    :param now_location_str: 传感器位置字符串
    :param database_name: 使用的数据库名
    :return:
    """

    print('*' * 30)
    print('Creating table if this table not exist, table name:  ' + now_location_str + '_Sensor')

    # 新建数据表
    datatable_process = table_process()
    # 创建数据表使用的字典 {'列名': [类型, 是否可以为空, 默认值, 注释]}
    mysql_dict = {'0': {'id': ['int', 'NOT NULL', '自动新增_数字类型', 'ID'],
                        'time': ['Datetime', 'NULL', None, '日期时间年月日时：分：秒']},
                  '1': {'air_temperature': ['decimal(10,3)', 'NULL', None, '空气温度（℃）']},
                  '2': {'air_humidity': ['decimal(10,3)', 'NULL', None, '空气湿度（%）']},
                  '3': {'wall_temperature': ['decimal(10,3)', 'NULL', None, '壁面温度（℃）']},
                  '4': {'wind_speed': ['decimal(10,3)', 'NULL', None, '风速（m/s）']},
                  '5': {'wind_direction': ['decimal(10,3)', 'NULL', None, '风向（°）']},
                  '6': {'atmos': ['decimal(10,3)', 'NULL', None, '大气压（Pa）']},
                  '7': {'annual_cumulative_rainfall': ['decimal(10,3)', 'NULL', None, '年累积雨量（mm）']},
                  '8': {'pressure': ['decimal(10,3)', 'NULL', None, '气压（Pa）']},
                  '9': {'hourly_rainfall': ['decimal(10,3)', 'NULL', None, '小时雨量（mm）']},
                  '10': {'total_radiation': ['decimal(10,3)', 'NULL', None, '总辐射（J/m^2）']},
                  }

    # 找到该传感器位置所测量的指标
    dict_temp = {}
    for k in range(len(config_dict[now_location_str])):
        dict_temp = {**dict_temp, **mysql_dict[str(config_dict[now_location_str][k])]}

    # 创建表
    datatable_process.create_table(now_location_str + '_Sensor', {**mysql_dict['0'], **dict_temp},
                                   'id', database_name)

    print('Finish operation, table name:  ' + now_location_str + '_Sensor !')
    print('*' * 30)

    return


def sensor_location(now_location_str, sensor_operate, sensor_type_int, database_name):
    """
    根据传感器位置存入数据
    :param now_location_str: 传感器位置字符串
    :param sensor_operate: 操作名称，'insert'或者'update'
    :param sensor_type_int: 针对哪一个指标进行操作，根据指标字典传入数字
    :param database_name: 使用的数据库名
    :return:
    """
    # 找到目标文件并读取，同时处理NULL值
    try:
        f = Data_path + (sensor_type_dict[str(sensor_type_int)])[0] + '-' + now_location_str + '监测值.CSV'
        content = data_process_func.load_csv_dataset(f, header=0)
        # 删除全空行
        content = content.dropna(how='all')
        # 处理NULL值
        content = content.astype(object).where(pd.notnull(content), None)

    except Exception as e:
        print('Some errors received during reading data in sensor:  ' + now_location_str
              + '  ' + (sensor_type_dict[str(sensor_type_int)])[0] + ' . Concrete Problem: ', e)
        return

    table = table_info()
    table.table = now_location_str + '_Sensor'
    table_list = table_process.make_multi_insert_list(content)

    if sensor_operate == 'insert':
        print('-' * 50)
        print('Inserting data, table name:  ' + now_location_str + '_Sensor, ' +
              (sensor_type_dict[str(sensor_type_int)])[0])

        table.insert_data(table_list, insert_str=(sensor_type_dict[str(sensor_type_int)])[1],
                          database_name=database_name, sql_print=False)

        print('Finish insert, table name:  ' + now_location_str + '_Sensor, ' +
              (sensor_type_dict[str(sensor_type_int)])[0])
        print('-' * 50)
    elif sensor_operate == 'update':
        print('-' * 50)
        print('Updating data, table name:  ' + now_location_str + '_Sensor, ' +
              (sensor_type_dict[str(sensor_type_int)])[0])

        table.update_data(table_list, (sensor_type_dict[str(sensor_type_int)])[1],
                          database_name=database_name, sql_print=False)

        print('Finish update, table name:  ' + now_location_str + '_Sensor, ' +
              (sensor_type_dict[str(sensor_type_int)])[0])
        print('-' * 50)
    else:
        print('The Sensor Operate is wrong during reading data in sensor:  ' + now_location_str
              + '  ' + (sensor_type_dict[str(sensor_type_int)])[0] + ' .')
    return


def sensor_location_insert_all(now_location_str, database_name):
    """
    根据传感器位置存入数据（一次性）
    :param now_location_str: 传感器位置字符串
    :param database_name: 使用的数据库名
    :return:
    """
    # csv文件的初始时间和终止时间作为校验，以后可能更新的时候也有用
    start_time = None
    end_time = None

    for m in range(len(config_dict[now_location_str])):

        # 操作完一次后隔一段时间再进行操作（second）
        time.sleep(2)

        try:
            f = Data_path + (sensor_type_dict[str(config_dict[now_location_str][m])])[0] \
                + '-' + now_location_str + '监测值.CSV'
            content_temp = data_process_func.load_csv_dataset(f, header=0, parse_dates=['时间'])
            # 删除全空行
            content_temp = content_temp.dropna(how='all')

            # 设索引为时间
            content_temp = content_temp.set_index('时间')

            # 校验是不是第一次，若是则赋值start_time和end_time进行校验，并初始化content
            if (start_time is None) and (end_time is None):
                start_time = content_temp.iloc[0, 0]
                end_time = content_temp.iloc[-1, 0]
                content = content_temp
            # 若不是第一次则进行合并
            else:
                content = pd.merge(content, content_temp, on='时间', how='outer')

        except Exception as e:
            print('Some errors received during reading data in sensor:  ' + now_location_str + '  '
                  + (sensor_type_dict[str(config_dict[now_location_str][m])])[0] + ' . Concrete Problem: ', e)
            return

    # 按时间升序
    content = content.sort_index()

    # 重置索引
    content = content.reset_index()

    # 处理NULL值
    content = content.astype(object).where(pd.notnull(content), None)

    # 插入数据
    table = table_info()
    table.table = now_location_str + '_Sensor'
    table_list = table_process.make_multi_insert_list(content)

    print('-' * 50)
    print('Inserting data, table name:  ' + now_location_str + '_Sensor')

    table.insert_data(table_list, now_location_str=now_location_str,
                      database_name=database_name, sql_print=False)

    print('Finish insert, table name:  ' + now_location_str + '_Sensor')
    print('-' * 50)

    return


def sensor_location_update_all(now_location_str, database_name):
    """
    根据传感器位置更新数据（一次性）
    :param now_location_str: 传感器位置字符串
    :param database_name: 使用的数据库名
    :return:
    """
    # 获取数据库全部表名
    table = table_info()
    exist_sensors = table.select_table_name_list(database_name=database_name)

    if (now_location_str.lower() + '_sensor') not in exist_sensors:
        print("The " + now_location_str + "_sensor is not exist! Can't update! Please Create at first!")
        return
    else:
        pass

    # csv文件的初始时间作为是否为初次的校验
    start = True
    # 取出末行
    last_col = table.select_time_last_col(now_location_str, database_name=database_name, sql_print=False)
    # 检查last_col的元组内有没有None值，有则需要根据time进一步查找缺失数据的最后时间（删除最后一行），没有则继续
    print('-' * 10)
    print("Checking for missing data at the end in  " + now_location_str + "_sensor  ! IF EXIST, DELETE! ")
    while any(element is None for element in last_col[0]):
        table.delete_time_last_col(now_location_str, database_name=database_name, sql_print=False)
        # 操作完一次后隔一段时间再进行操作（second）
        time.sleep(1)
        last_col = table.select_time_last_col(now_location_str, database_name=database_name, sql_print=False)
    # 重置自增ID
    table.reset_id(now_location_str, database_name=database_name, sql_print=False)
    print("Checking Over!")
    print('-' * 10)

    for n in range(len(config_dict[now_location_str])):

        # 操作完一次后隔一段时间再进行操作（second）
        time.sleep(3)

        try:
            f = Data_path + (sensor_type_dict[str(config_dict[now_location_str][n])])[0] \
                + '-' + now_location_str + '监测值.CSV'
            content_temp = data_process_func.load_csv_dataset(f, header=0, parse_dates=['时间'])
            # 删除全空行
            content_temp = content_temp.dropna(how='all')

            # 设索引为时间
            content_temp = content_temp.set_index('时间')

            # 选择末行时间点及之后的数据
            content_temp = content_temp[(content_temp['时间'] >= str(last_col[0][1]))]

            # 校验是不是第一次，若是则赋值start_time和end_time进行校验，并初始化content
            if start:
                start = False
                content = content_temp
            # 若不是第一次则合并
            else:
                content = pd.merge(content, content_temp, on='时间', how='outer')

        except Exception as e:
            print('Some errors received during reading data in sensor:  ' + now_location_str + '  '
                  + (sensor_type_dict[str(config_dict[now_location_str][n])])[0] + ' . Concrete Problem: ', e)
            return

    # 按时间升序
    content = content.sort_index()

    # 重置索引
    content = content.reset_index()

    # 处理NULL值
    content = (content.iloc[1:]).astype(object).where(pd.notnull(content), None)

    # 插入数据
    table = table_info()
    table.table = now_location_str + '_Sensor'
    table_list = table_process.make_multi_insert_list(content)

    print('-' * 50)
    print('Inserting data, table name:  ' + now_location_str + '_Sensor')

    table.insert_data(table_list, now_location_str=now_location_str,
                      database_name=database_name, sql_print=False)

    print('Finish insert, table name:  ' + now_location_str + '_Sensor')
    print('-' * 50)

    return


def resample_between_time(now_location_str, resample_interval, time_between_list,
                          original_database, now_database, to_csv=False, to_csv_path='', **kwargs):
    """
    关于时间的重采样，填补上空缺的时间时刻
    :param now_location_str: 传感器位置字符串
    :param resample_interval: 时间间隔，为asfreq函数的一个参数
    :param time_between_list: 时间取值范围
    :param original_database: 原来的数据库
    :param now_database: 要插入的新的数据库
    :param to_csv: 是否要输出到csv文件保存
    :param to_csv_path: 若要输出到csv文件保存，则保存的path
    :param kwargs: 其他参数
    :return:
    """
    # 如果不存在表则创建表
    create_table(now_location_str, now_database)

    # 拼接sql并执行
    sql = "SELECT time, "
    for k in config_dict[now_location_str]:
        sql = sql + (sensor_type_dict[str(k)])[1] + ", "
    sql = sql[:-2] + " FROM " + now_location_str + "_sensor" + " WHERE time BETWEEN " + "'" + time_between_list[
        0] + "' AND '" + time_between_list[1] + "' ORDER BY time;"
    table = table_info()
    result_dataframe = table.execute_select_sql(sql, database_name=original_database,
                                                result_to_dataframe=True, sql_print=False)
    print('Select data from original database between ' + time_between_list[0] +
          ' and ' + time_between_list[1] + ' !')
    print('*' * 30)

    # 将时间轴设置为索引并进行重采样
    # result_dataframe = result_dataframe.resample(resample_interval, on='time')
    result_dataframe.set_index('time', inplace=True)
    result_dataframe = result_dataframe.asfreq(freq=resample_interval)

    # 将索引改为列
    result_dataframe = result_dataframe.reset_index()

    # 输出至CSV
    if to_csv:
        # 若不存在Path则创建
        dir_path_exist_or_makedir(to_csv_path)

        # 输出至CSV
        print('+' * 15)
        print('Dataframe to CSV in ' + now_location_str + ' START ! The Path: ' + to_csv_path +
              now_location_str + '.csv')

        try:
            result_dataframe.to_csv(to_csv_path + now_location_str + ".csv")
            print('Dataframe to CSV in ' + now_location_str + ' OVER ! The Path: ' + to_csv_path +
                  now_location_str + '.csv')
        except Exception as e:
            print('Some errors received during Dataframe to CSV. Concrete Problem: ', e)

        print('+' * 15)
    else:
        pass

    # 处理NULL值
    result_dataframe = (result_dataframe.iloc[:]).astype(object).where(pd.notnull(result_dataframe), None)

    # 插入数据
    table = table_info()
    table.table = now_location_str + '_Sensor'
    table_list = table_process.make_multi_insert_list(result_dataframe)

    print('-' * 50)
    print('Inserting data, table name:  ' + now_location_str + '_Sensor')

    table.insert_data(table_list, now_location_str=now_location_str,
                      database_name=now_database, sql_print=False)

    print('Finish insert, table name:  ' + now_location_str + '_Sensor')
    print('-' * 50)

    return


def interpolate_between_time(now_location_str, time_between_list, original_database,
                             now_database, to_csv=False, to_csv_path='', **kwargs):
    """
    关于时间的插值，填补上Nan的时间时刻
    :param now_location_str: 传感器位置字符串
    :param time_between_list: 时间取值范围
    :param original_database: 原来的数据库
    :param now_database: 要插入的新的数据库
    :param to_csv: 是否要输出到csv文件保存
    :param to_csv_path: 若要输出到csv文件保存，则保存的path
    :param kwargs: 其他参数
    :return:
    """
    # 如果不存在表则创建表
    create_table(now_location_str, now_database)

    # 拼接sql并执行
    sql = "SELECT time, "
    for k in config_dict[now_location_str]:
        sql = sql + "CAST(" + (sensor_type_dict[str(k)])[1] + " AS FLOAT) as " \
              + (sensor_type_dict[str(k)])[1] + ", "
    sql = sql[:-2] + " FROM " + now_location_str + "_sensor" + " WHERE time BETWEEN " + "'" + time_between_list[
        0] + "' AND '" + time_between_list[1] + "' ORDER BY time;"
    table = table_info()
    result_dataframe = table.execute_select_sql(sql, database_name=original_database,
                                                result_to_dataframe=True, sql_print=False)
    print('Select data from original database between ' + time_between_list[0] +
          ' and ' + time_between_list[1] + ' !')
    print('*' * 30)

    # 填充Nan值（可能出现在头与尾的情况），否则万一出现会报错
    if result_dataframe.iloc[0].isna().any():
        result_dataframe.fillna(method='bfill', inplace=True)
    else:
        pass
    if result_dataframe.iloc[-1].isna().any():
        result_dataframe.fillna(method='ffill', inplace=True)

    # 将时间轴设置为索引并进行插值
    result_dataframe = result_dataframe.set_index('time')
    result_dataframe = result_dataframe.interpolate(method='time')
    # 将索引改为列
    result_dataframe = result_dataframe.reset_index()

    # 输出至CSV
    if to_csv:
        # 若不存在Path则创建
        dir_path_exist_or_makedir(to_csv_path)

        # 输出至CSV
        print('+' * 15)
        print('Dataframe to CSV in ' + now_location_str + ' START ! The Path: ' + to_csv_path +
              now_location_str + '.csv')

        try:
            result_dataframe.to_csv(to_csv_path + now_location_str + ".csv")
            print('Dataframe to CSV in ' + now_location_str + ' OVER ! The Path: ' + to_csv_path +
                  now_location_str + '.csv')
        except Exception as e:
            print('Some errors received during Dataframe to CSV. Concrete Problem: ', e)

        print('+' * 15)
    else:
        pass

    # 插入数据
    table = table_info()
    table.table = now_location_str + '_Sensor'
    table_list = table_process.make_multi_insert_list(result_dataframe)

    print('-' * 50)
    print('Inserting data, table name:  ' + now_location_str + '_Sensor')

    table.insert_data(table_list, now_location_str=now_location_str,
                      database_name=now_database, sql_print=False)

    print('Finish insert, table name:  ' + now_location_str + '_Sensor')
    print('-' * 50)

    return


def get_entire_average_data(average_table_name, sensor_str_list: list, attribution_list: list,
                            database_name, to_csv=False, to_csv_path='', **kwargs):
    """
    得到平均数据作为整个SJD的整体数据
    :param average_table_name: 新建的表明
    :param sensor_str_list: 传感器名称列表（SJD_table_list）
    :param attribution_list: 属性名称列表
    :param database_name: 使用的数据库名
    :param to_csv: 是否要输出到csv文件保存
    :param to_csv_path: 若要输出到csv文件保存，则保存的path
    :param kwargs: 其他参数
    :return:
    """

    # 新建的表名
    entire_average_table_name = average_table_name

    table = table_info()
    datatable_process = table_process()

    # 如果不存在表则创建表
    col_param = {'id': ['int', 'NOT NULL', '自动新增_数字类型', 'ID'],
                 'time': ['Datetime', 'NOT NULL', '1000-01-01 00:00:00', '日期时间年月日时：分：秒']}
    for n in attribution_list:
        col_param[n + '_average'] = ['decimal(10,2)', 'NOT NULL', 0.00, '']
    datatable_process.create_table(table_name=entire_average_table_name,
                                   col_param=col_param, primary_key='id', database_name=database_name)

    res_dataframe = None
    for k in attribution_list:
        print('*' * 30)
        print('Select attribution data( ' + k + ' ) from database : ' + database_name + ' !')

        attribution_dataframe = None
        for m in tqdm(sensor_str_list):
            # 拼接sql并执行
            sql = "SELECT time, "
            sql = sql + k + " as " + m + "_" + k + " FROM " + m.lower() + ";"
            result_dataframe_temp = table.execute_select_sql(sql, database_name=database_name,
                                                             result_to_dataframe=True,
                                                             sql_print=False)
            if m == sensor_str_list[0]:
                attribution_dataframe = result_dataframe_temp
            else:
                attribution_dataframe = pd.merge(attribution_dataframe, result_dataframe_temp, on='time')

        attribution_dataframe[k + '_average'] = attribution_dataframe.iloc[:, 1:].mean(axis=1)

        print(k + ' Dataframe : ')
        print(attribution_dataframe)

        print('*' * 30)

        if k == attribution_list[0]:
            res_dataframe = attribution_dataframe[['time', k + '_average']]
        else:
            res_dataframe = pd.merge(res_dataframe, attribution_dataframe[['time', k + '_average']],
                                     on='time')

    print('Final Dataframe : ')
    print(res_dataframe)
    print('+' * 60)

    # 输出至CSV
    if to_csv:
        # 若不存在Path则创建
        dir_path_exist_or_makedir(to_csv_path)

        # 输出至CSV
        print('+' * 50)
        print('Dataframe to CSV START ! The Path: ' + to_csv_path + '_' +
              entire_average_table_name + '.csv')

        try:
            res_dataframe.to_csv(to_csv_path + '_' + entire_average_table_name + ".csv")
            print('Dataframe to CSV OVER ! The Path: ' + to_csv_path + '_' +
                  entire_average_table_name + '.csv')
        except Exception as e:
            print('Some errors received during Dataframe to CSV. Concrete Problem: ', e)

        print('+' * 50)

    # # 插入数据
    # table_list = table_process.make_multi_insert_list(res_dataframe)
    #
    # print('-' * 15)
    # print('Inserting data, table name:  ' + entire_average_table_name)
    #
    # # 赋值给table.table
    # table.table = entire_average_table_name
    # column_names = (res_dataframe.columns.tolist())[1:]
    # table.insert_data(table_list, insert_str=','.join(column_names), database_name=database_name,
    #                   sql_print=False)
    #
    # print('Finish insert, table name:  ' + 'overall_value_sensor')
    # print('-' * 15)


if __name__ == "__main__":
    pass

    # Path
    Data_path = r'D:/PythonProject/MachineLearning/My_Dataset/YunGang_Grottoes_Data/deal-24-6-25-year/'
    # 更新完一张表后隔一段时间再更新下一张表（second）
    time_sleep_second = 3
    # 使用的数据库名称
    use_database_name = 'yungang_grottoes_20240725'
    process_database_name = {'resample_5min_database_name': 'yungang_grottoes_resample_20240725_5min',
                             'resample_30min_database_name': 'yungang_grottoes_resample_20240725_30min',
                             'resample_1h_database_name': 'yungang_grottoes_resample_20240725_1h',
                             'interpolate_5min_database_name': 'yungang_grottoes_interpolation_20240725_5min',
                             'interpolate_30min_database_name': 'yungang_grottoes_interpolation_20240725_30min',
                             'interpolate_1h_database_name': 'yungang_grottoes_interpolation_20240725_1h'}
    # New Path
    New_Data_path = r'D:/PythonProject/MachineLearning/My_Dataset/YunGang_Grottoes_Data/SJD_23.0626-24.0606/'
    # 5min/30min/1h
    resample_time = '5min'
    time_between = ['2023-06-26 00:00:00', '2024-06-06 23:59:59']

    # 求整体平均（作为Y）的表名称
    average_table_name = 'overall_value_sensor'

    # test
    temp_dict_1 = {'5窟中室立佛右侧': [1, 2], '9窟东侧列柱': [1, 2],
                   '3窟顶气象站': [1, 2, 4, 5, 6, 10], '核心区': [1, 2, 4, 5, 7, 8, 9],
                   '9窟后室东北角上': [1, 3], '9窟后室东北角下': [1, 2, 3],
                   '9窟后室西北角上': [1, 2, 3], '9窟后室西北角下': [1, 2, 3],
                   '9窟前室': [1, 2, 3], '9窟中室东侧立佛北侧': [1, 2, 3],
                   '9窟中室西南角': [1, 2, 3], '10窟后室东北角': [1, 2, 3],
                   '10窟后室西北角': [1, 3], '10窟中室东南角': [1, 2, 3]}
    # A层传感器
    temp_dict_2 = {'A01': [1, 2, 3], 'A02': [1, 2, 3], 'A03': [1, 2, 3], 'A04': [1, 2, 3],
                   'A05': [1, 2, 3], 'A06': [1, 2, 3], 'A07': [1, 2, 3], 'A08': [1, 2, 3],
                   'A09': [1, 2, 3], 'A10': [1, 2, 3], 'A11': [1, 2, 3], 'A12': [1, 2, 3],
                   'A13': [1, 2, 3], 'A14': [1, 2, 3], 'A15': [1, 2, 3], 'A16': [1, 2, 3],
                   'A17': [1, 2, 3], 'A18': [1, 2, 3], 'A19': [1, 2, 3], 'A20': [1, 2, 3],
                   'A21': [1, 2, 3], 'A22': [1, 2, 3], 'A23': [1, 2, 3], 'A24': [1, 2, 3],
                   'A25': [1, 2, 3], 'A26': [1, 2, 3], 'A27': [1, 2, 3], 'A28': [1, 2, 3],
                   'A29': [1, 2, 3], 'A30': [1, 2, 3], 'A31': [1, 2, 3], 'A32': [1, 2, 3],
                   'A33': [1, 2, 3], 'A35': [1, 2, 3], 'A36': [1, 2, 3], 'A37': [1, 2, 3],
                   'A38': [1, 2, 3], 'A44': [1, 2, 3], 'A45': [1, 2, 3], 'A46': [1, 2, 3],
                   'A47': [1, 2, 3], 'A48': [1, 2, 3], 'A49': [1, 2, 3], 'A50': [1, 2, 3],
                   'A51': [1, 2, 3], 'A52': [1, 2, 3], 'A53': [1, 2, 3], 'A54': [1, 2, 3],
                   'A56': [1, 2, 3], 'A57': [1, 2, 3], 'A59': [1, 2, 3], 'A60': [1, 2, 3],
                   'A61': [1, 2, 3], 'A62': [1, 2, 3], 'A63': [1, 2, 3], 'A64': [1, 2, 3],
                   'A65': [1, 2, 3], 'A66': [1, 2, 3], 'A67': [1, 2, 3], 'A68': [1, 2, 3],
                   'A69': [1, 2, 3], }
    # AB层传感器
    temp_dict_3 = {'AB01': [1, 2, 3], 'AB02': [1, 2, 3], 'AB03': [1, 2, 3], 'AB04': [1, 2, 3],
                   'AB05': [1, 2, 3], 'AB06': [1, 2, 3], 'AB07': [1, 2, 3], 'AB08': [1, 2, 3],
                   'AB09': [1, 2, 3], 'AB10': [1, 2, 3], 'AB11': [1, 2, 3], 'AB12': [1, 2, 3]}
    # B层传感器
    temp_dict_4 = {'B01': [1, 2, 3], 'B02': [1, 2, 3], 'B03': [1, 2, 3], 'B04': [1, 2, 3],
                   'B05': [1, 2, 3], 'B06': [1, 2, 3], 'B07': [1, 2, 3], 'B08': [1, 2, 3],
                   'B09': [1, 2, 3], 'B10': [1, 2, 3], 'B11': [1, 2, 3], 'B12': [1, 2, 3],
                   'B13': [1, 2, 3], 'B14': [1, 2, 3], 'B15': [1, 2, 3], 'B16': [1, 2, 3],
                   'B17': [1, 2, 3], 'B18': [1, 2, 3], 'B19': [1, 2, 3], 'B20': [1, 2, 3],
                   'B21': [1, 2, 3], 'B22': [1, 2, 3], 'B23': [1, 2, 3], 'B24': [1, 2],
                   'B25': [1, 2], 'B26': [1, 2], 'B27': [1, 2, 3], 'B31': [1, 2, 3],
                   'B35': [1, 2, 3], 'B36': [1, 2, 3], 'B37': [1, 2, 3], 'B38': [1, 2, 3],
                   'B39': [1, 2, 3], 'B41': [1, 2, 3], 'B42': [1, 2, 3], 'B44': [1, 2, 3],
                   'B45': [1, 2, 3], 'B46': [1, 2, 3], 'B47': [1, 2, 3], 'B48': [1, 2, 3],
                   'B49': [1, 2, 3], 'B50': [1, 2, 3], 'B51': [1, 2, 3], 'B52': [1, 2, 3],
                   'B53': [1, 2, 3], 'B54': [1, 2, 3], 'B59': [1, 2, 3], 'B60': [1, 2, 3],
                   'B61': [1, 2, 3], 'B62': [1, 2, 3], 'B63': [1, 2, 3], 'B64': [1, 2, 3],
                   'B65': [1, 2, 3], 'B66': [1, 2, 3], 'B67': [1, 2, 3], 'B68': [1, 2, 3],
                   'B69': [1, 2, 3]}
    # C层传感器
    temp_dict_5 = {'C01': [1, 2, 3], 'C02': [1, 2, 3], 'C03': [1, 2, 3], 'C04': [1, 2, 3],
                   'C05': [1, 2, 3], 'C06': [1, 2], 'C07': [1, 2], 'C08': [1, 2, 3],
                   'C09': [1, 2, 3], 'C10': [1, 2, 3], 'C11': [1, 2, 3], 'C12': [1, 2, 3],
                   'C13': [1, 2, 3], 'C14': [1, 2, 3], 'C15': [1, 2, 3], 'C16': [1, 2, 3],
                   'C17': [1, 2, 3], 'C18': [1, 2], 'C19': [1, 2], 'C20': [1, 2],
                   'C21': [1, 2, 3], 'C25': [1, 2, 3], 'C29': [1, 2, 3], 'C30': [1, 2, 3],
                   'C31': [1, 2, 3], 'C32': [1, 2, 3], 'C33': [1, 2, 3], 'C35': [1, 2, 3],
                   'C36': [1, 2, 3]}
    # D层传感器
    temp_dict_6 = {'D01': [1, 2, 3], 'D02': [1, 2, 3], 'D03': [1, 2, 3], 'D04': [1, 2, 3],
                   'D05': [1, 2, 3], 'D06': [1, 2, 3], 'D07': [1, 2, 3], 'D08': [1, 2, 3],
                   'D09': [1, 2, 3], 'D10': [1, 2, 3], 'D11': [1, 2, 3], 'D12': [1, 2, 3],
                   'D13': [1, 2, 3], 'D14': [1, 2, 3], 'D15': [1, 2, 3], 'D16': [1, 2, 3],
                   'D17': [1, 2, 3], 'D18': [1, 2, 3], 'D19': [1, 2, 3], 'D20': [1, 2, 3],
                   'D21': [1, 2, 3], 'D22': [1, 2, 3], 'D23': [1, 2, 3], 'D24': [1, 2, 3],
                   'D25': [1, 2, 3], 'D26': [1, 2, 3], 'D27': [1, 2, 3], 'D29': [1, 2, 3],
                   'D30': [1, 2, 3], 'D31': [1, 2, 3], 'D32': [1, 2, 3]}
    # E层传感器
    temp_dict_7 = {'E01': [4, 5], 'E02': [4, 5], 'E03': [4, 5], 'E04': [4, 5], 'E05': [4, 5],
                   'E06': [4, 5], 'E07': [4, 5], 'E08': [4, 5], 'E09': [4, 5], 'E10': [4, 5],
                   'E11': [4, 5]}
    # F层传感器
    temp_dict_8 = {'F01': [4, 5], 'F02': [4, 5], 'F03': [4, 5], 'F04': [4, 5], 'F05': [4, 5],
                   'F06': [4, 5], 'F07': [4, 5], 'F08': [4, 5], 'F09': [4, 5], 'F10': [4, 5],
                   'F11': [4, 5], 'F12': [4, 5], 'F13': [4, 5], 'F14': [4, 5], 'F15': [4, 5],
                   'F17': [4, 5], 'F18': [4, 5], 'F19': [4, 5], 'F21': [4, 5],
                   'F31': [1, 2, 4, 5], 'F32': [1, 2, 4, 5]}

    config_dict_temp = {'9窟后室东北角上': [1, 3]}

    # # 一次性insert
    # for key, value in temp_dict_8.items():
    #     # 创建表
    #     create_table(key, use_database_name)
    #
    #     # 插入数据
    #     sensor_location_insert_all(key, use_database_name)
    #
    #     # 延时
    #     time.sleep(time_sleep_second)

    # # 一次性update
    # for key, value in config_dict.items():
    #     # 更新数据
    #     sensor_location_update_all(key, use_database_name)
    #     # 延时
    #     time.sleep(time_sleep_second)

    # 得到9窟、10窟 SJD 各18个点的数据
    config_dict_SJD = [['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'AB01', 'AB02', 'AB03',
                        'AB04', 'AB05', 'AB06', 'B01', 'B02', 'B03', 'B04', 'B05', 'B06'],
                       ['A63', 'A64', 'A65', 'A66', 'A67', 'A68', 'AB07', 'AB08', 'AB09',
                        'AB10', 'AB11', 'AB12', 'B63', 'B64', 'B65', 'B66', 'B67', 'B68']]
    # config_dict_SJD = ['A63']

    # # 将原始表的数据经过重采样处理后转入新表
    # for ss in config_dict_SJD:
    #     for s in ss:
    #         resample_between_time(s, resample_interval=resample_time,
    #                               time_between_list=time_between,
    #                               original_database=use_database_name,
    #                               now_database=process_database_name['resample_'
    #                                                                  + resample_time
    #                                                                  + '_database_name'],
    #                               to_csv=True, to_csv_path=New_Data_path + 'after_resample_'
    #                               + resample_time + "/")

    # # 将经过重采样处理后的数据经过插值转入新表
    # for ss in config_dict_SJD:
    #     for s in ss:
    #         interpolate_between_time(s, time_between_list=time_between,
    #                                  original_database=process_database_name['resample_'
    #                                                                          + resample_time
    #                                                                          + '_database_name'],
    #                                  now_database=process_database_name['interpolate_'
    #                                                                     + resample_time
    #                                                                     + '_database_name'],
    #                                  to_csv=True, to_csv_path=New_Data_path + 'after_interpolate_'
    #                                  + resample_time + "/")

    # # 得到SJD 18个点的平均数据作为整个SJD的整体数据
    # get_entire_average_data(average_table_name=average_table_name, sensor_str_list=SJD_table_list,
    #                         attribution_list=['air_temperature', 'air_humidity'],
    #                         database_name='yungang_grottoes_interpolation_20240725_5min',
    #                         to_csv=True, to_csv_path=New_Data_path + '/')

    # # 获取指定时间段的数据
    # get_entire_average_data(average_table_name=average_table_name, sensor_str_list=SJD_table_list,
    #                         attribution_list=['air_temperature', 'air_humidity', 'wall_temperature'],
    #                         database_name=process_database_name['interpolate_' + resample_time + '_database_name'],
    #                         to_csv=True, to_csv_path=New_Data_path + 'after_interpolate_' + resample_time + '/')
