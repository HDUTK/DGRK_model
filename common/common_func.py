#!/usr/bin/env python3.6.8
# -*- coding: utf-8 -*-
"""
function description: 此文件用于处理公共函数
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2021.2.3
time: 2022/3/20 22:39
version: V1.0
"""

import subprocess
import ctypes
import sys
import os
import datetime


def run_command(order, is_admin=False):
    """
    运行命令行指令，并返回输出信息
    :param order:命令行指令
    :param is_admin:是否以管理员身份运行，True/False
    :return:输出信息，忽略空行
    """
    print("正在执行cmd命令：" + order)

    if is_admin is True:
        try:
            ctypes.windll.shell32.IsUserAnAdmin()
        except Exception as e:
            print("以管理员运行失败，失败原因：" + str(e))
            return

    popen = subprocess.Popen(order, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, err = popen.communicate()

    # try:
    #     print('std_err: ' + err)
    # except Exception as e:
    #     print(e)

    # print(type(err))
    # print(err)
    print('执行结果: ' + out.decode('gbk'))
    # print('std_err: ' + err.decode('gbk'))
    # print('returncode: ' + str(popen.returncode))

    return


def path_and_name_exist(path_and_name, str_='/'):
    """
    确认绝对路径下文件夹是否存在，若不存在则创建文件夹（路径的斜杠必须用/）
    :param path_and_name: 输入绝对路径，包括文件的名字（.txt/.xls）
    :param str_: 路径的斜杠
    :return:
    """
    # 获取最后一个该字符的下标
    str_index = path_and_name.rfind(str_)
    path = path_and_name[:str_index]
    dir_path_exist_or_makedir(path)
    return


def dir_path_exist_or_makedir(dir_path):
    """
    确认path是否存在，若不存在则创建文件夹
    :param dir_path: path路径
    :return:
    """
    # 如果不存在path，则创建
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        pass
    return


def write_to_txt(txt_dir_path, txt_name, write_context: list, mode='w', encoding=None, **kwargs):
    """
    将要打印的内容输出至txt
    :param txt_dir_path: txt文件路径
    :param txt_name: txt文件名称
    :param write_context: 要写入的内容，list格式，尽量保证里面内容全为str
    :param mode: 打开的模式
    :param encoding: 编码格式
    :param kwargs: 其他参数
    :return:
    """
    # 如果不存在path，则创建
    dir_path_exist_or_makedir(txt_dir_path)

    # 内容写入
    print('-' * 50)
    file = open(txt_dir_path + '/' + txt_name, mode=mode, encoding=encoding)
    for z in write_context:
        file.write(str(z))
    file.close()
    print('-' * 50)

    return


def train_start_end_time(start_or_end: str, pre_flag=False):
    """
    输出现在时间，训练网络用
    :param start_or_end: 开始训练 or 结束训练，start/end
    :param pre_flag: 预训练标识
    :return: 现在时间
    """
    now_time = datetime.datetime.now()
    if start_or_end == 'start':
        print('-----------------------------------------------')
        if pre_flag:
            print('Start Pre Training! ')
            print('Pre Training Start Time: ', now_time)
        else:
            print('Start Training! ')
            print('Training Start Time: ', now_time)
        print('-----------------------------------------------')
    elif start_or_end == 'end':
        print('-----------------------------------------------')
        if pre_flag:
            print('Start Pre Training! ')
            print('Pre Training End Time: ', now_time)
        else:
            print('End Training! ')
            print('Training End Time: ', now_time)
        print('-----------------------------------------------')
    else:
        print('Time Error!')
        exit(0)
    return now_time

# run_command('ping', True)
#
#
# def is_admin():
#     try:
#         return ctypes.windll.shell32.IsUserAnAdmin()
#     except:
#         return False
#
#
# if is_admin():
#     run_command('net start mysql')
# else:
#     # Re-run the program with admin rights
#     ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, __file__, None, 1)

# a = CMD()
# a.is_admin = False
# a.run_cmd('net start mysql')
