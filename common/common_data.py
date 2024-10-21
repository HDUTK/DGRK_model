#!/usr/bin/env python3.6.8
# -*- coding: utf-8 -*-
"""
function description: 此文件用于提供处理随机数据生成及日期格式转换等函数
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2021.2.3
time: 2022/3/15 23:12
version: V1.0
"""
import time


class DateData(object):
    """
    处理日期与时间数据方法
    """

    def __init__(self):
        pass

    @staticmethod
    def timestap_ms():
        """
        生成毫秒时间戳
        :return: 当前时间的毫秒级时间戳
        """
        return int(round(time.time() * 1000000))

    @staticmethod
    def timestamp2date(timestamp):
        """
        毫秒级时间戳转日期
        :param timestamp: 毫秒级时间戳
        :return: 对应格式日期
        """
        return time.strftime("%Y-%m-%d", time.localtime(timestamp / 1000))

    @staticmethod
    def timestamp2time(timestamp):
        """
        毫秒级时间戳转日期
        :param timestamp: 毫秒级时间戳
        :return: 对应格式日期时间
        """
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp / 1000))

    @staticmethod
    def date2timestap(date):
        """
        日期转毫秒级时间戳  有误
        :param date: 日期
        :param date_fmt: 日期格式
        :return: 毫秒级时间戳
        """
        return int(time.mktime(time.strftime(date))) * 1000


# b = '2022-03-15 23:22:13'
# a = 1647357733688
# print(DateData.date2timestap(b))
# 2022-03-15 23:22:13
