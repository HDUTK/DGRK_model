#!/usr/bin/env python3.8.10
# -*- coding: utf-8 -*-
"""
function description: 此文件用于云冈石窟数据KAN训练
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2020.2.5
time: 2024/6/11 15:00
version: V1.0
"""
import torch
from torch import nn

from config.Yungang_Grottoes_config import *

from data.Dataset_util.Yungang_Grottoes.YunGang_Grottoes_Data_used import Yungang_Grottoes_Data_preprocessing
from LogicAlgorithm.ML_method.KAN.Train_Kolmogorov_Arnold_Network import packaged_implement_KAN

from LogicAlgorithm.ML_method.KAN.ALL_KAN.pykan.kan import KAN as original_KAN
from LogicAlgorithm.ML_method.KAN.ALL_KAN.efficient_KAN.efficient_kan import KAN as efficient_KAN
from LogicAlgorithm.ML_method.KAN.ALL_KAN.Chebyshev_KAN.ChebyKANLayer import Chebyshev_KAN
from LogicAlgorithm.ML_method.KAN.ALL_KAN.Chebyshev_KAN.ChebyKANLayer_2 import Chebyshev_KAN as Chebyshev_KAN_2
from LogicAlgorithm.ML_method.KAN.ALL_KAN.Chebyshev_KAN.KACnet import KAC_Net as Chebyshev_KAN_3
from LogicAlgorithm.ML_method.KAN.ALL_KAN.Fast_KAN.fastkan import FastKAN as Fast_KAN
from LogicAlgorithm.ML_method.KAN.ALL_KAN.Faster_KAN.fasterkan import FasterKAN as Faster_KAN
from LogicAlgorithm.ML_method.KAN.ALL_KAN.RBF_KAN.Fast_RBF_KAN import RBFKAN as RBF_KAN
from LogicAlgorithm.ML_method.KAN.ALL_KAN.RBF_KAN.bsrbf_kan import BSRBF_KAN
from LogicAlgorithm.ML_method.KAN.ALL_KAN.Legendre_KAN.KALnet import KAL_Net as Legendre_KAN
from LogicAlgorithm.ML_method.KAN.ALL_KAN.Jacobi_KAN.JacobiKANLayer import Jacobi_KAN
from LogicAlgorithm.ML_method.KAN.ALL_KAN.Gottlieb_kan.gottlieb_kan import GottliebKAN as Gottlieb_KAN
from LogicAlgorithm.ML_method.KAN.ALL_KAN.Fourier_KAN.fftKAN import Fourier_KAN
from LogicAlgorithm.ML_method.KAN.ALL_KAN.Wav_KAN.KAN import KAN as Wav_KAN


"""
config
"""
# 5min/30min/1h
resample_time = '5min'

# 9/10
grottoes_number = 10

# B68,A65,AB11,B66
class_sensor_list = 'A03,B05,AB02'.split(',')

# air_temperature/air_humidity/wall_temperature
my_columns_name_fliter = 'air_temperature'

# all/summer/winter
my_season_attribute = 'summer'

# 求整体平均（作为Y）的表名称
grottoes_number_dict = {'9': 'a63-b68', '10': 'a01-b06'}
average_table_name = 'overall_value_sensor_' + grottoes_number_dict[str(grottoes_number)]

# 5min/30min/1h
Data_path = r'D:/PythonProject/MachineLearning/My_Dataset/YunGang_Grottoes_Data/SJD_23.0626-24.0606/' \
            r'after_interpolate_' + resample_time + '/'

# 用于验证的周
x = 0  # x=0/1/2/3/4
if x in [0, 1, 2, 3]:
    validate_week_list = [summer_week_list[x], summer_week_list[x+8]]
else:
    validate_week_list = [summer_week_list[7]]

# 是否使用train_test_split
train_test_split_flag = False

# 是否测试
test_flag = True
if test_flag:
    train_test_split_flag = False
# 测试用
validate_week_list = []

# 用于测试的周
test_week_list = [['2023-07-26 00:00:00', '2023-08-01 23:59:59'],
                  ['2023-09-20 00:00:00', '2023-09-26 23:59:59']]

# 是否显示图片
plot_show_flag = False

# batch size
my_batch_size = 32

"""
config
"""


"""
导入数据
"""
res_data_dict, res_TensorDataset_dict, res_DataLoader_dict, res_data_tensor_dict = \
    Yungang_Grottoes_Data_preprocessing(Data_path=Data_path, class_sensor_list=class_sensor_list,
                                        my_columns_name_fliter=my_columns_name_fliter,
                                        my_season_attribute=my_season_attribute,
                                        validate_week_list=validate_week_list,
                                        test_week_list=test_week_list,
                                        std_flag=False,
                                        my_batch_size=my_batch_size,
                                        train_test_split_flag=train_test_split_flag,
                                        average_table_name=average_table_name)


"""
模型初始化及训练
"""
# 若可以，使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 初始化，[4, 2, 1]/[4, 3, 1]/[4, 4, 1]/[4, 5, 1]/[4, 6, 1]
# nn_list = [4, 2, 1]
# str_nn = str(nn_list)
# # 训练次数
# epochs = 50
# # optimizer：LBFGS/Adam/AdamW
# optimizer = 'AdamW'
# # optimizer_learning_rate：0.05/0.01/0.005/0.001/0.0005/0.0001
# optimizer_learning_rate = 0.05
# # 是否学习率衰减
# lr_scheduler_flag = False


def train():
    if KAN_used_str == 'old_KAN':
        # old_KAN

        # 数据
        old_KAN_data = {'train_input': res_data_tensor_dict['train_input'],
                        'train_label': res_data_tensor_dict['train_label'],
                        'test_input': res_data_tensor_dict['test_input'],
                        'test_label': res_data_tensor_dict['test_label']}

        # 初始化
        my_KAN = original_KAN(width=nn_list, grid=5, k=3, device=device)

        # 是否保存训练KAN中间图片，若保存会大大拖慢训练速度
        save_train_fig_flag = False

        # optimizer：LBFGS/Adam/AdamW
        optimizer = 'LBFGS'
        # # optimizer_learning_rate：0.05/0.01/0.005/0.001
        # optimizer_learning_rate = 0.001

        # 训练
        implement_KAN = packaged_implement_KAN(
            Data_path, my_season_attribute, my_columns_name_fliter, device, res_data_dict,
            res_TensorDataset_dict, res_DataLoader_dict, res_data_tensor_dict, my_KAN, KAN_used_str,
            nn_list)
        my_KAN = implement_KAN.implement_train_old_KAN_model(
            old_KAN_data, prune_flag=True, train_2nd_flag=False, optimizer=optimizer,
            optimizer_learning_rate=optimizer_learning_rate, epochs=epochs,
            plot_show_flag=plot_show_flag, save_train_fig_flag=save_train_fig_flag)

    elif KAN_used_str == 'efficient_KAN':
        # efficient_KAN

        # 初始化   nn.Identity/nn.SiLU
        my_KAN = efficient_KAN(layers_hidden=nn_list, base_activation=nn.SiLU).to(device)

        # optimizer：LBFGS/Adam/AdamW
        optimizer = 'AdamW'
        # # optimizer_learning_rate：0.05/0.01/0.005/0.001/0.0005/0.0001
        # optimizer_learning_rate = 0.05

    elif KAN_used_str in ['Chebyshev_KAN', 'Chebyshev_KAN_2', 'Chebyshev_KAN_3']:
        # Chebyshev_KAN/Chebyshev_KAN_2/Chebyshev_KAN_3  训练模型成功后测试会失败

        # 初始化
        if KAN_used_str == 'Chebyshev_KAN':
            my_KAN = Chebyshev_KAN(layer_list=nn_list, degree_list=[6, 6], with_LayerNorm_flag=True).to(device)
        elif KAN_used_str == 'Chebyshev_KAN_2':
            my_KAN = Chebyshev_KAN_2(layer_list=nn_list, degree_list=[10, 10], with_LayerNorm_flag=False).to(device)
        elif KAN_used_str == 'Chebyshev_KAN_3':
            my_KAN = Chebyshev_KAN_3(layers_hidden=nn_list, polynomial_order=6, base_activation=nn.Identity).to(device)
        else:
            pass

        # optimizer：LBFGS/Adam/AdamW
        optimizer = 'AdamW'
        # # optimizer_learning_rate：0.05/0.01/0.005/0.001
        # optimizer_learning_rate = 0.05

    elif KAN_used_str == 'Fast_KAN':
        # Fast_KAN

        # 初始化
        my_KAN = Fast_KAN(layers_hidden=nn_list).to(device)

        # optimizer：LBFGS/Adam/AdamW
        optimizer = 'AdamW'
        # # optimizer_learning_rate：0.05/0.01/0.005/0.001
        # optimizer_learning_rate = 0.001

    elif KAN_used_str == 'Faster_KAN':
        # Faster_KAN

        # 初始化   nn.Identity/nn.SiLU
        my_KAN = Faster_KAN(layers_hidden=nn_list, base_activation=nn.Identity).to(device)

        # optimizer：LBFGS/Adam/AdamW
        optimizer = 'AdamW'
        # # optimizer_learning_rate：0.05/0.01/0.005/0.001
        # optimizer_learning_rate = 0.05

    elif KAN_used_str == 'RBF_KAN':
        # RBF_KAN

        # 初始化 需要选择使用的RBF函数，默认gaussian_rbf
        # gaussian_rbf/multiquadratic_rbf/thin_plate_spline_rbf(Loss为nan)/inverse_quadric
        my_KAN = RBF_KAN(layers_hidden=nn_list, rbf_mode='multiquadratic_rbf').to(device)

        # optimizer：LBFGS/Adam/AdamW
        optimizer = 'AdamW'
        # # optimizer_learning_rate：0.05/0.01/0.005/0.001
        # optimizer_learning_rate = 0.005

    elif KAN_used_str == 'BSRBF_KAN':
        # BSRBF_KAN   训练模型完成后测试失败

        # 初始化
        my_KAN = BSRBF_KAN(layers_hidden=nn_list, base_activation=torch.nn.Identity).to(device)

        # optimizer：LBFGS/Adam/AdamW
        optimizer = 'AdamW'
        # # optimizer_learning_rate：0.05/0.01/0.005/0.001
        # optimizer_learning_rate = 0.005

    elif KAN_used_str == 'Legendre_KAN':
        # Legendre_KAN   训练模型成功后测试会失败

        # 初始化
        my_KAN = Legendre_KAN(layers_hidden=nn_list, polynomial_order=6, base_activation=nn.Identity).to(device)

        # optimizer：LBFGS/Adam/AdamW
        optimizer = 'AdamW'
        # # optimizer_learning_rate：0.05/0.01/0.005/0.001
        # optimizer_learning_rate = 0.05

    elif KAN_used_str == 'Jacobi_KAN':
        # Jacobi_KAN   训练模型成功后测试会失败

        # 初始化
        my_KAN = Jacobi_KAN(layer_list=nn_list, degree_list=[6, 6], with_LayerNorm_flag=False).to(device)

        # optimizer：LBFGS/Adam/AdamW
        optimizer = 'AdamW'
        # # optimizer_learning_rate：0.05/0.01/0.005/0.001
        # optimizer_learning_rate = 0.01

    elif KAN_used_str == 'Gottlieb_KAN':
        # Gottlieb_KAN   训练模型成功后测试会失败

        # 初始化
        my_KAN = Gottlieb_KAN(layers_hidden=nn_list).to(device)

        # optimizer：LBFGS/Adam/AdamW
        optimizer = 'AdamW'
        # # optimizer_learning_rate：0.05/0.01/0.005/0.001
        # optimizer_learning_rate = 0.005

    elif KAN_used_str == 'Fourier_KAN':
        # Fourier_KAN

        # 初始化
        my_KAN = Fourier_KAN(layer_list=nn_list, gridsize=10).to(device)

        # optimizer：LBFGS/Adam/AdamW
        optimizer = 'AdamW'
        # # optimizer_learning_rate：0.05/0.01/0.005/0.001
        # optimizer_learning_rate = 0.01

    elif KAN_used_str == 'Wav_KAN':
        # Wav_KAN

        # 初始化 需要选择使用的小波函数，默认mexican_hat
        # mexican_hat/morlet/dog/meyer/shannon
        my_KAN = Wav_KAN(layers_hidden=nn_list, wavelet_type='mexican_hat').to(device)

        # optimizer：LBFGS/Adam/AdamW
        optimizer = 'AdamW'
        # # optimizer_learning_rate：0.05/0.01/0.005/0.001
        # optimizer_learning_rate = 0.01

    else:
        implement_KAN = None
        print('The name of KAN is Wrong!')
        exit(0)

    # 训练
    if KAN_used_str == 'old_KAN':
        pass
    else:
        implement_KAN = packaged_implement_KAN(
            Data_path, my_season_attribute, my_columns_name_fliter, device, res_data_dict,
            res_TensorDataset_dict, res_DataLoader_dict, res_data_tensor_dict, my_KAN, KAN_used_str,
            nn_list)
        my_KAN = implement_KAN.implement_train_KAN_model(
            epochs=epochs, optimizer=optimizer, optimizer_learning_rate=optimizer_learning_rate,
            criterion='MSE', lr_scheduler_flag=lr_scheduler_flag, plot_show_flag=plot_show_flag,
            AdamW_weight_decay=1e-4, scheduler_name='ExponentialLR', ExponentialLR_gamma=0.8,
            gradient_check=False, clip_grad_flag=False)

    return implement_KAN, my_KAN


def validate(implement_KAN_temp):
    """
    模型验证
    """
    if validate_week_list and (KAN_used_str != 'old_KAN'):
        implement_KAN_temp.implement_validate_KAN_model(
            validate_week_list, train_test_split_flag, validate_index=x, plot_show_flag=True)
    return


def my_test(implement_KAN_temp, compare_with_NN_temp=False, **kwargs):
    """
    模型测试
    :param compare_with_NN_temp: 是否和NN比较, 默认否
    :param kwargs: 其他参数
    :return:
    """
    if test_flag:
        implement_KAN_temp.implement_test_KAN_model(test_week_list, plot_show_flag=True,
                                                    compare_with_NN=compare_with_NN_temp,
                                                    **kwargs)
    return


def save(my_KAN_temp, implement_KAN_temp):
    """
    模型保存
    """
    if lr_scheduler_flag:
        scheduler_str = 'Decay'
    else:
        scheduler_str = 'NoDecay'

    if KAN_used_str == 'old_KAN':
        my_KAN_temp.save_ckpt(name='My_KAN_Model', folder=Data_path + 'KAN/' + KAN_used_str + '/'
                              + my_season_attribute + '_' + my_columns_name_fliter + '/' + str_nn
                              + '_' + str(optimizer_learning_rate) + '_' + scheduler_str)
    else:
        implement_KAN_temp.model_train.save_model(save_path=Data_path + 'KAN/' + KAN_used_str + '/'
                                                  + my_season_attribute + '_' + my_columns_name_fliter
                                                  + '/' + str_nn + '_' + str(optimizer_learning_rate)
                                                  + '_' + scheduler_str, model_name='My_KAN_Model')
    return


if __name__ == '__main__':

    # 使用的KAN网络：old_KAN/efficient_KAN/Chebyshev_KAN/Chebyshev_KAN_2/Chebyshev_KAN_3/Fast_KAN/
    # Faster_KAN/RBF_KAN/BSRBF_KAN/Legendre_KAN/Jacobi_KAN/Gottlieb_KAN/Fourier_KAN/Wav_KAN/
    KAN_used_str = 'old_KAN'

    # 是否和NN比较
    compare_with_NN = False

    # optimizer_learning_rate：[0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    optimizer_learning_rate_list = [0.005]
    # 是否学习率衰减：[True, False]
    lr_scheduler_flag_list = [True]

    # 中间节点的节点数：[2, 3, 4, 5, 6]
    layer_num_list = [3]
    # 训练次数
    epochs = 50

    for layer_num in layer_num_list:
        # 初始化，[3, 2, 1]/[3, 3, 1]/[3, 4, 1]/[3, 5, 1]/[3, 6, 1]
        nn_list = [len(class_sensor_list), layer_num, 1]
        str_nn = str(nn_list)
        for lr_scheduler_flag in lr_scheduler_flag_list:
            for optimizer_learning_rate in optimizer_learning_rate_list:
                my_implement_KAN, my_KAN_now = train()
                validate(my_implement_KAN)
                my_test(my_implement_KAN, compare_with_NN, NN_str_nn=str(nn_list),
                        NN_optimizer_learning_rate=optimizer_learning_rate,
                        grottoes_number=grottoes_number)
                save(my_KAN_now, my_implement_KAN)


