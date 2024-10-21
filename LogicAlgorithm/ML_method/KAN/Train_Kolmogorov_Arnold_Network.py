#!/usr/bin/env python3.8.10
# -*- coding: utf-8 -*-
"""
function description: 此文件用于集成KAN及歌中歌KAN的变体，方便应用于实例
在terminal中输入jupyter notebook实现从网页中打开ipynb
在terminal中ctrl+C停止运行
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2020.2.5
time: 2024/6/1 14:09
version: V1.0
"""
import torch
import ast
import numpy as np
import torch.utils.data
import pandas as pd
import os
import sys
import shutil
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sympy

from common.common_func import write_to_txt, train_start_end_time
from common.plot_func import my_plot_func
from LogicAlgorithm.Network_common_func import Network_common


class Init_Train_Kolmogorov_Arnold_Network(object):
    """
    初始化、训练Kolmogorov-Arnold网络（efficient_KAN）
    """

    def __init__(self, KAN_model, optimizer_algorithm='Adam', optimizer_learning_rate=0.01,
                 criterion='MSE', lr_scheduler_flag=False, **kwargs):
        """
        初始化Kolmogorov-Arnold网络
        :param model: KAN模型，需要一个KAN类或KAN类的变体生成的对象
        :param optimizer_algorithm: 优化算法，默认Adam
        :param optimizer_learning_rate: 优化算法的学习率，默认0.01
        :param criterion: 损失函数，默认MSE
        :param lr_scheduler_flag: 是否调整学习率，默认否，若是，则需额外传入参数scheduler_name(及其他调整器参数)
        :param kwargs: 其他参数
        """
        self.model = KAN_model
        self.optimizer_learning_rate = optimizer_learning_rate
        self.optimizer_algorithm_str = optimizer_algorithm
        self.criterion_str = criterion
        self.lr_scheduler_flag = lr_scheduler_flag
        self.lr_scheduler = None

        # 如果可以，则使用GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

        # 优化算法
        self.optimizer_algorithm = Network_common.optimizer_algorithm(self.model,
                                                                      self.optimizer_algorithm_str,
                                                                      optimizer_learning_rate=
                                                                      self.optimizer_learning_rate,
                                                                      **kwargs)

        # 损失函数
        self.criterion = Network_common.loss_function(self.criterion_str, self.device, **kwargs)

        # 定义调整学习率
        if self.lr_scheduler_flag:
            scheduler_name = kwargs.get("scheduler_name", 'ExponentialLR')
            self.lr_scheduler = Network_common.optimizer_scheduler(optimizer_input=self.optimizer_algorithm,
                                                                   optimizer_lr_scheduler_name=scheduler_name,
                                                                   **kwargs)

    def train_KAN(self, train_loader, epochs=50, epochs_print_Flag=True, **kwargs):
        """
        训练KAN模型
        :param train_loader: 输入的训练数据
        :param epochs: 训练的次数，默认50次
        :param epochs_print_Flag: 训练过程是否输出，默认为True
        :param kwargs: 其他参数
        :return: 训练完成的模型model和loss，及训练时间相关
        """
        loss_list = []

        # 是否在训练过程中检查梯度
        gradient_check = kwargs.get("gradient_check", False)
        # 防止梯度爆炸可以采用梯度裁剪
        clip_grad_norm_flag = kwargs.get("clip_grad_flag", False)

        # 开始训练
        # 训练起始时间
        start_time = train_start_end_time('start')

        # 查看当前优化器中参数及其参数名
        # for param_group in self.optimizer_algorithm.param_groups:
        #     for param in param_group['params']:
        #         print(param)
        # print('=' * 50)
        # for name, parameter in self.model.named_parameters():
        #     print(name, parameter)
        # exit(0)

        for epoch in range(epochs):
            train_num = 0
            loss = 0

            with tqdm(train_loader, file=sys.stdout) as pbar:
                for batch_x, batch_y in pbar:

                    # 将梯度重置为0，等待重新计算
                    self.optimizer_algorithm.zero_grad()

                    # 如果可能，放至GPU
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                    # 计算网络的输出
                    predict_y = self.model.forward(batch_x)

                    # 计算Loss
                    train_loss = self.criterion(predict_y, batch_y)

                    # 反向计算梯度
                    train_loss.backward()

                    # 根据梯度优化网络的参数
                    if self.optimizer_algorithm_str == 'LBFGS':
                        model = self.model
                        criterion = self.criterion

                        # 构造闭包供LBFGS调用
                        def closure():
                            # 计算网络的输出
                            predict_y_temp = model.forward(batch_x)

                            # 计算Loss
                            train_loss_temp = criterion(predict_y_temp, batch_y)

                            # 反向计算梯度
                            train_loss_temp.backward()

                            return train_loss_temp

                        # 防止梯度爆炸可以采用梯度裁剪
                        if clip_grad_norm_flag:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        else:
                            pass

                        self.optimizer_algorithm.step(closure=closure)
                    else:
                        # 防止梯度爆炸可以采用梯度裁剪
                        if clip_grad_norm_flag:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        else:
                            pass

                        self.optimizer_algorithm.step()

                    # 计算训练数量
                    train_num = train_num + batch_x.size(0)

                    # 将该个mini-batch的loss加入至这个epoch下的整体loss中
                    loss = loss + train_loss.item()

                    pbar.set_postfix(lr=self.optimizer_algorithm.param_groups[0]['lr'])

            # 计算这个epoch下的loss
            loss = loss / train_num
            loss_list.append(loss)

            # 输出该次epoch的Loss
            if epochs_print_Flag:
                # display the epoch training loss
                print("Epoch : {}/{}, Loss = {:.5f}".format(epoch + 1, epochs, loss))
            else:
                pass

            if self.lr_scheduler_flag:
                # 更新学习率
                self.lr_scheduler.step()
            else:
                pass

            # 输出此次epoch的梯度
            if gradient_check:
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        print(f"Gradient for {name}: {param.grad.abs().mean().item()}")
            else:
                pass

        # 结束训练
        # 训练结束时间
        end_time = train_start_end_time('end')
        # 训练时长
        print('Training time: ', end_time - start_time)
        print('-----------------------------------------------')

        return self.model, loss_list, [start_time, end_time]

    def validate_test_model(self, my_model, my_validate_test_dataset, result_to_dataframe_flag=False, **kwargs):
        """
        验证/测试模型
        :param my_model: 待验证/测试的模型
        :param my_validate_test_dataset: 验证/测试用数据集
        :param result_to_dataframe_flag: 结果是否转成Dataframe
        :param kwargs: 其他参数
        :return: 验证/测试结果，[预测结果list, 实际结果list],loss列表/Dataframe,loss列表
        """

        with torch.no_grad():
            val_test_loss = 0
            outputs_predict_res = []
            outputs_true_res = []
            val_test_num = 0
            for validate_test_X, validate_test_Y in my_validate_test_dataset:
                temp_outputs = my_model.forward(validate_test_X)

                val_test_loss = val_test_loss \
                                + self.criterion(temp_outputs,
                                                 validate_test_Y.to(self.device)).item() * validate_test_X.size(0)

                outputs_predict_res.append(temp_outputs.cpu().numpy()[0])
                outputs_true_res.append(validate_test_Y.cpu().numpy()[0])
                val_test_num = val_test_num + validate_test_X.size(0)

            # 计算loss
            val_test_loss = val_test_loss / val_test_num

        print('-' * 50)
        print('Verification/Test Set Prediction Results: ')
        print(outputs_predict_res)
        print('Verification/Test Set True Results: ')
        print(outputs_true_res)
        print('Verification/Test Loss: ')
        print(val_test_loss)
        print('-' * 50)

        # 结果是否转成Dataframe
        if result_to_dataframe_flag:
            res_dataframe = {'predict': outputs_predict_res, 'true': outputs_true_res}
            validate_test_result = pd.DataFrame(res_dataframe)
            print('Validate/Test Result Dataframe: ')
            print(validate_test_result)
            print('-' * 50)
            return validate_test_result, val_test_loss
        else:
            pass

        return [outputs_predict_res, outputs_true_res], val_test_loss

    def save_model(self, save_path, model_name='My_KAN_Model'):
        """
        保存此时的模型
        :param save_path: 保存路径，例如'D:/result'
        :param model_name: 保存时的模型名字，默认：My_KAN_Model
        :return:
        """
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        Path = save_path + '/' + model_name + '.pt'

        torch.save(self.model, Path)
        print('===============================')
        print('Save Model to (' + Path + ')!')
        print('===============================')
        return


class packaged_implement_KAN(object):
    """
    包装 使用Kolmogorov-Arnold网络，包括初始化、训练等
    """

    def __init__(self, Data_path, my_season_attribute, my_columns_name_fliter, device, res_data_dict,
                 res_TensorDataset_dict, res_DataLoader_dict, res_data_tensor_dict, my_KAN,
                 KAN_used_str, nn_list, **kwargs):
        """
        包装 使用Kolmogorov-Arnold网络，包括初始化、训练等
        :param Data_path: 数据地址
        :param my_season_attribute: 季节，all/summer/winter
        :param my_columns_name_fliter: 属性，air_temperature/air_humidity/wall_temperature
        :param device: cpu or gpu
        :param res_data_dict: datarame字典，为Yungang_Grottoes_Data_preprocessing函数的返回
        :param res_TensorDataset_dict: TensorDataset字典，为Yungang_Grottoes_Data_preprocessing函数的返回
        :param res_DataLoader_dict: DataLoader字典，为Yungang_Grottoes_Data_preprocessing函数的返回
        :param res_data_tensor_dict: tensor字典，为Yungang_Grottoes_Data_preprocessing函数的返回
        :param my_KAN: 初始化后的KAN
        :param KAN_used_str: 使用的KAN名字，例如：efficient_KAN/Chebyshev_KAN/Fast_KAN
        :param nn_list: KAN结构，如[4, 3, 1]
        :param kwargs: 其他参数
        """
        self.Data_path = Data_path
        self.my_season_attribute = my_season_attribute
        self.my_columns_name_fliter = my_columns_name_fliter
        self.device = device
        self.res_data_dict = res_data_dict
        self.res_TensorDataset_dict = res_TensorDataset_dict
        self.res_DataLoader_dict = res_DataLoader_dict
        self.res_data_tensor_dict = res_data_tensor_dict
        self.my_KAN = my_KAN
        self.KAN_used_str = KAN_used_str
        self.nn_list = nn_list
        self.str_nn = str(self.nn_list)

        self.model_train = None
        self.optimizer_learning_rate = 0.01
        self.scheduler_str = ''

    def implement_train_old_KAN_model(self, old_KAN_data, prune_flag=True, train_2nd_flag=True,
                                      optimizer='LBFGS', optimizer_learning_rate=1e-4, epochs=50,
                                      plot_show_flag=True, save_train_fig_flag=False, **kwargs):
        """
        打包 最老的KAN训练
        :param old_KAN_data: 训练最老的KAN时用的数据
        :param prune_flag: 是否剪枝，默认是
        :param train_2nd_flag: 是否用第二次训练，默认是
        :param optimizer: 优化器，默认'LBFGS'
        :param optimizer_learning_rate: 第一次训练的学习率，默认1e-4
        :param epochs: 第一次训练的次数
        :param plot_show_flag: 是否
        :param save_train_fig_flag: 是否展示/保存所有训练中节点的激活函数图
        :param kwargs: 其他参数
        :return: 训练完成的KAN
        """

        print('+' * 50)
        print(self.KAN_used_str + ' is used!')
        print('+' * 50)

        # 展示网络结构细节及每层的参数总数
        print("Model Structure: ")
        print(self.my_KAN)  # 打印模型结构
        print("Total number of parameters: ", sum(p.numel() for p in self.my_KAN.parameters()))  # 打印模型参数数量
        # 导出至txt
        write_context = ["Model Structure: \n\n", str(self.my_KAN), "\n\nTotal number of parameters: ",
                         str(sum(p.numel() for p in self.my_KAN.parameters()))]
        write_to_txt(self.Data_path + 'KAN/' + self.KAN_used_str + '/' + self.my_season_attribute
                     + '_' + self.my_columns_name_fliter + '/' + self.str_nn, 'model_structure.txt',
                     write_context)

        # 第1次训练
        # old_KAN_train_res_dict 包含:
        # results['train_loss'], 1D array of training losses (RMSE)
        # results['test_loss'], 1D array of test losses (RMSE)
        # results['reg'], 1D array of regularization
        # 训练起始时间
        start_time = train_start_end_time('start')

        old_KAN_train_res_dict_1 = self.my_KAN.train(
            old_KAN_data, opt=optimizer, steps=epochs, lamb=optimizer_learning_rate,
            save_fig=save_train_fig_flag, img_folder=self.Data_path + 'KAN/' + self.KAN_used_str
                                                     + '/' + self.my_season_attribute + '_' + self.my_columns_name_fliter,
            device=self.device)

        # 训练结束时间
        end_time = train_start_end_time('end')
        # 训练时长
        print('Training time: ', end_time - start_time)
        print('-----------------------------------------------')

        # 导出至txt
        write_context = ["\nOptimizer : ", str(optimizer),
                         "\nOptimizer Learning Rate: ", str(optimizer_learning_rate),
                         "\nEpochs: ", str(epochs),
                         "\nTraining Start Time: ", str(start_time),
                         "\nTraining End Time: ", str(end_time),
                         "\nTraining time: ", str(end_time - start_time)]
        write_to_txt(self.Data_path + 'KAN/' + self.KAN_used_str + '/' + self.my_season_attribute
                     + '_' + self.my_columns_name_fliter + '/' + self.str_nn, 'model_structure.txt',
                     write_context, mode='a')

        # 损失
        print('+' * 50)
        print('Train Loss: ', old_KAN_train_res_dict_1['train_loss'])
        # 导出至txt
        write_context = ['1st Train Loss: \n', str(old_KAN_train_res_dict_1['train_loss'])]
        write_to_txt(self.Data_path + 'KAN/' + self.KAN_used_str + '/' + self.my_season_attribute
                     + '_' + self.my_columns_name_fliter + '/' + self.str_nn, 'Train_Loss.txt',
                     write_context)
        print('+' * 50)
        # 画损失图
        my_plot_func.time_curve({'y': old_KAN_train_res_dict_1['train_loss'], 'color_curve': 'k-',
                                 'curvename': 'train loss'}, title_name='Train Loss', xlabelname='epochs',
                                ylabelname='loss', show_flag=plot_show_flag, dpi=300,
                                save_dir_path=self.Data_path + 'KAN/' + self.KAN_used_str + '/'
                                              + self.my_season_attribute + '_' + self.my_columns_name_fliter + '/'
                                              + self.str_nn, save_figure_name='old_KAN_1_train_loss.png',
                                legend_font_size=30, xlabel_font_size=25, ylabel_font_size=25,
                                label_font_size=20, figure_size=(15, 8), title_font_size=30)
        my_plot_func.time_curve({'y': old_KAN_train_res_dict_1['train_loss'], 'color_curve': 'k-',
                                 'curvename': 'train loss'},
                                {'y': old_KAN_train_res_dict_1['test_loss'], 'color_curve': 'r-',
                                 'curvename': 'test loss'}, title_name='Train Loss', xlabelname='epochs',
                                ylabelname='loss', show_flag=plot_show_flag, dpi=300,
                                save_dir_path=self.Data_path + 'KAN/' + self.KAN_used_str + '/'
                                              + self.my_season_attribute + '_' + self.my_columns_name_fliter + '/'
                                              + self.str_nn, save_figure_name='old_KAN_1_train_test_loss.png',
                                legend_font_size=30, xlabel_font_size=25, ylabel_font_size=25,
                                label_font_size=20, figure_size=(15, 8), title_font_size=30)

        # 画图
        self.my_KAN.plot()
        # 存储图片
        plt.savefig(self.Data_path + 'KAN/' + self.KAN_used_str + '/' + self.my_season_attribute
                    + '_' + self.my_columns_name_fliter + '/' + self.str_nn + '/'
                    + 'old_KAN_model_structure_before_prune.png', dpi=300)
        print('+' * 20)
        print('Save picture to ' + self.Data_path + 'KAN/' + self.KAN_used_str + '/'
              + self.my_season_attribute + '_' + self.my_columns_name_fliter
              + '/' + self.str_nn + '/' + 'old_KAN_model_structure_before_prune.png')
        print('+' * 20)
        # 显示图片
        if plot_show_flag:
            plt.show()

        if prune_flag:
            # 剪枝
            self.my_KAN.prune()

            # 画图
            self.my_KAN.plot()
            # 存储图片
            plt.savefig(self.Data_path + 'KAN/' + self.KAN_used_str + '/' + self.my_season_attribute
                        + '_' + self.my_columns_name_fliter + '/' + self.str_nn + '/'
                        + 'old_KAN_model_structure_after_prune.png', dpi=300)
            print('+' * 20)
            print('Save picture to ' + self.Data_path + 'KAN/' + self.KAN_used_str + '/'
                  + self.my_season_attribute + '_' + self.my_columns_name_fliter + '/' + self.str_nn
                  + '/' + 'old_KAN_model_structure_after_prune.png')
            print('+' * 20)
            # 显示图片
            if plot_show_flag:
                plt.show()

        # 模型使用的函数
        print(self.my_KAN.suggest_symbolic(0, 0, 0)[0])
        self.my_KAN.auto_symbolic(lib=['tan', 'exp', 'sqrt', 'x^2', 'log', 'sin', 'abs', 'arctan',
                                       '1/sqrt(x)', '1/x'])

        if train_2nd_flag:
            # 第2次训练
            # old_KAN_train_res_dict 包含:
            # results['train_loss'], 1D array of training losses (RMSE)
            # results['test_loss'], 1D array of test losses (RMSE)
            # results['reg'], 1D array of regularization
            old_KAN_train_res_dict_2 = self.my_KAN.train(
                old_KAN_data, opt=optimizer, steps=10, lamb=1e-8, update_grid=False,
                save_fig=save_train_fig_flag, img_folder=self.Data_path + 'KAN/' + self.KAN_used_str
                                                         + '/' + self.my_season_attribute + '_' + self.my_columns_name_fliter + '/'
                                                         + self.str_nn, device=self.device)

            # 损失
            print('+' * 50)
            print('Train Loss: ', old_KAN_train_res_dict_2['train_loss'])
            # 导出至txt
            write_context = ['\n\n\n2st Train Loss: \n', str(old_KAN_train_res_dict_2['train_loss'])]
            write_to_txt(self.Data_path + 'KAN/' + self.KAN_used_str + '/' + self.my_season_attribute
                         + '_' + self.my_columns_name_fliter + '/' + self.str_nn, 'Train_Loss.txt',
                         write_context, mode='a')
            print('+' * 50)
            # 画损失图
            my_plot_func.time_curve({'y': old_KAN_train_res_dict_2['train_loss'], 'color_curve': 'k-',
                                     'curvename': 'train loss'}, title_name='Train Loss', xlabelname='epochs',
                                    ylabelname='loss', show_flag=plot_show_flag, dpi=300,
                                    save_dir_path=self.Data_path + 'KAN/' + self.KAN_used_str + '/'
                                                  + self.my_season_attribute + '_' + self.my_columns_name_fliter + '/'
                                                  + self.str_nn, save_figure_name='old_KAN_2_train_loss.png',
                                    legend_font_size=30, xlabel_font_size=25, ylabel_font_size=25,
                                    label_font_size=20, figure_size=(15, 8), title_font_size=30)
            my_plot_func.time_curve({'y': old_KAN_train_res_dict_2['train_loss'], 'color_curve': 'k-',
                                     'curvename': 'train loss'},
                                    {'y': old_KAN_train_res_dict_2['test_loss'], 'color_curve': 'r-',
                                     'curvename': 'test loss'}, title_name='Train Loss', xlabelname='epochs',
                                    ylabelname='loss', show_flag=plot_show_flag, dpi=300,
                                    save_dir_path=self.Data_path + 'KAN/' + self.KAN_used_str + '/'
                                                  + self.my_season_attribute + '_' + self.my_columns_name_fliter + '/'
                                                  + self.str_nn, save_figure_name='old_KAN_2_train_test_loss.png',
                                    legend_font_size=30, xlabel_font_size=25, ylabel_font_size=25,
                                    label_font_size=20, figure_size=(15, 8), title_font_size=30)

            # 画图
            self.my_KAN.plot()
            # 存储图片
            plt.savefig(self.Data_path + 'KAN/' + self.KAN_used_str + '/' + self.my_season_attribute
                        + '_' + self.my_columns_name_fliter + '/' + self.str_nn + '/'
                        + 'old_KAN_model_structure_final.png', dpi=300)
            print('+' * 20)
            print('Save picture to ' + self.Data_path + 'KAN/' + self.KAN_used_str + '/'
                  + self.my_season_attribute + '_' + self.my_columns_name_fliter + '/' + self.str_nn
                  + '/' + 'old_KAN_model_structure_final.png')
            print('+' * 20)
            # 显示图片
            if plot_show_flag:
                plt.show()

        # 打印公式
        formula, variables = self.my_KAN.symbolic_formula()
        print(formula[0])
        # 导出至txt
        write_context = ['Model Formula: \n', str(formula[0])]
        write_to_txt(self.Data_path + 'KAN/' + self.KAN_used_str + '/' + self.my_season_attribute
                     + '_' + self.my_columns_name_fliter + '/' + self.str_nn, 'model_formula.txt',
                     write_context)

        return self.my_KAN

    def implement_train_KAN_model(self, epochs=50, optimizer='Adam', optimizer_learning_rate=0.01,
                                  criterion='MSE', lr_scheduler_flag=False, plot_show_flag=True,
                                  **kwargs):
        """
        打包，应用KAN进行训练
        :param epochs: 迭代次数
        :param optimizer: 优化器，例如LBFGS/Adam/AdamW
        :param optimizer_learning_rate: 学习率，0.05/0.01/0.005/0.001
        :param criterion: 计算损失函数，'MSE'等
        :param lr_scheduler_flag: 是否使用学习率衰减
        :param plot_show_flag: 是否画图
        :param kwargs: 其他参数，例如使用学习率衰减时需要的参数等
        :return: 模型训练的实例model_train、训练完的模型my_KAN
        """

        print('+' * 50)
        print(self.KAN_used_str + ' is used!')
        print('+' * 50)

        self.optimizer_learning_rate = optimizer_learning_rate
        if lr_scheduler_flag:
            self.scheduler_str = 'Decay'
        else:
            self.scheduler_str = 'NoDecay'

        # 展示网络结构细节及每层的参数总数
        print(self.my_KAN)  # 打印模型结构
        print("Total number of parameters: ", sum(p.numel() for p in self.my_KAN.parameters()))  # 打印模型参数数量
        print('-' * 50)
        # 导出至txt
        write_context = ["Model Structure: \n\n", str(self.my_KAN), "\n\nTotal number of parameters: ",
                         str(sum(p.numel() for p in self.my_KAN.parameters()))]
        write_to_txt(self.Data_path + 'KAN/' + self.KAN_used_str + '/' + self.my_season_attribute + '_'
                     + self.my_columns_name_fliter + '/' + self.str_nn + '_'
                     + str(self.optimizer_learning_rate) + '_' + self.scheduler_str,
                     'model_structure.txt', write_context)

        # 训练
        self.model_train = Init_Train_Kolmogorov_Arnold_Network(
            self.my_KAN, optimizer_algorithm=optimizer, optimizer_learning_rate=optimizer_learning_rate,
            criterion=criterion, lr_scheduler_flag=lr_scheduler_flag, **kwargs)

        my_KAN, train_loss_list, [start_time, end_time] = \
            self.model_train.train_KAN(self.res_DataLoader_dict['train_DataLoader'], epochs=epochs, **kwargs)
        # 导出至txt
        write_context = ["\nOptimizer : ", str(optimizer),
                         "\nOptimizer Learning Rate: ", str(optimizer_learning_rate),
                         "\nEpochs: ", str(epochs),
                         "\nTraining Start Time: ", str(start_time),
                         "\nTraining End Time: ", str(end_time),
                         "\nTraining time: ", str(end_time - start_time)]
        write_to_txt(self.Data_path + 'KAN/' + self.KAN_used_str + '/' + self.my_season_attribute
                     + '_' + self.my_columns_name_fliter + '/' + self.str_nn + '_'
                     + str(self.optimizer_learning_rate) + '_' + self.scheduler_str,
                     'model_structure.txt', write_context, mode='a')

        # 损失
        print('+' * 50)
        print('Train Loss: ', train_loss_list)
        # 导出至txt
        write_context = ['Train Loss: \n', str(train_loss_list)]
        write_to_txt(self.Data_path + 'KAN/' + self.KAN_used_str + '/' + self.my_season_attribute
                     + '_' + self.my_columns_name_fliter + '/' + self.str_nn + '_'
                     + str(self.optimizer_learning_rate) + '_' + self.scheduler_str, 'Train_Loss.txt',
                     write_context)
        print('+' * 50)
        # 画损失图
        my_plot_func.time_curve({'y': train_loss_list, 'color_curve': 'k-', 'curvename': 'lost'},
                                title_name='Train Loss', xlabelname='epochs', ylabelname='loss',
                                show_flag=plot_show_flag, dpi=300, save_dir_path=self.Data_path
                                                                                 + 'KAN/' + self.KAN_used_str + '/' + self.my_season_attribute
                                                                                 + '_' + self.my_columns_name_fliter + '/' + self.str_nn + '_'
                                                                                 + str(
                self.optimizer_learning_rate) + '_' + self.scheduler_str,
                                save_figure_name='train_loss.png', legend_font_size=30,
                                xlabel_font_size=25, ylabel_font_size=25, label_font_size=20,
                                figure_size=(15, 8), title_font_size=30)

        return my_KAN

    def implement_validate_KAN_model(self, validate_week_list, train_test_split_flag, validate_index,
                                     plot_show_flag=True, **kwargs):
        """
        打包，应用KAN进行验证
        :param validate_week_list: 验证的周list
        :param train_test_split_flag: 是否使用sklearn的train_test_split
        :param validate_index: 验证指数，用来区分第几次验证
        :param plot_show_flag: 是否画图
        :param kwargs: 其他参数
        :return:
        """
        print('-' * 50)
        print('Verification:')
        [val_predict_res, val_true_res], val_loss = \
            self.model_train.validate_test_model(self.my_KAN,
                                                 self.res_TensorDataset_dict['validate_tensor_dataset'])
        # 导出至txt
        write_context = ['Verification Predict Result: \n', str(val_predict_res)]
        write_to_txt(self.Data_path + 'KAN/' + self.KAN_used_str + '/' + self.my_season_attribute
                     + '_' + self.my_columns_name_fliter + '/' + self.str_nn + '_'
                     + str(self.optimizer_learning_rate) + '_' + self.scheduler_str,
                     'Verification_Predict_Data.txt', write_context)
        write_context = ['Verification True Result: \n', str(val_true_res)]
        write_to_txt(self.Data_path + 'KAN/' + self.KAN_used_str + '/' + self.my_season_attribute
                     + '_' + self.my_columns_name_fliter + '/' + self.str_nn + '_'
                     + str(self.optimizer_learning_rate) + '_' + self.scheduler_str,
                     'Verification_True_Data.txt', write_context)
        write_context = ['Verification Loss: \n', str(val_loss)]
        write_to_txt(self.Data_path + 'KAN/' + self.KAN_used_str + '/' + self.my_season_attribute
                     + '_' + self.my_columns_name_fliter + '/' + self.str_nn + '_'
                     + str(self.optimizer_learning_rate) + '_' + self.scheduler_str,
                     'Verification_Loss.txt', write_context)

        # 画图
        # 计算得到每一周的数据点数
        week_data_len = int((self.res_data_dict['validate_label'].shape[0]) / len(validate_week_list))

        if not train_test_split_flag:
            # 画图
            for i in range(len(validate_week_list)):
                outputs_validate_predict_res_temp = val_predict_res[i * week_data_len: (i + 1) * week_data_len]
                outputs_validate_true_res_temp = val_true_res[i * week_data_len: (i + 1) * week_data_len]
                my_plot_func.time_curve({'y': outputs_validate_predict_res_temp, 'color_curve': 'b-',
                                         'curvename': 'predict'}, {'y': outputs_validate_true_res_temp,
                                                                   'color_curve': 'r-', 'curvename': 'true'},
                                        x_time=self.res_data_dict['validate_label'].iloc[i * week_data_len:
                                                                                         (i + 1) * week_data_len,
                                               0].tolist(),
                                        title_name='Predicted and True Values in Verification',
                                        xlabelname='time',
                                        ylabelname=self.my_columns_name_fliter.replace('_', ' '),
                                        show_flag=plot_show_flag, dpi=300,
                                        save_dir_path=self.Data_path + 'KAN/' + self.KAN_used_str
                                                      + '/' + self.my_season_attribute + '_'
                                                      + self.my_columns_name_fliter + '/' + str(validate_index)
                                                      + '_verification/' + self.str_nn + '_'
                                                      + str(self.optimizer_learning_rate) + '_' + self.scheduler_str,
                                        save_figure_name='Predicted_and_True_Values_in_Verification_'
                                                         + str(i) + '.png', legend_font_size=20, xlabel_font_size=25,
                                        ylabel_font_size=25, label_font_size=20, figure_size=(20, 10),
                                        title_font_size=30)

        # RMSE、MAE、R^2指标
        val_result_rmse = np.sqrt(mean_squared_error(val_true_res, val_predict_res))
        val_result_mae = mean_absolute_error(val_true_res, val_predict_res)
        val_result_r2_score = r2_score(val_true_res, val_predict_res)
        print('RMSE,MAE,R^2: ', ','.join([str(val_result_rmse), str(val_result_mae), str(val_result_r2_score)]))
        print('+' * 30)
        print('-' * 50)
        # 导出至txt
        write_context = ['RMSE,MAE,R^2: \n', str(','.join([str(val_result_rmse), str(val_result_mae),
                                                           str(val_result_r2_score)]))]
        write_to_txt(self.Data_path + 'KAN/' + self.KAN_used_str + '/' + self.my_season_attribute
                     + '_' + self.my_columns_name_fliter + '/' + self.str_nn + '_'
                     + str(self.optimizer_learning_rate) + '_' + self.scheduler_str,
                     'Verification_Result.txt', write_context)

        return

    def implement_test_KAN_model(self, test_week_list, plot_show_flag=True, **kwargs):
        """
        打包，应用KAN进行测试
        :param test_week_list: 测试的周list
        :param plot_show_flag: 是否画图
        :param kwargs: 其他参数
        :return:
        """

        # 是否和NN比较
        compare_with_NN_flag = kwargs.get("compare_with_NN", False)
        if compare_with_NN_flag:
            NN_str_nn = kwargs.get("NN_str_nn", '[3,16,32,16,1]')
            NN_optimizer_learning_rate = kwargs.get("NN_optimizer_learning_rate", 0.005)
            grottoes_number = kwargs.get("grottoes_number", 9)
            # 源文件路径
            source_file_path = (self.Data_path + 'BP_NN/' + str(grottoes_number) + '_'
                                + self.my_season_attribute + '_'
                                + self.my_columns_name_fliter + '/' + NN_str_nn + '_'
                                + str(NN_optimizer_learning_rate) + '/')

            # 目标文件路径
            destination_file_path = (self.Data_path + 'KAN/' + self.KAN_used_str + '/'
                                     + self.my_season_attribute + '_'
                                     + self.my_columns_name_fliter + '/' + self.str_nn
                                     + '_' + str(self.optimizer_learning_rate) + '_'
                                     + self.scheduler_str + '/')
            # 复制并重命名文件
            shutil.copy2(source_file_path + 'model_structure.txt',
                         destination_file_path + 'NN_model_structure.txt')
            shutil.copy2(source_file_path + 'Test_Predict_Data.txt',
                         destination_file_path + 'NN_Test_Predict_Data.txt')
            shutil.copy2(source_file_path + 'Test_Result.txt',
                         destination_file_path + 'NN_Test_Result.txt')
        else:
            pass

        print('-' * 50)
        print('Test:')

        if self.KAN_used_str == 'old_KAN':
            with torch.no_grad():
                test_predict_res = self.my_KAN.forward(self.res_data_tensor_dict['test_input'].to(self.device))
            test_predict_res = test_predict_res.view(-1).cpu().tolist()
            test_true_res = self.res_data_tensor_dict['test_label'].view(-1).tolist()
        else:
            [test_predict_res, test_true_res], test_loss = \
                self.model_train.validate_test_model(self.my_KAN,
                                                     self.res_TensorDataset_dict['test_tensor_dataset'])
        # 导出至txt
        write_context = ['Test Predict Result: \n', str(test_predict_res)]
        write_to_txt(self.Data_path + 'KAN/' + self.KAN_used_str + '/' + self.my_season_attribute
                     + '_' + self.my_columns_name_fliter + '/' + self.str_nn + '_'
                     + str(self.optimizer_learning_rate) + '_' + self.scheduler_str,
                     'Test_Predict_Data.txt', write_context)
        write_context = ['Test True Result: \n', str(test_true_res)]
        write_to_txt(self.Data_path + 'KAN/' + self.KAN_used_str + '/' + self.my_season_attribute
                     + '_' + self.my_columns_name_fliter + '/' + self.str_nn + '_'
                     + str(self.optimizer_learning_rate) + '_' + self.scheduler_str,
                     'Test_True_Data.txt', write_context)

        # 计算得到每一周的数据点数
        week_data_len = int((self.res_data_dict['test_label'].shape[0]) / len(test_week_list))
        # 画图
        # 对比用的NN
        if compare_with_NN_flag:
            NN_res_txt_path = (self.Data_path + 'KAN/' + self.KAN_used_str + '/'
                               + self.my_season_attribute + '_'
                               + self.my_columns_name_fliter + '/' + self.str_nn + '_'
                               + str(self.optimizer_learning_rate) + '_' + self.scheduler_str)
            with open(NN_res_txt_path + '/' + 'NN_Test_Predict_Data.txt', 'r') as file:
                content = file.read()
                start_index = content.find('[')
                end_index = content.find(']')
                if start_index != -1 and end_index != -1:
                    # 使用 ast.literal_eval 安全地解析字符串为列表
                    data_str = content[start_index:end_index + 1]  # 包括括号
                    NN_temp_res = ast.literal_eval(data_str)
        else:
            NN_temp_res = None

        for i in range(len(test_week_list)):
            if compare_with_NN_flag:
                NN_temp_res_temp = NN_temp_res[i * week_data_len: (i + 1) * week_data_len]
            else:
                NN_temp_res_temp = None

            outputs_test_predict_res_temp = test_predict_res[i * week_data_len: (i + 1) * week_data_len]
            outputs_test_true_res_temp = test_true_res[i * week_data_len: (i + 1) * week_data_len]

            save_dir_path_temp = self.Data_path + 'KAN/' + self.KAN_used_str + '/' \
                                 + self.my_season_attribute + '_' + self.my_columns_name_fliter \
                                 + '/' + self.str_nn + '_' + str(self.optimizer_learning_rate) \
                                 + '_' + self.scheduler_str

            if self.my_columns_name_fliter == 'air_humidity':
                my_columns_name_fliter_temp = 'Air Relative Humidity'
            else:
                my_columns_name_fliter_temp = self.my_columns_name_fliter.replace('_', ' ').title()

            if compare_with_NN_flag:
                my_plot_func.time_curve({'y': outputs_test_true_res_temp,
                                         'color_curve': 'b-', 'curvename': 'True Curve'},
                                        {'y': NN_temp_res_temp,
                                         'color_curve': 'g-', 'curvename': 'NN Predict Curve'},
                                        {'y': outputs_test_predict_res_temp, 'color_curve': 'r-',
                                         'curvename': self.KAN_used_str.replace('_', '-') + 's Predict Curve'},
                                        x_time=self.res_data_dict['test_label'].iloc[i * week_data_len:
                                        (i + 1) * week_data_len, 0].tolist(),
                                        # title_name='Predicted and True Values in Test',
                                        xlabelname='Time',
                                        ylabelname=my_columns_name_fliter_temp,
                                        show_flag=plot_show_flag, dpi=300, save_dir_path=save_dir_path_temp,
                                        save_figure_name='Predicted_and_True_Values_in_Test_' + str(i) + '.png',
                                        legend_font_size=30, xlabel_font_size=30, ylabel_font_size=30,
                                        label_font_size=25, figure_size=(20, 10)
                                        # , title_font_size=30
                                        , rotation=0
                                        )
            else:
                my_plot_func.time_curve({'y': outputs_test_predict_res_temp, 'color_curve': 'b-',
                                         'curvename': self.KAN_used_str.replace('_', '-')
                                                      + ' Predict Curve'},
                                        {'y': outputs_test_true_res_temp,
                                         'color_curve': 'r-', 'curvename': 'True Curve'},
                                        x_time=self.res_data_dict['test_label'].iloc[i * week_data_len:
                                        (i + 1) * week_data_len, 0].tolist(),
                                        # title_name='Predicted and True Values in Test',
                                        xlabelname='Time',
                                        ylabelname=my_columns_name_fliter_temp,
                                        show_flag=plot_show_flag, dpi=300, save_dir_path=save_dir_path_temp,
                                        save_figure_name='Predicted_and_True_Values_in_Test_' + str(i) + '.png',
                                        legend_font_size=30, xlabel_font_size=30, ylabel_font_size=30,
                                        label_font_size=25, figure_size=(20, 10)
                                        # , title_font_size=30
                                        , rotation=0
                                        )

        # RMSE、MAE、R^2指标
        test_result_rmse = np.sqrt(mean_squared_error(test_true_res, test_predict_res))
        test_result_mae = mean_absolute_error(test_true_res, test_predict_res)
        test_result_r2_score = r2_score(test_true_res, test_predict_res)
        print('RMSE,MAE,R^2: ', ','.join([str(test_result_rmse), str(test_result_mae), str(test_result_r2_score)]))
        print('+' * 30)
        print('-' * 50)
        # 导出至txt
        write_context = ['RMSE,MAE,R^2: \n', str(','.join([str(test_result_rmse), str(test_result_mae),
                                                           str(test_result_r2_score)]))]
        write_to_txt(self.Data_path + 'KAN/' + self.KAN_used_str + '/' + self.my_season_attribute
                     + '_' + self.my_columns_name_fliter + '/' + self.str_nn + '_'
                     + str(self.optimizer_learning_rate) + '_' + self.scheduler_str,
                     'Test_Result.txt', write_context)

        return
