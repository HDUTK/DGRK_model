#!/usr/bin/env python3.8.10
# -*- coding: utf-8 -*-
"""
function description: 此文件用于一般的神经网络
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2020.2.5
time: 2024/5/24 21:19
version: V1.0
"""
import torch
import torch.utils.data
from torch import nn
import pandas as pd
from tqdm import tqdm
import datetime
import os
import sys

from LogicAlgorithm.Network_common_func import Network_common
from common.common_func import train_start_end_time


class NeuralNetwork(nn.Module):
    def __init__(self, Forward_Network_Layers_1: list, Forward_Network_Layers_2: list):
        """
        神经网络结构
        :param Forward_Network_Layers_1: 输入层至隐层的结构，例如
        [['Conv2d', 1, 32, (6, 4), (2, 2), (2, 2)],
         ['BatchNorm2d', 32], ['LeakyReLU', 0.2, True],
         ['Conv2d', 32, 64, (6, 5), (2, 2), (2, 2)],
         ['BatchNorm2d', 64], ['LeakyReLU', 0.2, True],
         ['Conv2d', 64, 128, (6, 4), (2, 2), (2, 2)]]
        :param Forward_Network_Layers_2: 隐层至输出层结构，例如
        [['ConvTranspose2d', 128, 64, (6, 4), (2, 2), (2, 2)],
         ['ReLU', True],
         ['ConvTranspose2d', 64, 32, (6, 5), (2, 2), (2, 2)],
         ['ReLU', True],
         ['ConvTranspose2d', 32, 1, (6, 4), (2, 2), (2, 2)],
         ['Sigmoid']]
        """
        super(NeuralNetwork, self).__init__()

        self.forward_propagation_list_1 = Forward_Network_Layers_1
        self.forward_propagation_list_2 = Forward_Network_Layers_2

        # 网络结构
        forward_propagation_seq_1 = Network_common.Network_Structure(self.forward_propagation_list_1)
        self.forward_propagation_layers_1 = nn.Sequential(*forward_propagation_seq_1)
        forward_propagation_seq_2 = Network_common.Network_Structure(self.forward_propagation_list_2)
        self.forward_propagation_layers_2 = nn.Sequential(*forward_propagation_seq_2)

    def forward(self, x_input):
        """
        前向传播
        :param x_input: 输入的张量
        :return: 输出最中间的隐层和最后一层张量
        """
        hidden_data = self.forward_propagation_layers_1(x_input)
        output_data = self.forward_propagation_layers_2(hidden_data)
        return hidden_data, output_data


class Init_Train_NeuralNetwork(object):
    """
    初始化、训练神经网络
    """

    def __init__(self, model, optimizer_algorithm='Adam', optimizer_learning_rate=0.01,
                 criterion='MSE', **kwargs):
        """
        初始化神经网络
        :param model: 神经网络模型，需要一个NeuralNetwork类生成的对象
        :param optimizer_algorithm: 优化算法，默认Adam
        :param optimizer_learning_rate: 优化算法的学习率，默认0.01
        :param criterion: 损失函数，默认MSE
        :param kwargs: 其他参数
        """
        self.model = model
        self.optimizer_learning_rate = optimizer_learning_rate
        self.optimizer_algorithm_str = optimizer_algorithm
        self.criterion_str = criterion

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

    def train_NN(self, train_loader, epochs=100, epochs_print_Flag=True, **kwargs):
        """
        训练神经网络模型
        :param train_loader: 输入的训练数据
        :param epochs: 训练的次数，默认100次
        :param epochs_print_Flag: 训练过程是否输出，默认为True
        :param kwargs: 其他参数
        :return: 训练完成的模型model和loss
        """
        loss_list = []

        # 开始训练
        # 训练起始时间
        start_time = train_start_end_time('start')

        for epoch in range(epochs):
            train_num = 0
            loss = 0
            with tqdm(train_loader, file=sys.stdout) as pbar:
                for batch_x, batch_y in pbar:

                    # if 'Conv' in self.model.encoder_list[0][0]:
                    #     # 首层为卷积相关层
                    #     batch_features = batch_features.to(self.device)
                    # else:
                    # 如果可能，放至GPU
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                    # 计算网络的输出
                    _, predict_y = self.model.forward(batch_x)

                    # 计算Loss
                    train_loss = self.criterion(predict_y, batch_y)

                    # 将梯度重置为0，等待重新计算
                    self.optimizer_algorithm.zero_grad()
                    # 反向计算梯度
                    train_loss.backward()
                    # 根据梯度优化网络的参数
                    self.optimizer_algorithm.step()

                    # 计算训练数量
                    train_num = train_num + batch_x.size(0)

                    # 将该个mini-batch的loss加入至这个epoch下的整体loss中
                    loss = loss + train_loss.item()

            # 计算这个epoch下的loss
            loss = loss / train_num
            loss_list.append(loss)

            # 输出该次epoch的Loss
            if epochs_print_Flag:
                # display the epoch training loss
                print("Epoch : {}/{}, Loss = {:.5f}".format(epoch + 1, epochs, loss))
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
                _, temp_outputs = my_model.forward(validate_test_X)

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

    def save_model(self, save_path, model_name='My_NN_Model'):
        """
        保存此时的模型
        :param save_path: 保存路径，例如'D:/result'
        :param model_name: 保存时的模型名字，默认：My_NN_Model
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
