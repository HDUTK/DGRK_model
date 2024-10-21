#!/usr/bin/env python3.8.10
# -*- coding: utf-8 -*-
"""
function description: 此文件用于云冈石窟BP网络
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2020.2.5
time: 2024/5/22 15:03
version: V1.0
"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from torchsummary import summary

from data.Dataset_util.Yungang_Grottoes.YunGang_Grottoes_Data_used import Yungang_Grottoes_Data_preprocessing
from LogicAlgorithm.ML_method.NN.Neural_Network import NeuralNetwork, Init_Train_NeuralNetwork
from common.plot_func import my_plot_func
from config.Yungang_Grottoes_config import *
from common.common_func import write_to_txt

"""
config
"""
# 5min/30min/1h
resample_time = '5min'

# 9/10
grottoes_number = 10

class_sensor_list = 'A03,B03,AB02'.split(',')

# air_temperature/air_humidity/wall_temperature
my_columns_name_fliter = 'air_humidity'

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


# 若可以，使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


# 初始化参数
str_nn = '[3,3,1]'
# my_NN = NeuralNetwork(Forward_Network_Layers_1=
#                       [['Linear', 3, 16, True],
#                        ['LeakyReLU', 0.2],
#                        ['Linear', 16, 32, True]],
#                       Forward_Network_Layers_2=
#                       [['Linear', 32, 16, True],
#                        ['LeakyReLU', 0.2],
#                        ['Linear', 16, 1, True]]
#                       ).to(device)
my_NN = NeuralNetwork(Forward_Network_Layers_1=
                      [['Linear', 3, 3, True],
                       ['LeakyReLU', 0.2]],
                      Forward_Network_Layers_2=
                      [['Linear', 3, 1, True]]
                      ).to(device)

# my_NN = NeuralNetwork(Forward_Network_Layers_1=
#                       [['Linear', 3, 3, True],
#                        ['LeakyReLU', 0.2]],
#                       Forward_Network_Layers_2=
#                       [['Linear', 3, 1, True]]
#                       ).to(device)

# 展示网络结构细节及每层的参数总数
summary(my_NN, [(1, 1, 3)], batch_size=my_batch_size, device='cuda' if torch.cuda.is_available() else "cpu")

print(my_NN)  # 打印模型结构
print("Total number of parameters: ", sum(p.numel() for p in my_NN.parameters()))  # 打印模型参数数量
print('-' * 50)

# 训练的初始化，0.05/0.01/0.005/0.001
optimizer_learning_rate = 0.05
optimizer_algorithm = 'Adam'
my_NN_train = Init_Train_NeuralNetwork(my_NN, optimizer_algorithm=optimizer_algorithm,
                                       optimizer_learning_rate=optimizer_learning_rate,
                                       criterion='MSE')  # Adam_weight_decay=0.1
# 导出至txt
write_context = ["Model Structure: \n\n", str(my_NN), "\n\nTotal number of parameters: ",
                 str(sum(p.numel() for p in my_NN.parameters()))]
write_to_txt(Data_path + 'BP_NN/' + my_season_attribute + '_' + my_columns_name_fliter
             + '/' + str_nn + '_' + str(optimizer_learning_rate), 'model_structure.txt',
             write_context)

# 训练
epochs = 50
my_NN, loss_list, time_list = my_NN_train.train_NN(res_DataLoader_dict['train_DataLoader'],
                                                   epochs=epochs, epochs_print_Flag=True)
# 导出至txt
write_context = ["\nOptimizer : ", str(optimizer_algorithm),
                 "\nOptimizer Learning Rate: ", str(optimizer_learning_rate),
                 "\nEpochs: ", str(epochs),
                 "\nTraining Start Time: ", str(time_list[0]),
                 "\nTraining End Time: ", str(time_list[1]),
                 "\nTraining time: ", str(time_list[1] - time_list[0])]
write_to_txt(Data_path + 'BP_NN/' + my_season_attribute + '_' + my_columns_name_fliter
             + '/' + str_nn + '_' + str(optimizer_learning_rate), 'model_structure.txt',
             write_context, mode='a')

# 损失
print('+' * 50)
print('Train Loss: ', loss_list)
# 导出至txt
write_context = ['Train Loss: \n', str(loss_list)]
write_to_txt(Data_path + 'BP_NN/' + my_season_attribute + '_' + my_columns_name_fliter
             + '/' + str_nn + '_' + str(optimizer_learning_rate), 'Train_Loss.txt', write_context)
print('+' * 50)
# 损失图
print('+' * 50)
print('Train Loss: ', loss_list)
print('+' * 50)
my_plot_func.time_curve({'y': loss_list, 'color_curve': 'k-', 'curvename': 'lost'},
                        title_name='Train Loss', xlabelname='epochs', ylabelname='loss',
                        show_flag=plot_show_flag, dpi=300, save_dir_path=Data_path + 'BP_NN/'
                        + my_season_attribute + '_' + my_columns_name_fliter + '/' + str_nn
                        + '_' + str(optimizer_learning_rate),
                        save_figure_name='train_loss.png', legend_font_size=30, xlabel_font_size=25,
                        ylabel_font_size=25, label_font_size=20, figure_size=(15, 8),
                        title_font_size=30)

"""
模型验证
"""
if validate_week_list:
    print('-' * 50)
    print('Verification:')
    [val_predict_res, val_true_res], val_loss = my_NN_train.validate_test_model(
        my_NN, res_TensorDataset_dict['validate_tensor_dataset'])
    # 导出至txt
    write_context = ['Verification Predict Result: \n', str(val_predict_res)]
    write_to_txt(Data_path + 'BP_NN/' + my_season_attribute + '_' + my_columns_name_fliter
                 + '/' + str_nn + '_' + str(optimizer_learning_rate),
                 'Verification_Predict_Data.txt', write_context)
    write_context = ['Verification True Result: \n', str(val_true_res)]
    write_to_txt(Data_path + 'BP_NN/' + my_season_attribute + '_' + my_columns_name_fliter
                 + '/' + str_nn + '_' + str(optimizer_learning_rate),
                 'Verification_True_Data.txt', write_context)
    write_context = ['Verification Loss: \n', str(val_loss)]
    write_to_txt(Data_path + 'BP_NN/' + my_season_attribute + '_' + my_columns_name_fliter
                 + '/' + str_nn + '_' + str(optimizer_learning_rate),
                 'Verification_Loss.txt', write_context)

    # 计算得到每一周的数据点数
    week_data_len = int((res_data_dict['validate_label'].shape[0]) / len(validate_week_list))
    # 画图
    for i in range(len(validate_week_list)):
        outputs_validate_predict_res_temp = val_predict_res[i * week_data_len: (i + 1) * week_data_len]
        outputs_validate_true_res_temp = val_true_res[i * week_data_len: (i + 1) * week_data_len]
        my_plot_func.time_curve({'y': outputs_validate_predict_res_temp, 'color_curve': 'b-',
                                 'curvename': 'predict'}, {'y': outputs_validate_true_res_temp,
                                                           'color_curve': 'r-', 'curvename': 'true'},
                                x_time=res_data_dict['validate_label'].iloc[i * week_data_len:
                                                                            (i + 1) * week_data_len, 0].tolist(),
                                title_name='Predicted and True Values in Verification',
                                xlabelname='time', ylabelname=my_columns_name_fliter.replace('_', ' '),
                                show_flag=plot_show_flag, dpi=300,
                                save_dir_path=Data_path + 'BP_NN/' + my_season_attribute + '_'
                                + my_columns_name_fliter + '/' + str(x) + '_verification/'
                                + str_nn + '_' + str(optimizer_learning_rate),
                                save_figure_name='Predicted_and_True_Values_in_Verification_'
                                + str(i) + '.png', legend_font_size=20, xlabel_font_size=25,
                                ylabel_font_size=25, label_font_size=20, figure_size=(20, 10),
                                title_font_size=30)

    val_result_rmse = np.sqrt(mean_squared_error(val_true_res, val_predict_res))
    val_result_mae = mean_absolute_error(val_true_res, val_predict_res)
    val_result_r2_score = r2_score(val_true_res, val_predict_res)

    print('RMSE,MAE,R^2: ', ','.join([str(val_result_rmse), str(val_result_mae), str(val_result_r2_score)]))
    print('+' * 30)
    print('-' * 50)
    # 导出至txt
    write_context = ['RMSE,MAE,R^2: \n', str(','.join([str(val_result_rmse), str(val_result_mae),
                                                       str(val_result_r2_score)]))]
    write_to_txt(Data_path + 'BP_NN/' + my_season_attribute + '_' + my_columns_name_fliter
                 + '/' + str_nn + '_' + str(optimizer_learning_rate),
                 'Verification_Result.txt', write_context)

"""
模型测试
"""
if test_flag:
    print('-' * 50)
    print('Test:')
    [test_predict_res, test_true_res], test_loss = my_NN_train.validate_test_model(
        my_NN, res_TensorDataset_dict['test_tensor_dataset'])
    # 导出至txt
    write_context = ['Test Predict Result: \n', str(test_predict_res)]
    write_to_txt(Data_path + 'BP_NN/' + my_season_attribute + '_' + my_columns_name_fliter
                 + '/' + str_nn + '_' + str(optimizer_learning_rate),
                 'Test_Predict_Data.txt', write_context)
    write_context = ['Test True Result: \n', str(test_true_res)]
    write_to_txt(Data_path + 'BP_NN/' + my_season_attribute + '_' + my_columns_name_fliter
                 + '/' + str_nn + '_' + str(optimizer_learning_rate),
                 'Test_True_Data.txt', write_context)

    # 计算得到每一周的数据点数
    week_data_len = int((res_data_dict['test_label'].shape[0]) / len(test_week_list))
    # 画图
    for i in range(len(test_week_list)):
        outputs_test_predict_res_temp = test_predict_res[i * week_data_len: (i + 1) * week_data_len]
        outputs_test_true_res_temp = test_true_res[i * week_data_len: (i + 1) * week_data_len]
        my_plot_func.time_curve({'y': outputs_test_predict_res_temp, 'color_curve': 'b-',
                                'curvename': 'predict'}, {'y': outputs_test_true_res_temp,
                                'color_curve': 'r-', 'curvename': 'true'},
                                x_time=res_data_dict['test_label'].iloc[i * week_data_len:
                                                                        (i + 1) * week_data_len, 0].tolist(),
                                title_name='Predicted and True Values in Test',
                                xlabelname='time', ylabelname=my_columns_name_fliter.replace('_', ' '),
                                show_flag=plot_show_flag, dpi=300, save_dir_path=Data_path
                                + 'BP_NN/' + my_season_attribute + '_' + my_columns_name_fliter
                                + '/' + str_nn + '_' + str(optimizer_learning_rate),
                                save_figure_name='Predicted_and_True_Values_in_Test_'
                                + str(i) + '.png', legend_font_size=20, xlabel_font_size=25,
                                ylabel_font_size=25, label_font_size=20, figure_size=(20, 10),
                                title_font_size=30)

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
    write_to_txt(Data_path + 'BP_NN/' + my_season_attribute + '_' + my_columns_name_fliter
                 + '/' + str_nn + '_' + str(optimizer_learning_rate),
                 'Test_Result.txt', write_context)

"""
模型保存
"""
my_NN_train.save_model(save_path=Data_path + 'BP_NN/' + my_season_attribute + '_'
                       + my_columns_name_fliter + '/' + str_nn + '_' + str(optimizer_learning_rate),
                       model_name='My_NN_Model')
