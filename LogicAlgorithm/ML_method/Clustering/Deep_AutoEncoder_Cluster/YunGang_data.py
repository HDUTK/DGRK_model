#!/usr/bin/env python3.8.10
# -*- coding: utf-8 -*-
"""
function description: 此文件用于云冈数据
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2020.2.5
time: 2024/7/20 14:53
version: V1.0
"""

import argparse

from LogicAlgorithm.ML_method.Clustering.Deep_AutoEncoder_Cluster.others.main import *


# # 5min/30min/1h
# Data_path = r'D:/PythonProject/MachineLearning/My_Dataset/YunGang_Grottoes_Data/SJD_23.0607-24.0406/after_interpolate_' + '5min' + '/'

# # air_temperature/air_humidity/wall_temperature
# # all/summer/winter
# Dataset_dataframe = (pd.read_csv(Data_path + '_' + 'summer' + '_' + 'air_temperature'
#                                  + '_features.CSV', header=0)).iloc[:, 1:]
#
# Dataset_dataframe = Dataset_dataframe.iloc[:, 1:]


class AutoEncoder_2(nn.Module):
    def __init__(self, AE_input_size):
        super(AutoEncoder_2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(AE_input_size, 500),
            nn.ReLU(True),
            nn.Linear(500, 100),
            nn.ReLU(True),
            nn.Linear(100, 50),
            nn.ReLU(True),
            nn.Linear(50, 10))
        self.decoder = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(True),
            nn.Linear(50, 100),
            nn.ReLU(True),
            nn.Linear(100, 500),
            nn.ReLU(True),
            nn.Linear(500, AE_input_size))
        self.model = nn.Sequential(self.encoder, self.decoder)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        x = self.model(x)
        return x


parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# 5min/30min/1h
resample_str = '5min'
# all/summer/winter
season_attribute = 'summer'
# air_temperature/air_humidity/wall_temperature/merge
column_str = 'merge'
# 取9窟还是10窟的SJD，A01-B06(10窟，用0代表)/A63-B68(9窟，用1代表)
SJD_name_flag = 9
# SJD_name = ['A01-B06', 'A63-B68'][SJD_name_flag]
SJD_name = {'10窟': 'A01-B06', '9窟': 'A63-B68'}[str(SJD_name_flag) + '窟']

# 先False确定最佳聚类数，再True注意当当前设置n_clusters和最好n_clusters不一致时，预训练的模型存储位置不一样
load_flag_AE = True
load_flag_train_model = False

parser.add_argument('--Data_path', default='D:/PythonProject/MachineLearning/My_Dataset/'
                                           'YunGang_Grottoes_Data/SJD_23.0626-24.0606/'
                                           'after_interpolate_' + resample_str + '/', type=str)
parser.add_argument('--season_attribute', default=season_attribute, type=str)
parser.add_argument('--column_str', default=column_str, type=str)

parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--pretrain_epochs', default=50, type=int)
parser.add_argument('--train_epochs', default=500, type=int)
parser.add_argument('--save_dir', default='saves', type=str)
parser.add_argument('--interval_plot', default=25, type=int)
parser.add_argument('--n_clusters', default=3, type=int)
# 可视化图时显示在图上的图例，必须按每行传感器的顺序
sensors_dict = {'10窟': ["A01", "A02", "A03", "A04", "A05", "A06",
                        "AB01", "AB02", "AB03", "AB04", "AB05", "AB06",
                        "B01", "B02", "B03", "B04", "B05", "B06"],
                '9窟': ["A63", "A64", "A65", "A66", "A67", "A68",
                       "AB07", "AB08", "AB09", "AB10", "AB11", "AB12",
                       "B63", "B64", "B65", "B66", "B67", "B68"]}
parser.add_argument('--picture_legend',
                    default=sensors_dict[str(SJD_name_flag) + '窟'],
                    type=list)

args = parser.parse_args()
print(args)
# AE的输入维度
if args.column_str == 'merge':
    input_size = 286
else:
    input_size = 143

epochs_pre = args.pretrain_epochs
batch_size = args.batch_size

# dataset
Dataset_dataframe = (pd.read_csv(args.Data_path + '_' + args.season_attribute + '_'
                                 + args.column_str + '_features_' + SJD_name + '.CSV', header=0)).iloc[:, 1:]
Dataset_dataframe = Dataset_dataframe.iloc[:, 1:]
x = torch.tensor(np.array(Dataset_dataframe))

autoencoder = AutoEncoder_2(input_size).to(device)

save_path_parents = args.Data_path + '/Deep_Cluster/' + str(SJD_name_flag) \
                    + '_' + args.season_attribute + '_' + args.column_str + '_' \
                    + str(args.n_clusters) + '_clusters' + '/'

# 创建路径
if not os.path.exists(save_path_parents):
    os.makedirs(save_path_parents)
if not os.path.exists(save_path_parents + 'plots/'):
    os.makedirs(save_path_parents + 'plots/')

ae_save_path = save_path_parents + 'sim_autoencoder.pth'

if load_flag_AE:
    if os.path.isfile(ae_save_path):
        print('Loading {}'.format(ae_save_path))
        checkpoint = torch.load(ae_save_path)
        autoencoder.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(ae_save_path))
        checkpoint = {
            "epoch": 1,
            "best": float("inf")
        }
else:
    checkpoint = {
        "epoch": 1,
        "best": float("inf")
    }

pretrain(data=x, model=autoencoder, num_epochs=epochs_pre, save_path_parents=save_path_parents,
         checkpoint=checkpoint, input_size=input_size)

if not load_flag_AE:
    # 确定聚类数
    X_train_temp = autoencoder.encode(x.to('cuda').float())
    n_cluster = get_n_cluster(X_train_temp.cpu().detach().numpy(), args.n_clusters,
                              save_path_parents)
    exit(0)

dec_save_path = save_path_parents + 'dec.pth'
dec = DEC(n_clusters=args.n_clusters, autoencoder=autoencoder, hidden=10, cluster_centers=None,
          alpha=1.0).to(device)

if load_flag_train_model:
    if os.path.isfile(dec_save_path):
        print('Loading {}'.format(dec_save_path))
        checkpoint = torch.load(dec_save_path)
        dec.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(dec_save_path))
        checkpoint = {
            "epoch": 1,
            "best": float("inf")
        }
else:
    checkpoint = {
        "epoch": 1,
        "best": float("inf")
    }

train(data=x, model=dec, num_epochs=args.train_epochs, save_path_parents=save_path_parents,
      checkpoint=checkpoint, n_clusters=args.n_clusters, interval_plot=args.interval_plot,
      picture_legend=args.picture_legend)

