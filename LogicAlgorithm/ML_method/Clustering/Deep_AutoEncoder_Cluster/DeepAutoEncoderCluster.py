#!/usr/bin/env python3.8.10
# -*- coding: utf-8 -*-
"""
function description: 此文件用于基于自编码器的深度聚类，源码来自
https://github.com/xuyxu/Deep-Clustering-Network
https://github.com/Deepayan137/DeepClustering/tree/master
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2020.2.5
time: 2024/7/13 13:53
version: V1.0
"""

import os
import torch
from torch import nn
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from common.common_func import train_start_end_time
from LogicAlgorithm.Network_common_func import Network_common
from LogicAlgorithm.DL_method.AutoEncoder.Auto_Encoder import AutoEncoder, Init_Train_AutoEncoder


class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters=10, hidden=10, cluster_centers=None, alpha=1.0):
        super(ClusteringLayer, self).__init__()

        # 如果可以，则使用GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.hidden = hidden
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.n_clusters,
                self.hidden,
                dtype=torch.float
            ).to(self.device)
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = nn.Parameter(initial_cluster_centers)

    def forward(self, x):
        norm_squared = torch.sum((x.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        t_dist = (numerator.t() / torch.sum(numerator, 1)).t()  # soft assignment using t-distribution
        return t_dist


class DECN(nn.Module):
    def __init__(self, autoencoder, cluster_input_dim, n_clusters, cluster_centers=None, alpha=1.0,
                 **kwargs):
        super(DECN, self).__init__()

        # 如果可以，则使用GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

        self.autoencoder = autoencoder.to(self.device)

        self.cluster_input_dim = cluster_input_dim
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.cluster_centers = cluster_centers

        self.clusteringlayer = ClusteringLayer(n_clusters, self.cluster_input_dim,
                                               self.cluster_centers, self.alpha)

    @staticmethod
    def target_distribution(q_):
        weight = (q_ ** 2) / torch.sum(q_, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def forward(self, x):
        y, _ = self.autoencoder.forward(x)
        z = self.clusteringlayer(y)
        return y, z

    def visualize(self, epoch, x, y):
        from matplotlib import pyplot as plt
        from sklearn.manifold import TSNE

        fig = plt.figure()
        ax = plt.subplot(111)
        _, x = self.autoencoder.forward(x)
        x = x.detach().cpu().numpy()[:2000]
        x_embedded = TSNE(n_components=2).fit_transform(x)
        plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=y[:2000], cmap='tab10', s=1)
        fig.savefig('D:/PythonProject/MachineLearning/LogicAlgorithm/ML_method/Clustering/'
                    'Deep_AutoEncoder_Cluster/save//plots/mnist_{}.png'.format(epoch))
        plt.close(fig)


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment
    # ind = linear_assignment(w.max() - w)
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T

    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


class Init_train_DECN(object):
    def __init__(self, DECN_model, optimizer_algorithm='SDG', optimizer_learning_rate=0.1,
                 ae_criterion='MSE', cluster_criterion='KLDivLoss', ae_lamda=1, cluster_beta=1,
                 **kwargs):

        self.model = DECN_model
        self.optimizer_learning_rate = optimizer_learning_rate
        self.optimizer_algorithm_str = optimizer_algorithm
        self.ae_criterion_str = ae_criterion
        self.cluster_criterion_str = cluster_criterion

        # coefficient of the clustering term
        self.beta = cluster_beta
        # coefficient of the reconstruction term
        self.lamda = ae_lamda

        # 如果可以，则使用GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

        # 优化算法
        self.optimizer_algorithm = Network_common.optimizer_algorithm(
            self.model, self.optimizer_algorithm_str,
            optimizer_learning_rate=self.optimizer_learning_rate, **kwargs)

        # 损失函数
        self.AE_criterion = Network_common.loss_function(self.ae_criterion_str, self.device, **kwargs)
        self.Cluster_criterion = Network_common.loss_function(self.cluster_criterion_str,
                                                              self.device, **kwargs)

    def init_DECN(self, train_loader_temp, **kwargs):
        # 将autoencoder转为预测模式
        self.model.autoencoder.eval()
        # Initialize clusters in self.model.kmeans after pre-training
        batch_X = []
        for batch_idx, data in enumerate(train_loader_temp):
            batch_size_temp = data.size()[0]
            data = data.to(self.device).view(batch_size_temp, -1)
            # with torch.no_grad():
            latent_X, _ = self.model.autoencoder.forward(data.float())
            batch_X.append(latent_X.detach().cpu())
        batch_X = torch.cat(batch_X)

        # ============K-means=======================================
        kmeans = KMeans(n_clusters=10, random_state=0, n_init='auto').fit(batch_X)
        cluster_centers = kmeans.cluster_centers_
        cluster_centers = torch.tensor(cluster_centers, dtype=torch.float).to(self.device)
        self.model.clusteringlayer.cluster_centers = torch.nn.Parameter(cluster_centers)
        # =========================================================
        y_pred = kmeans.predict(batch_X)
        accuracy = acc(train_dataset.targets.cpu().numpy(), y_pred)
        print('Initial Accuracy: {}'.format(accuracy))

        print('plotting')
        visualize_data_x = kwargs.get("visualize_data_x", None)
        visualize_data_y = kwargs.get("visualize_data_y", None)
        img = visualize_data_x.float().to(self.device)
        self.model.visualize(-1, img, visualize_data_y)

        return

    """ Compute the Equation (5) in the original paper on a data batch """
    def _loss(self, X, Rec_X, Cluster_Y, only_use_cluster_loss=False):

        # Reconstruction error
        rec_loss = self.lamda * self.AE_criterion(X, Rec_X)

        # Regularization term on clustering
        target = self.model.target_distribution(Cluster_Y).detach()
        dist_loss = self.beta * (self.Cluster_criterion(Cluster_Y.log(), target) / Cluster_Y.shape[0])

        if only_use_cluster_loss:
            loss = dist_loss
        else:
            loss = rec_loss + dist_loss

        return (loss,
                rec_loss.detach().cpu().numpy(),
                dist_loss.detach().cpu().numpy())

    def train(self, train_loader_temp, train_data, n_clusters, epochs=200, epochs_print_Flag=True,
              batch_idx_print_flag=False, print_interval=1, **kwargs):
        # batch_X = []
        # for batch_idx, data in enumerate(train_loader_temp):
        #     # batch_size_temp = data.size()[0]
        #     # data = data.to(self.device).view(batch_size_temp, -1)
        #     img = data.float().to(self.device)
        #     # with torch.no_grad():
        #     latent_X, _ = self.model.autoencoder.forward(img)
        #     batch_X.append(latent_X.detach().cpu())
        # batch_X = torch.cat(batch_X)
        #
        # # ============K-means=======================================
        # kmeans = KMeans(n_clusters=10, random_state=0, n_init='auto').fit(batch_X)
        # cluster_centers = kmeans.cluster_centers_
        # cluster_centers = torch.tensor(cluster_centers, dtype=torch.float).to(self.device)
        # self.model.clusteringlayer.cluster_centers = torch.nn.Parameter(cluster_centers)
        # # =========================================================
        # y_pred = kmeans.predict(batch_X)
        # y = train_dataset.targets.cpu().numpy()
        # accuracy = acc(y, y_pred)
        # print('Initial Accuracy: {}'.format(accuracy))


        # 将autoencoder转为训练模式
        self.model.autoencoder.train()

        print('plotting')
        visualize_data_x = kwargs.get("visualize_data_x", None)
        visualize_data_y = kwargs.get("visualize_data_y", None)
        img = visualize_data_x.float().to(self.device)
        self.model.visualize(-1, img, visualize_data_y)

        # 开始训练
        # 训练起始时间
        start_time = train_start_end_time('start')

        # loss_list = []
        # rec_loss_list = []
        # dist_loss_list = []

        optimizer = torch.optim.SGD(params=self.model.parameters(), lr=0.1, momentum=0.9)
        loss_func = nn.KLDivLoss(size_average=False)

        for epoch in range(epochs):

            # loss_list_batch = []
            # rec_loss_list_batch = []
            # dist_loss_list_batch = []
            # count = 0

            batch = train_data
            img = batch.float()
            img = img.to(device)
            _, output = self.model.forward(img)
            target = self.model.target_distribution(output).detach()
            out = output.argmax(1)
            if epoch % 20 == 0:
                print('plotting')
                visualize_data_x = kwargs.get("visualize_data_x", None)
                visualize_data_y = kwargs.get("visualize_data_y", None)
                img = visualize_data_x.float().to(self.device)
                self.model.visualize(epoch, img, visualize_data_y)
            loss = loss_func(output.log(), target) / output.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accuracy = acc(train_dataset.targets.cpu().numpy(), out.cpu().numpy())

            print('Epochs: [{}/{}] Accuracy:{}, Loss:{}'.format(epoch, epochs, accuracy, loss))

            # for batch_idx, data in enumerate(train_loader_temp):
            #     batch_size_temp = data.size()[0]
            #     data = data.view(batch_size_temp, -1).to(self.device)
            #
            #     # with torch.no_grad():
            #     _, rec_x = self.model.autoencoder.forward(data)
            #     _, cluster_res = self.model.forward(data)
            #
            #     loss, rec_loss, dist_loss = self._loss(X=data, Rec_X=rec_x, Cluster_Y=cluster_res,
            #                                            only_use_cluster_loss=True)
            #
            #     self.optimizer_algorithm.zero_grad()
            #     loss.backward()
            #     self.optimizer_algorithm.step()
            #
            #     if batch_idx_print_flag and batch_idx % print_interval == 0:
            #         msg = 'Epoch: {:02d} | Batch: {:03d} | Loss: {:.5f} | Rec-Loss: {:.5f} | ' \
            #               'Dist-Loss: {:.5f}'
            #         print(msg.format(epoch+1, batch_idx,
            #                          loss.detach().cpu().numpy(),
            #                          rec_loss, dist_loss))
            #
            #     count = count + 1
            #
            # visualize_data_x = kwargs.get("visualize_data_x", None)
            # visualize_data_y = kwargs.get("visualize_data_y", None)
            # if (visualize_data_x is not None) and (visualize_data_y is not None):
            #     print('plotting')
            #     img = visualize_data_x.float().to(self.device)
            #     self.model.visualize(epoch+1, img, visualize_data_y)

            # save loss, rec_loss, dist_loss of this epoch
            # loss_epoch = sum(loss_list_batch) / count
            # rec_loss_epoch = sum(rec_loss_list_batch) / count
            # dist_loss_epoch = sum(dist_loss_list_batch) / count
            # loss_epoch = sum(loss_list_batch)
            # rec_loss_epoch = sum(rec_loss_list_batch)
            # dist_loss_epoch = sum(dist_loss_list_batch)
            # loss_list.append(loss_epoch)
            # rec_loss_list.append(rec_loss_epoch)
            # dist_loss_list.append(dist_loss_epoch)

            # if epochs_print_Flag:
            #     msg = 'Epoch: {:02d} | Loss: {:.5f} | Rec-Loss: {:.5f} | Dist-Loss: {:.5f}'
            #     print(msg.format(epoch+1, loss_epoch, rec_loss_epoch, dist_loss_epoch))
            # else:
            #     pass

        # 结束训练
        # 训练结束时间
        end_time = train_start_end_time('end')
        # 训练时长
        print('Training time: ', end_time - start_time)
        print('-----------------------------------------------')

        return self.model, loss_list, rec_loss_list, dist_loss_list

    def save_model(self, save_path_temp, model_name='My_DECN_Model'):
        """
        保存此时的模型
        :param save_path_temp: 保存路径，例如'D:/result'
        :param model_name: 保存时的模型名字，默认：My_DECN_Model
        :return:
        """
        if not os.path.exists(save_path_temp):
            os.mkdir(save_path_temp)
        else:
            pass

        Path = save_path_temp + '/' + model_name + '.pt'

        torch.save(self.model, Path)
        print('===============================')
        print('Save Model to (' + Path + ')!')
        print('===============================')
        return

    def visualize(self, epoch, x_temp, y_temp):

        from sklearn.manifold import TSNE
        from matplotlib import pyplot as plt

        fig = plt.figure()
        ax = plt.subplot(111)
        with torch.no_grad():
            x_temp, _ = self.model.autoencoder.forward(x_temp)
        x_temp = x_temp.detach().cpu().numpy()[:2000]
        x_embedded = TSNE(n_components=2).fit_transform(x_temp)
        plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=y_temp[:2000], cmap='tab10', s=1)
        fig.savefig('D:/PythonProject/MachineLearning/LogicAlgorithm/ML_method/Clustering/'
                    'Deep_AutoEncoder_Cluster/save//plots/mnist_{}.png'.format(epoch), dpi=300)
        plt.close(fig)


if __name__ == '__main__':
    import numpy as np
    from data.Dataset_util.MINST.MINST_Data import load_dataset

    # 如果可以，则使用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # batch_size
    batch_size = 128
    # pre-train flag
    pre_train_flag = True

    train_dataset, test_dataset, train_loader_orl, test_loader_orl = \
        load_dataset(batch_size=batch_size)
    train_dataset.data = np.divide(train_dataset.data.reshape((train_dataset.data.shape[0], -1)), 255.)
    test_dataset.data = np.divide(test_dataset.data.reshape((test_dataset.data.shape[0], -1)), 255.)
    # 创建数据加载器
    train_loader = DataLoader(dataset=train_dataset.data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset.data, batch_size=batch_size, shuffle=True)

    auto_encoder = AutoEncoder(
        Encoded_Layers=[['Linear', 28 * 28, 500], ['ReLU', True], ['Linear', 500, 500], ['ReLU', True],
                        ['Linear', 500, 500], ['ReLU', True], ['Linear', 500, 2000], ['ReLU', True],
                        ['Linear', 2000, 10]],
        Decoded_Layers=[['Linear', 10, 2000], ['ReLU', True], ['Linear', 2000, 500], ['ReLU', True],
                        ['Linear', 500, 500], ['ReLU', True], ['Linear', 500, 500], ['ReLU', True],
                        ['Linear', 500, 28 * 28], ['ReLU', True]]).to(device)

    save_path = 'D:/PythonProject/MachineLearning/LogicAlgorithm/ML_method/Clustering/' \
                'Deep_AutoEncoder_Cluster/save/'

    if pre_train_flag or (not pre_train_flag and not os.path.isfile(save_path + 'Pre_AE_model.pt')):
        # 创建自编码器的预训练实例对象
        train_AE_model_object = Init_Train_AutoEncoder(model=auto_encoder, optimizer_algorithm='Adam',
                                                       optimizer_learning_rate=0.001, criterion='MSE',
                                                       Adam_weight_decay=1e-6)
        auto_encoder, pre_loss_list = train_AE_model_object.train_AE(train_loader=train_loader,
                                                                     epochs=20,
                                                                     epochs_print_Flag=True)
        train_AE_model_object.save_model(save_path=save_path[: -1], model_name='Pre_AE_model')
    else:
        auto_encoder = torch.load(save_path + 'Pre_AE_model.pt')
        print('+' * 50)
        print('Loading {} !'.format(save_path + 'Pre_AE_model.pt'))
        print('+' * 50)


    # 创建DECN的实例对象
    print('-' * 50)
    print('Create DECN model !')
    decn = DECN(autoencoder=auto_encoder, cluster_input_dim=10, n_clusters=10).to(device)
    Init_train_DECN_object = Init_train_DECN(DECN_model=decn, optimizer_algorithm='SGD',
                                             optimizer_learning_rate=0.1,
                                             ae_criterion='MSE', cluster_criterion='KLDivLoss',
                                             SGD_momentum=0.9)

    # 初始化DECN
    print('-' * 50)
    Init_train_DECN_object.init_DECN(train_loader_temp=train_loader,
                                     visualize_data_x=train_dataset.data,
                                     visualize_data_y=train_dataset.targets)
    print('Initialize DECN model !')
    print('-' * 50)


    # 训练DECN
    Init_train_DECN_object.train(train_loader_temp=train_loader, train_data=train_dataset.data,
                                 n_clusters=10, epochs=200,
                                 epochs_print_Flag=True, batch_idx_print_flag=False,
                                 print_interval=1, visualize_data_x=train_dataset.data,
                                 visualize_data_y=train_dataset.targets)
