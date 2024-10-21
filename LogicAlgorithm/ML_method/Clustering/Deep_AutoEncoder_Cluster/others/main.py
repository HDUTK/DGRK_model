import os
import time
import pdb
from tqdm import *
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import Parameter
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from sklearn.cluster import KMeans
import numpy as np
from tqdm import *
from LogicAlgorithm.ML_method.Clustering.Deep_AutoEncoder_Cluster.others.metrics import *
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import pandas as pd

from adjustText import adjust_text

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, 2000),
            nn.ReLU(True),
            nn.Linear(2000, 10))
        self.decoder = nn.Sequential(
            nn.Linear(10, 2000),
            nn.ReLU(True),
            nn.Linear(2000, 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, 28 * 28))
        self.model = nn.Sequential(self.encoder, self.decoder)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        x = self.model(x)
        return x


class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters=10, hidden=10, cluster_centers=None, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.hidden = hidden
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.n_clusters,
                self.hidden,
                dtype=torch.float
            ).cuda()
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, x):
        norm_squared = torch.sum((x.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        t_dist = (numerator.t() / torch.sum(numerator, 1)).t()  # soft assignment using t-distribution
        return t_dist


class DEC(nn.Module):
    def __init__(self, n_clusters=10, autoencoder=None, hidden=10, cluster_centers=None, alpha=1.0):
        super(DEC, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.hidden = hidden
        self.cluster_centers = cluster_centers
        self.autoencoder = autoencoder
        self.clusteringlayer = ClusteringLayer(self.n_clusters, self.hidden, self.cluster_centers, self.alpha)

    def target_distribution(self, q_):
        weight = (q_ ** 2) / torch.sum(q_, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def forward(self, x):
        x = self.autoencoder.encode(x)
        return self.clusteringlayer(x)

    def visualize(self, epoch, x, labels, n_cluster, save_path, picture_legend=None):

        fig = plt.figure()
        ax = plt.subplot(111)
        x = self.autoencoder.encode(x).detach()
        x = x.cpu().numpy()
        x_embedded = TSNE(n_components=2, perplexity=5).fit_transform(x)

        # Times New Roman
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

        unique_labels = np.unique(labels)
        for label in unique_labels:
            indices = np.where(labels == label)
            plt.scatter(x_embedded[indices, 0], x_embedded[indices, 1], label="Cluster " + str(label + 1))

        # plt.scatter(x_embedded[:, 0], x_embedded[:, 1])
        plt.legend()

        if picture_legend:
            legend_plot = []
            for i, txt in enumerate(picture_legend):
                legend_temp = plt.annotate(txt, (x_embedded[i, 0], x_embedded[i, 1]), fontsize=12)
                legend_plot.append(legend_temp)

            # 调整标签位置
            # adjust_text(legend_plot, arrowprops=dict(arrowstyle='->', color='red'))
            adjust_text(legend_plot)
        else:
            pass

        # Times New Roman
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
        plt.tick_params(axis='x', labelsize=20)
        plt.tick_params(axis='y', labelsize=20)
        plt.legend(fontsize=20, loc='upper right')
        plt.tight_layout()
        fig.savefig(save_path + '/plots/YunGang_{}.png'.format(epoch), dpi=300)
        plt.close(fig)


def add_noise(img):
    noise = torch.randn(img.size()) * 0.2
    noisy_img = img + noise
    return noisy_img


def save_checkpoint(state, filename, is_best):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print("=> Saving new checkpoint")
        torch.save(state, filename)
    else:
        print("=> Validation Accuracy did not improve")


def pretrain(**kwargs):
    data = kwargs['data']
    model = kwargs['model']
    num_epochs = kwargs['num_epochs']
    save_path_parents = kwargs['save_path_parents']
    checkpoint = kwargs['checkpoint']
    input_size = kwargs['input_size']
    start_epoch = checkpoint['epoch']
    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=1e-3, weight_decay=1e-5)
    train_loader = DataLoader(dataset=data,
                              batch_size=1,
                              shuffle=False)
    for epoch in range(start_epoch, num_epochs):
        for data in train_loader:
            img = data.float()
            # noisy_img = add_noise(img)
            # noisy_img = noisy_img.to(device)
            img = img.to(device)
            # ===================forward=====================
            output = model(img)
            output = output.squeeze(1)
            output = output.view(output.size(0), input_size)
            loss = nn.MSELoss()(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ===================log========================
        print('epoch [{}/{}], MSE_loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.item()))
        state = loss.item()
        is_best = False
        if state < checkpoint['best']:
            checkpoint['best'] = state
            is_best = True

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best': state,
            'epoch': epoch
        }, save_path_parents + 'sim_autoencoder.pth',
            is_best)


def get_n_cluster(X_train, n_cluster, save_path_temp):
    from sklearn.metrics import silhouette_score

    # 设置matplotlib支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    K = range(2, 18)
    score_1, score_2 = [], []
    for k in K:
        kmeans_temp = KMeans(n_clusters=k, init='k-means++', random_state=0, n_init=10)
        kmeans_temp.fit(X_train)
        # 轮廓得分(Silhouette Score)是评价聚类算法性能的一种指标,它结合了簇内平均距离和簇间平均距离,值越大表示聚类效果越好
        score_1.append(silhouette_score(X_train, kmeans_temp.labels_, metric='euclidean'))
        # 肘部法判断聚类中心
        score_2.append(kmeans_temp.inertia_)

    # Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

    plt.plot(K, score_1, 'r*-')
    plt.xlabel('Number of Clusters(k)', fontsize=20)
    plt.ylabel('Silhouette Score', fontsize=20)
    plt.tick_params(axis='x', labelsize=20)
    plt.xticks(np.arange(1, 18, 2))
    plt.tick_params(axis='y', labelsize=20)
    plt.tight_layout()
    # plt.title(u'轮廓系数确定最佳的K值')
    plt.savefig(save_path_temp + '/plots/_轮廓系数确定最佳的K值.png', dpi=300)
    plt.close()

    plt.plot(K, score_2, 'r*-')
    plt.xlabel('k')
    plt.ylabel(u'Silhouette Score')
    # plt.title(u'手肘法确定最佳的K值')
    plt.savefig(save_path_temp + '/plots/_手肘法确定最佳的K值.png', dpi=300)
    plt.close()

    file = open(save_path_temp + '/plots/score_轮廓系数.txt', mode='w')
    file.write(str(score_1))
    file.close()
    file = open(save_path_temp + '/plots/score_手肘法.txt', mode='w')
    file.write(str(score_2))
    file.close()

    return


def train(**kwargs):
    data = kwargs['data']
    # labels = kwargs['labels']
    model = kwargs['model']
    num_epochs = kwargs['num_epochs']
    # savepath = kwargs['savepath']
    save_path_parents = kwargs['save_path_parents']
    checkpoint = kwargs['checkpoint']
    start_epoch = checkpoint['epoch']

    interval_plot = kwargs['interval_plot']

    n_clusters = kwargs.get("n_clusters", 3)
    picture_legend = kwargs.get("picture_legend", None)

    features = []
    train_loader = DataLoader(dataset=data,
                              batch_size=1,
                              shuffle=False)

    for i, batch in enumerate(train_loader):
        img = batch.float()
        img = img.to(device)
        features.append(model.autoencoder.encode(img).detach().cpu())
    features = torch.cat(features)
    # ============K-means=======================================
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit(features)
    cluster_centers = kmeans.cluster_centers_
    cluster_centers = torch.tensor(cluster_centers, dtype=torch.float).cuda()
    model.clusteringlayer.cluster_centers = torch.nn.Parameter(cluster_centers)
    # =========================================================
    # y_pred = kmeans.predict(features)
    # accuracy = acc(y.cpu().numpy(), y_pred)
    # print('Initial Accuracy: {}'.format(accuracy))

    # 新增可视化
    batch = data
    img = batch.float()
    img = img.to(device)
    labels = kmeans.labels_
    model.visualize(0, img, labels, n_clusters, save_path_parents, picture_legend)


    # loss_function = nn.KLDivLoss(size_average=False)
    loss_function = nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1, momentum=0.9)
    print('Training')
    row = []
    for epoch in range(start_epoch, num_epochs + 1):

        batch = data
        img = batch.float()
        img = img.to(device)
        output = model(img)
        target = model.target_distribution(output).detach()
        # out = output.argmax(1)

        loss = loss_function(output.log(), target) / output.shape[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # accuracy = acc(y.cpu().numpy(), out.cpu().numpy())
        # row.append([epoch, accuracy])
        # print('Epochs: [{}/{}] Accuracy:{}, Loss:{}'.format(epoch+1, num_epochs, accuracy, loss))
        print('Epochs: [{}/{}] , Loss:{}'.format(epoch, num_epochs, loss))

        # 新增可视化
        if epoch % interval_plot == 0:
            print('plotting')
            output = model(img)
            labels = kmeans.labels_
            model.visualize(epoch, img, labels, n_clusters, save_path_parents, picture_legend)

        state = loss.item()
        is_best = False
        if state < checkpoint['best']:
            checkpoint['best'] = state
            is_best = True

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best': state,
            'epoch': epoch
        }, save_path_parents+'dec.pth',
            is_best)

    # df = pd.DataFrame(row, columns=['epochs', 'accuracy'])
    # df.to_csv('log.csv')


def load_mnist():
    # the data, shuffled and split between train and test sets
    train = MNIST(root='./data/',
                  train=True,
                  transform=transforms.ToTensor(),
                  download=True)

    test = MNIST(root='./data/',
                 train=False,
                 transform=transforms.ToTensor())
    x_train, y_train = train.train_data, train.train_labels
    x_test, y_test = test.test_data, test.test_labels
    x = torch.cat((x_train, x_test), 0)
    y = torch.cat((y_train, y_test), 0)
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)
    print('MNIST samples', x.shape)
    return x, y


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--pretrain_epochs', default=20, type=int)
    parser.add_argument('--train_epochs', default=200, type=int)
    parser.add_argument('--save_dir', default='saves')
    args = parser.parse_args()
    print(args)
    epochs_pre = args.pretrain_epochs
    batch_size = args.batch_size

    x, y = load_mnist()
    autoencoder = AutoEncoder().to(device)
    save_path = 'D:/PythonProject/Cluster_Projects/DeepClustering-master/save/'

    if os.path.isfile(save_path+'sim_autoencoder.pth'):
        print('Loading {}'.format(save_path+'sim_autoencoder.pth'))
        checkpoint = torch.load(save_path+'sim_autoencoder.pth')
        autoencoder.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(save_path+'sim_autoencoder.pth'))
        checkpoint = {
            "epoch": 0,
            "best": float("inf")
        }
    pretrain(data=x, model=autoencoder, num_epochs=epochs_pre, save_path_parents=save_path, checkpoint=checkpoint)

    dec = DEC(n_clusters=10, autoencoder=autoencoder, hidden=10, cluster_centers=None, alpha=1.0).to(device)
    if os.path.isfile(save_path+'dec.pth'):
        print('Loading {}'.format(save_path+'dec.pth'))
        checkpoint = torch.load(save_path+'dec.pth')
        dec.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(save_path+'dec.pth'))
        checkpoint = {
            "epoch": 0,
            "best": float("inf")
        }
    train(data=x, labels=y, model=dec, num_epochs=args.train_epochs, savepath=save_path, checkpoint=checkpoint)
