#!/usr/bin/env python3.6.8
# -*- coding: utf-8 -*-
"""
function description: 此文件用于神经网络公共函数
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2021.2.3
time: 2023/4/26 14:34
version: V1.0
"""
from torch import nn, optim


class Network_common(object):
    """
    神经网络公共函数
    """

    def __init__(self):
        pass

    @staticmethod
    def Network_Structure(Layers: list):
        """
        神经网络的结构构造函数，输入形如：
        [['Linear',3,12],['LeakyReLU', 0.2, False],['Linear',12,48],['LeakyReLU'],
        ['Linear',48,12],['LeakyReLU'],['Linear',12,3],['Tanh']]
        :param Layers: 输入列表
        :return: 返回layers_seq列表用于nn.Sequential(*Layers_seq)函数构建网络
                 layers_node列表用于表示网络各层的节点
        """
        Layers_len = len(Layers)
        # 网络结构
        layers_seq = []

        # 开始构建
        for i in range(Layers_len):
            if isinstance(Layers[i], list):
                switch_dict = {
                    'Linear': lambda x: Network_common.Linear_func(x),
                    'LeakyReLU': lambda x: Network_common.LeakyReLU_func(x),
                    'ReLU': lambda x: Network_common.ReLU_func(x),
                    'Tanh': lambda x: Network_common.Tanh_func(x),
                    'Sigmoid': lambda x: Network_common.Sigmoid_func(x),
                    'Softplus': lambda x: Network_common.Softplus_func(x),
                    'Softmax': lambda x: Network_common.Softmax_func(x),
                    'Conv2d': lambda x: Network_common.Conv2d_func(x),
                    'BatchNorm2d': lambda x: Network_common.BatchNorm2d_func(x),
                    'ConvTranspose2d': lambda x: Network_common.ConvTranspose2d_func(x)
                }
                try:
                    layers_seq.append((switch_dict.get(Layers[i][0], None))(Layers[i]))
                except Exception as e:
                    print('============================='),
                    print('The Structure of the Network is error!————(' + str(Layers[i]) + ')')
                    print("Receive Error,Message:", e)
                    exit(1)

            else:
                print('=============================')
                print('The Structure of the Network is error!————(' + str(Layers[i]) + ')')
                exit(1)

        return layers_seq

    @staticmethod
    def Linear_func(layer_list):
        """
        Linear函数的调用返回，给Network_Structure函数使用 ['Linear',3,12]
        https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
        torch.nn.Linear(in_features, out_features, bias=True(, device=None, dtype=None))
        :param layer_list: 输入列表，列数分别代表Linear输入层和输出层、bias参数
        以['Linear', 3, 12, True]为例
        :return: 返回Linear的网络结构
        """
        len_list = len(layer_list)
        if len_list == 3:
            return nn.Linear(in_features=layer_list[1], out_features=layer_list[2])
        elif len_list == 4:
            return nn.Linear(in_features=layer_list[1], out_features=layer_list[2], bias=layer_list[3])
        else:
            print('The Length of The Linear Function List is wrong!')
            exit(1)

    @staticmethod
    def LeakyReLU_func(layer_list):
        """
        LeakyReLU函数的调用返回，给Network_Structure函数使用['LeakyReLU', 0.2]
        https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html#torch.nn.LeakyReLU
        torch.nn.LeakyReLU(negative_slope=0.01(, inplace=False))
        :param layer_list: 输入列表，列数分别代表LeakyReLU、negative_slope、inplace参数
        以['LeakyReLU', 0.2, False]为例
        :return: 返回LeakyReLU的网络结构
        """
        len_list = len(layer_list)
        if len_list == 2:
            return nn.LeakyReLU(negative_slope=layer_list[1])
        elif len_list == 3:
            return nn.LeakyReLU(negative_slope=layer_list[1], inplace=layer_list[2])
        else:
            print('The Length of The LeakyReLU Function List is wrong!')
            exit(1)

    @staticmethod
    def ReLU_func(layer_list):
        """
        ReLU函数的调用返回，给Network_Structure函数使用['ReLU']
        https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html#torch.nn.functional.relu
        torch.nn.functional.relu(input, inplace=False)
        :param layer_list: 输入列表，列数分别代表ReLU、inplace参数
        以['ReLU', False]为例
        :return: 返回ReLU的网络结构
        """
        len_list = len(layer_list)
        if len_list == 1:
            return nn.ReLU()
        elif len_list == 2:
            return nn.ReLU(inplace=layer_list[1])
        else:
            print('The Length of The ReLU Function List is wrong!')
            exit(1)

    @staticmethod
    def Tanh_func(layer_list):
        """
        Tanh函数的调用返回，给Network_Structure函数使用
        https://pytorch.org/docs/stable/generated/torch.nn.functional.tanh.html#torch.nn.functional.tanh
        torch.nn.functional.tanh(input)
        :param layer_list: 输入列表，列数分别代表Tanh参数
        以['Tanh']为例
        :return: 返回Tanh的网络结构
        """
        len_list = len(layer_list)
        if len_list == 1:
            return nn.Tanh()
        else:
            print('The Length of The Tanh Function List is wrong!')
            exit(1)

    @staticmethod
    def Sigmoid_func(layer_list):
        """
        Sigmoid函数的调用返回，给Network_Structure函数使用
        https://pytorch.org/docs/stable/generated/torch.nn.functional.sigmoid.html#torch.nn.functional.sigmoid
        torch.nn.functional.sigmoid(input)
        :param layer_list: 输入列表，列数分别代表Sigmoid参数
        以['Sigmoid']为例
        :return: 返回Sigmoid的网络结构
        """
        len_list = len(layer_list)
        if len_list == 1:
            return nn.Sigmoid()
        else:
            print('The Length of The Sigmoid Function List is wrong!')
            exit(1)

    @staticmethod
    def Softplus_func(layer_list):
        """
        Softplus函数的调用返回，给Network_Structure函数使用['Softplus']
        https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html#torch.nn.Softplus
        torch.nn.Softplus(beta=1, threshold=20)
        :param layer_list: 输入列表，列数分别代表Softplus参数
        以['Softplus', 1, 20]为例
        :return: 返回Softplus的网络结构
        """
        len_list = len(layer_list)
        if len_list == 1:
            return nn.Softplus()
        elif len_list == 2:
            return nn.Softplus(beta=layer_list[1])
        elif len_list == 3:
            return nn.Softplus(beta=layer_list[1], threshold=layer_list[2])
        else:
            print('The Length of The Softplus Function List is wrong!')
            exit(1)

    @staticmethod
    def Softmax_func(layer_list):
        """
        Softmax函数的调用返回，给Network_Structure函数使用['Softmax']
        https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html#torch.nn.Softmax
        torch.nn.Softmax(dim=None)
        :param layer_list: 输入列表，列数分别代表Softmax参数
        以['Softmax', 1]为例
        :return: 返回Softmax的网络结构
        """
        len_list = len(layer_list)
        if len_list == 1:
            return nn.Softmax()
        elif len_list == 2:
            return nn.Softmax(dim=layer_list[1])
        else:
            print('The Length of The Softmax Function List is wrong!')
            exit(1)

    @staticmethod
    def Conv2d_func(layer_list):
        """
        Conv2d函数的调用返回，给Network_Structure函数使用['Conv2d', 32, 64, (5,5)), 2, 1]
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
        torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
        groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        :param layer_list: 输入列表，列数分别代表Conv2d参数
        以['Conv2d', 32, 64, (5,5)), 2, 1, 1]为例
        :return: 返回Conv2d的网络结构
        """
        len_list = len(layer_list)
        if len_list == 4:
            return nn.Conv2d(in_channels=layer_list[1], out_channels=layer_list[2], kernel_size=layer_list[3])
        elif len_list == 5:
            return nn.Conv2d(in_channels=layer_list[1], out_channels=layer_list[2], kernel_size=layer_list[3],
                             stride=layer_list[4])
        elif len_list == 6:
            return nn.Conv2d(in_channels=layer_list[1], out_channels=layer_list[2], kernel_size=layer_list[3],
                             stride=layer_list[4], padding=layer_list[5])
        else:
            print('The Length of The Softmax Function List is wrong!')
            exit(1)

    @staticmethod
    def BatchNorm2d_func(layer_list):
        """
        BatchNorm2d函数的调用返回，给Network_Structure函数使用['BatchNorm2d', 64]
        https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d
        torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True,
        track_running_stats=True, device=None, dtype=None)
        :param layer_list: 输入列表，列数分别代表Softmax参数
        以['BatchNorm2d', 64]为例
        :return: 返回BatchNorm2d的网络结构
        """
        len_list = len(layer_list)
        if len_list == 2:
            return nn.BatchNorm2d(num_features=layer_list[1])
        else:
            print('The Length of The Softmax Function List is wrong!')
            exit(1)

    @staticmethod
    def ConvTranspose2d_func(layer_list):
        """
        ConvTranspose2d函数的调用返回，给Network_Structure函数使用['ConvTranspose2d', 32, 64, (5,5), 2, 1]
        https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html#torch.nn.ConvTranspose2d
        torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0,
        groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
        :param layer_list: 输入列表，列数分别代表Conv2d参数
        以['ConvTranspose2d', 32, 64, (5,5), 2, 1, 1]为例
        :return: 返回ConvTranspose2d的网络结构
        """
        len_list = len(layer_list)
        if len_list == 4:
            return nn.ConvTranspose2d(in_channels=layer_list[1], out_channels=layer_list[2], kernel_size=layer_list[3])
        elif len_list == 5:
            return nn.ConvTranspose2d(in_channels=layer_list[1], out_channels=layer_list[2], kernel_size=layer_list[3],
                                      stride=layer_list[4])
        elif len_list == 6:
            return nn.ConvTranspose2d(in_channels=layer_list[1], out_channels=layer_list[2], kernel_size=layer_list[3],
                                      stride=layer_list[4], padding=layer_list[5])
        elif len_list == 7:
            return nn.ConvTranspose2d(in_channels=layer_list[1], out_channels=layer_list[2], kernel_size=layer_list[3],
                                      stride=layer_list[4], padding=layer_list[5], output_padding=layer_list[6])
        else:
            print('The Length of The Softmax Function List is wrong!')
            exit(1)

    @staticmethod
    def optimizer_algorithm(model, optimizer_algorithm_name, **kwargs):
        """
        网络的优化算法
        :param model: 需要优化参数的模型
        :param optimizer_algorithm_name: 使用优化算法的方法名，例如'Adam'
        :param kwargs: 优化算法的参数，使用方式为字典形式，传入时为'Adam_optimizer_learning_rate=0.01'格式
        :return: 优化算法的定义
        """
        if optimizer_algorithm_name == 'Adam':
            # Adam
            # https://pytorch.org/docs/1.9.0/generated/torch.optim.Adam.html#torch.optim.Adam
            # lr（浮点数，可选）– 学习率（默认值：0.01）
            # weight_decay （float， 可选） – 权重衰减系数 （默认值：0）
            Adam_optimizer_learning_rate = kwargs.get("optimizer_learning_rate", 0.01)
            Adam_optimizer_weight_decay = kwargs.get("Adam_weight_decay", 0)
            optimizer_algorithm = optim.Adam(model.parameters(), lr=Adam_optimizer_learning_rate,
                                             weight_decay=Adam_optimizer_weight_decay)

        elif optimizer_algorithm_name == 'AdamW':
            # AdamW
            # https://pytorch.org/docs/1.9.0/generated/torch.optim.AdamW.html#torch.optim.AdamW
            # lr（浮点数，可选）– 学习率（默认值：0.01）
            # weight_decay （float， 可选） – 权重衰减系数 （默认值：0.01）
            AdamW_optimizer_learning_rate = kwargs.get("optimizer_learning_rate", 0.01)
            AdamW_optimizer_weight_decay = kwargs.get("AdamW_weight_decay", 0.01)
            optimizer_algorithm = optim.AdamW(model.parameters(), lr=AdamW_optimizer_learning_rate,
                                              weight_decay=AdamW_optimizer_weight_decay)

        elif optimizer_algorithm_name == 'SGD':
            # SGD
            # https://pytorch.org/docs/1.9.0/generated/torch.optim.SGD.html#torch.optim.SGD
            # lr (float) – 学习率
            # momentum (float, 可选) – 动量因子（默认：0）
            # weight_decay (float, 可选) – 权重衰减（L2惩罚）（默认：0）
            # dampening (float, 可选) – 动量的抑制因子（默认：0）
            # nesterov (bool, 可选) – 使用Nesterov动量（默认：False）
            SGD_optimizer_learning_rate = kwargs.get("optimizer_learning_rate", 0.01)
            SGD_optimizer_momentum = kwargs.get("SGD_momentum", 0)
            SGD_optimizer_weight_decay = kwargs.get("SGD_weight_decay", 0)
            SGD_optimizer_dampening = kwargs.get("SGD_dampening", 0)
            optimizer_algorithm = optim.SGD(model.parameters(), lr=SGD_optimizer_learning_rate,
                                            momentum=SGD_optimizer_momentum, dampening=SGD_optimizer_dampening,
                                            weight_decay=SGD_optimizer_weight_decay)

        elif optimizer_algorithm_name == 'Adagrad':
            # Adagrad
            # https://pytorch.org/docs/1.9.0/generated/torch.optim.Adagrad.html#torch.optim.Adagrad
            # lr (float, 可选) – 学习率（默认: 1e-2）
            # lr_decay (float, 可选) – 学习率衰减（默认: 0）
            # weight_decay (float, 可选) – 权重衰减（L2惩罚）（默认: 0）
            # initial_accumulator_value - 累加器的起始值，必须为正。
            Adagrad_optimizer_learning_rate = kwargs.get("optimizer_learning_rate", 0.01)
            Adagrad_optimizer_learning_rate_decay = kwargs.get("Adagrad_learning_rate_decay", 0)
            Adagrad_optimizer_weight_decay = kwargs.get("Adagrad_weight_decay", 0)
            optimizer_algorithm = optim.Adagrad(model.parameters(), lr=Adagrad_optimizer_learning_rate,
                                                lr_decay=Adagrad_optimizer_learning_rate_decay,
                                                weight_decay=Adagrad_optimizer_weight_decay)

        elif optimizer_algorithm_name == 'RMSProp':
            # RMSProp
            # https://pytorch.org/docs/1.9.0/generated/torch.optim.RMSprop.html#torch.optim.RMSprop
            # lr (float, 可选) – 学习率（默认: 1e-2）
            # alpha – 平滑常数α（默认: 0.99）
            # weight_decay (float, 可选) – 权重衰减（L2惩罚）（默认: 0）
            # momentum (float, 可选) – 动量因子（默认：0）
            # centered – centered和RMSProp并无直接关系，是为了让结果更平稳
            RMSProp_optimizer_learning_rate = kwargs.get("optimizer_learning_rate", 0.01)
            RMSProp_optimizer_alpha = kwargs.get("RMSProp_alpha", 0.99)
            RMSProp_optimizer_weight_decay = kwargs.get("RMSProp_weight_decay", 0)
            RMSProp_optimizer_momentum = kwargs.get("RMSProp_momentum", 0)
            RMSProp_centered = kwargs.get("RMSProp_centered", False)
            optimizer_algorithm = optim.RMSprop(model.parameters(), lr=RMSProp_optimizer_learning_rate,
                                                alpha=RMSProp_optimizer_alpha,
                                                weight_decay=RMSProp_optimizer_weight_decay,
                                                momentum=RMSProp_optimizer_momentum, centered=RMSProp_centered)

        elif optimizer_algorithm_name == 'LBFGS':
            # LBFGS
            # https://pytorch.org/docs/1.9.0/generated/torch.optim.LBFGS.html#torch.optim.LBFGS
            # lr （float） - 学习率（默认值：1）
            # max_iter （int） – 每个优化步骤的最大迭代次数（默认值：20）
            # max_eval （int） - 每个优化步骤的最大函数计算数（默认值：max_iter * 1.25）。
            LBFGS_optimizer_learning_rate = kwargs.get("optimizer_learning_rate", 1)
            LBFGS_optimizer_max_iter = kwargs.get("max_iter", 10)
            LBFGS_optimizer_max_eval = kwargs.get("max_eval", LBFGS_optimizer_max_iter * 1.25)
            optimizer_algorithm = optim.LBFGS(model.parameters(), lr=LBFGS_optimizer_learning_rate,
                                              max_iter=LBFGS_optimizer_max_iter,
                                              max_eval=LBFGS_optimizer_max_eval)

        else:
            print('=============================')
            print(r'The Optimizer Algorithm doesn\'t exist! Use Adam Algorithm! Learning Rate is 0.01!')
            Adam_optimizer_learning_rate = 0.01
            optimizer_algorithm = optim.Adam(model.parameters(), lr=Adam_optimizer_learning_rate)

        return optimizer_algorithm

    @staticmethod
    def optimizer_scheduler(optimizer_input, optimizer_lr_scheduler_name, **kwargs):
        """
        网络优化器的学习率的衰减调度器
        https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html
        参考：https://blog.csdn.net/qq_40206371/article/details/119910592
        其中CosineAnnealingWarmRestarts未写
        :param optimizer_input: 网络的优化器
        :param optimizer_lr_scheduler_name: 网络的优化器的衰减调度器名称
        :param kwargs: 衰减相关的参数，使用方式为字典形式，传入时为'Adam_optimizer_learning_rate=0.01'格式
        :return: 优化算法的定义
        """
        if optimizer_lr_scheduler_name == 'LambdaLR':
            # LambdaLR
            # optimizer （Optimizer）：要更改学习率的优化器
            # lr_lambda（function or list）：根据epoch计算λ的函数，这里为 1/(epoch+1)
            # last_epoch （int）：最后一个epoch的index，如果是训练了很多个epoch后中断了，继续训练，
            # 这个值就等于加载的模型的epoch。默认为-1表示从头开始训练，即从epoch=1开始。
            LambdaLR_last_epoch = kwargs.get("LambdaLR_last_epoch", -1)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer_input,
                                                    lr_lambda=lambda epoch: 1/(epoch+1),
                                                    last_epoch=LambdaLR_last_epoch)

        elif optimizer_lr_scheduler_name == 'StepLR':
            # StepLR
            # optimizer （Optimizer）：要更改学习率的优化器
            # step_size（int）：每训练step_size个epoch，更新一次参数
            # gamma（float）：更新lr的乘法因子
            # last_epoch （int）：最后一个epoch的index，如果是训练了很多个epoch后中断了，继续训练，
            # 这个值就等于加载的模型的epoch。默认为-1表示从头开始训练，即从epoch=1开始
            StepLR_step_size = kwargs.get("StepLR_step_size", 3)
            StepLR_gamma = kwargs.get("StepLR_gamma", 0.1)
            StepLR_last_epoch = kwargs.get("StepLR_last_epoch", -1)
            scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer_input, step_size=StepLR_step_size,
                                                  gamma=StepLR_gamma, last_epoch=StepLR_last_epoch)

        elif optimizer_lr_scheduler_name == 'MultiStepLR':
            # MultiStepLR
            # optimizer （Optimizer）：要更改学习率的优化器
            # milestones（list）：递增的list，存放要更新lr的epoch，在milestones里面的这几个点时相继乘以gamma系数
            # gamma（float）：更新lr的乘法因子
            # last_epoch （int）：最后一个epoch的index，如果是训练了很多个epoch后中断了，继续训练，
            # 这个值就等于加载的模型的epoch。默认为-1表示从头开始训练，即从epoch=1开始
            MultiStepLR_milestones = kwargs.get("MultiStepLR_milestones", [3, 9])
            MultiStepLR_gamma = kwargs.get("MultiStepLR_gamma", 0.1)
            MultiStepLR_last_epoch = kwargs.get("MultiStepLR_last_epoch", -1)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer_input,
                                                       milestones=MultiStepLR_milestones,
                                                       gamma=MultiStepLR_gamma,
                                                       last_epoch=MultiStepLR_last_epoch)

        elif optimizer_lr_scheduler_name == 'ExponentialLR':
            # ExponentialLR
            # optimizer （Optimizer）：要更改学习率的优化器
            # gamma（float）：更新lr的乘法因子
            # last_epoch （int）：最后一个epoch的index，如果是训练了很多个epoch后中断了，继续训练，
            # 这个值就等于加载的模型的epoch。默认为-1表示从头开始训练，即从epoch=1开始
            ExponentialLR_gamma = kwargs.get("ExponentialLR_gamma", 0.1)
            ExponentialLR_last_epoch = kwargs.get("ExponentialLR_last_epoch", -1)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer_input,
                                                         gamma=ExponentialLR_gamma,
                                                         last_epoch=ExponentialLR_last_epoch)

        elif optimizer_lr_scheduler_name == 'CosineAnnealingLR':
            # CosineAnnealingLR   采用周期变化的策略调整学习率，能够使模型跳出在训练过程中遇到的局部最低点和鞍点
            # optimizer （Optimizer）：要更改学习率的优化器
            # T_max（int）：lr的变化是周期性的，T_max是周期的1/2
            # eta_min（float）：lr的最小值，默认为0
            # last_epoch （int）：最后一个epoch的index，如果是训练了很多个epoch后中断了，继续训练，
            # 这个值就等于加载的模型的epoch。默认为-1表示从头开始训练，即从epoch=1开始
            CosineAnnealingLR_T_max = kwargs.get("CosineAnnealingLR_T_max", 10)
            CosineAnnealingLR_eta_min = kwargs.get("CosineAnnealingLR_eta_min", 0)
            CosineAnnealingLR_last_epoch = kwargs.get("CosineAnnealingLR_last_epoch", -1)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_input,
                                                             T_max=CosineAnnealingLR_T_max,
                                                             eta_min=CosineAnnealingLR_eta_min,
                                                             last_epoch=CosineAnnealingLR_last_epoch)

        elif optimizer_lr_scheduler_name == 'MultiplicativeLR':
            # MultiplicativeLR   采用周期变化的策略调整学习率，能够使模型跳出在训练过程中遇到的局部最低点和鞍点
            # optimizer （Optimizer）：要更改学习率的优化器
            # lr_lambda（function or list）：根据epoch计算λ的函数，这里为 1/(epoch+1)
            # last_epoch （int）：最后一个epoch的index，如果是训练了很多个epoch后中断了，继续训练，
            # 这个值就等于加载的模型的epoch。默认为-1表示从头开始训练，即从epoch=1开始
            MultiplicativeLR_lr_lambda = kwargs.get("MultiplicativeLR_lambda", lambda epoch: 1/(epoch+1))
            MultiplicativeLR_last_epoch = kwargs.get("MultiplicativeLR_last_epoch", -1)
            scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer=optimizer_input,
                                                            lr_lambda=MultiplicativeLR_lr_lambda,
                                                            last_epoch=MultiplicativeLR_last_epoch)

        elif optimizer_lr_scheduler_name == 'CyclicLR':
            # CyclicLR   每隔一段时间重启学习率，这样在单位时间内能收敛到多个局部最小值，可以得到很多个模型做集成
            # optimizer （Optimizer）：要更改学习率的优化器
            # base_lr	循环中学习率的下边界
            # max_lr	循环中学习率的上边界
            # step_size_up	学习率上升的步数
            # step_size_down	学习率下降的步数
            # mode	{triangular, triangular2, exp_range}中的一个。默认: 'triangular'
            # gamma (float)	在mode='exp_range'时，gamma**(cycle iterations)， 默认：1.0
            CyclicLR_lr_base_lr = kwargs.get("CyclicLR_base_lr", 0.1)
            CyclicLR_lr_max_lr = kwargs.get("CyclicLR_base_lr", 10)
            CyclicLR_lr_step_size_up = kwargs.get("CyclicLR_step_size_up", 10)
            CyclicLR_lr_step_size_down = kwargs.get("CyclicLR_step_size_down", 5)
            CyclicLR_mode = kwargs.get("CyclicLR_mode", 'triangular')
            CyclicLR_gamma = kwargs.get("CyclicLR_gamma", 1)
            if CyclicLR_mode == 'triangular' or CyclicLR_mode == 'triangular2':
                scheduler = optim.lr_scheduler.CyclicLR(optimizer=optimizer_input,
                                                        base_lr=CyclicLR_lr_base_lr,
                                                        max_lr=CyclicLR_lr_max_lr,
                                                        step_size_up=CyclicLR_lr_step_size_up,
                                                        step_size_down=CyclicLR_lr_step_size_down,
                                                        mode=CyclicLR_mode)

            elif CyclicLR_mode == 'exp_range':
                scheduler = optim.lr_scheduler.CyclicLR(optimizer=optimizer_input,
                                                        base_lr=CyclicLR_lr_base_lr,
                                                        max_lr=CyclicLR_lr_max_lr,
                                                        step_size_up=CyclicLR_lr_step_size_up,
                                                        step_size_down=CyclicLR_lr_step_size_down,
                                                        mode=CyclicLR_mode, gamma=CyclicLR_gamma)
            else:
                print('=============================')
                print(r'The mode of CyclicLR Scheduler doesn\'t exist! Use triangular! ')
                scheduler = optim.lr_scheduler.CyclicLR(optimizer=optimizer_input,
                                                        base_lr=CyclicLR_lr_base_lr,
                                                        max_lr=CyclicLR_lr_max_lr,
                                                        step_size_up=CyclicLR_lr_step_size_up,
                                                        step_size_down=CyclicLR_lr_step_size_down,
                                                        mode=CyclicLR_mode)

        else:
            print('=============================')
            print(r'The Scheduler of Optimizer doesn\'t exist! Use ExponentialLR! Gamma is 0.1!')
            ExponentialLR_gamma = 0.1
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer_input,
                                                         gamma=ExponentialLR_gamma)

        return scheduler

    @staticmethod
    def loss_function(criterion_name, device, **kwargs):
        """
        网络的损失函数
        :param criterion_name: 损失函数名字
        :param device: 使用cpu或者gpu
        :param kwargs: 其他参数
        :return: 损失函数的定义
        """
        if criterion_name == 'MSE':
            # MSE
            # https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss
            # 若reduction = 'none'、'sum'、'mean'，则返回向量形式的loss、返回loss之和、返回loss的平均值
            reduction_para = kwargs.get("reduction", 'mean')
            criterion = nn.MSELoss(reduction=reduction_para).to(device)
        elif criterion_name == 'L1Loss':
            # L1Loss
            # https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html#torch.nn.L1Loss
            # 计算 output 和 target 之差的绝对值
            # 若reduction = 'sum'、'mean'，则返回loss之和、返回loss的平均值
            reduction_para = kwargs.get("reduction", 'mean')
            criterion = nn.L1Loss(reduction=reduction_para).to(device)
        elif criterion_name == 'BCELoss':
            # BCELoss
            # https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss
            # 二元交叉熵，二分类
            # 若reduction = 'none'、'sum'、'mean'，则返回向量形式的loss、返回loss之和、返回loss的平均值
            reduction_para = kwargs.get("reduction", 'none')
            criterion = nn.BCELoss(reduction=reduction_para).to(device)
        elif criterion_name == 'CrossEntropyLoss':
            # CrossEntropyLoss
            # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
            # 多元交叉熵，多分类
            # 若reduction = 'none'、'sum'、'mean'，则返回向量形式的loss、返回loss之和、返回loss的平均值
            reduction_para = kwargs.get("reduction", 'none')
            criterion = nn.CrossEntropyLoss(reduction=reduction_para).to(device)
        elif criterion_name == 'KLDivLoss':
            # KLDivLoss
            # https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html#torch.nn.KLDivLoss
            # KL散度损失
            # size_average （bool， 可选） – 已弃用 （请参阅 reduction ）。
            # 默认情况下，损失是批处理中每个损失元素的平均值。请注意，对于某些损失，每个样品有多个元素。
            # 如果该字段 size_average 设置为 False，则将对每个小批量的损失求和。当 False 时 reduce 忽略。默认值：True
            # reduce （bool， 可选） – 已弃用（请参阅 reduction ）。
            # 默认情况下，损失值将根据每个 size_average 小批量的观测值进行平均或求和。如果 reduce 为 False，则返回每个批处理元素的损失并忽略 size_average 。默认值：True
            # reduction = “mean” 不返回真正的 KL 散度值，请使用 reduction = “batchmean”，它符合数学定义
            # reduction （str， 可选） – 指定要应用于输出的缩减。默认值：“mean”
            # log_target （bool， 可选） - 指定目标是否为日志空间。默认值：False
            size_average = kwargs.get("size_average", True)
            reduce = kwargs.get("reduce", True)
            reduction_para = kwargs.get("reduction", 'mean')
            log_target_para = kwargs.get("log_target", False)
            criterion = nn.KLDivLoss(size_average=size_average, reduce=reduce,
                                     reduction=reduction_para,
                                     log_target=log_target_para).to(device)
        else:
            print('=============================')
            print('The Criterion Function doesn\'t exist! Use MSE!')
            criterion = nn.MSELoss().to(device)

        return criterion
