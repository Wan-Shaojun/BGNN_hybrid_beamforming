import numpy as np
import random
import torch
import math
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, Tanh, BatchNorm1d as BN, ReLU6 as ReLU6, Softmax as Soft
from BGNN_global_value import *


########################################################################################################################基于CNN创建conventional DNN for fully-digital beamforming##########################################################################################
device_num = device_num_get()

# def MLP(channels, batch_norm=True):
#     return Seq(*[
#         Seq(Lin(channels[i - 1], channels[i]).double().to(device=device_num), BN(channels[i], eps=1e-05).double().to(device=device_num), ReLU().double().to(device=device_num))
#         for i in range(1, len(channels))
#     ])

def CNN(cnn_channel):           ####此处cnn不改变数据的H和W
    return Seq(*[
        Seq(nn.Conv2d(in_channels=cnn_channel[i - 1], out_channels=cnn_channel[i], kernel_size=3, padding=1, stride=1).double().to(device=device_num),
            nn.BatchNorm2d(cnn_channel[i], eps=1e-05).double().to(device=device_num), ReLU().double().to(device=device_num))
        for i in range(1, len(cnn_channel))
    ])

class DB_FC_CNN(nn.Module):
    def __init__(self, num_user, num_ant, P_pow, cnn_channel):                                            #__init__()写成__int__()会报错多余参数
        super(DB_FC_CNN, self).__init__()
        self.P_pow = P_pow
        # self.MLP_channel = channel
        # self.mlp = MLP(channel)
        # self.mlp = Seq(*[self.mlp, Seq(Lin(channel[len(channel) - 1], num_ant * 2 + num_user).double().to(device =device_num), Tanh().double().to(device =device_num))])
        self.cnn = CNN(cnn_channel)
        self.cnn = Seq(*[self.cnn, Seq(nn.Conv2d(in_channels=cnn_channel[len(cnn_channel) - 1], out_channels=1, kernel_size=3, padding=1, stride=1).double().to(device=device_num),
                                       nn.BatchNorm2d(1, eps=1e-05).double().to(device=device_num), ReLU().double().to(device=device_num))])
        self.mlp_p = Seq(Lin(2 * num_user * num_ant, num_user).double().to(device =device_num), Sigmoid().double().to(device =device_num))
        self.mlp_q = Seq(Lin(2 * num_user * num_ant, num_user).double().to(device =device_num), Sigmoid().double().to(device =device_num))

    def forward(self, H):

        batch_size, num_user, num_ant = H.shape

        H_real = H.real
        H_real_1 = torch.reshape(H_real, (batch_size, -1))
        H_imag = H.imag
        H_imag_1 = torch.reshape(H_imag, (batch_size, -1))
        H_input = torch.cat([H_real_1, H_imag_1], dim=1)
        CNN_input_temp = H_input.unsqueeze(dim = 1)
        CNN_input = CNN_input_temp.unsqueeze(dim = 3)

        CNN_output = self.cnn(CNN_input)
        output = CNN_output.squeeze()
        P_output = self.mlp_p(output)
        Q_output = self.mlp_q(output)
        P_1 = P_output.unsqueeze(dim=2)
        Q_1 = Q_output.unsqueeze(dim=2)
        D_temp = torch.cat([P_1, Q_1], dim=2)
        D_sum = torch.sum(D_temp, dim=1)
        D_out = torch.zeros([batch_size, num_user, 2], dtype=torch.float64).to(device =device_num)
        for nb in np.arange(0, batch_size):
            D_out[nb, :, :] = self.P_pow * (D_temp[nb, :, :] / D_sum[nb, :])

        V = generalDBF_generate(D_out, H)
        Rate = FD_Sum_rate(H, V)

        return V, Rate

def generalDBF_generate(D, H):                                                                                         #to generate digital beamforming matrix from D

    batch_size, num_user, num_ant = H.shape

    V_1 = torch.empty([batch_size, num_ant, num_user], dtype=torch.complex128).to(device =device_num)
    for nb in np.arange(0, batch_size):
        temp_H = H[nb, :, :]
        Noise_and_Interference = torch.eye(num_ant, dtype=torch.complex128).to(device =device_num)
        for nu in np.arange(0, num_user):
            a = temp_H[nu, :].unsqueeze(dim=0)
            d = torch.conj(a.t()) @ a
            Noise_and_Interference = Noise_and_Interference + D[nb, nu, 1] * d                 # \mathbf{I}_{N_t} + \sum_{l=1}^{K} q_{l} \hat{\mathbf{h}}_{l}^{H} \hat{\mathbf{h}}_{l}
        for nu in np.arange(0, num_user):
            a_1 = temp_H[nu, :].unsqueeze(dim=0)
            c_1 = torch.conj(a_1.t())
            upper = Noise_and_Interference.inverse() @ c_1
            down = torch.norm(upper, 'fro')
            v_temp = (torch.sqrt(D[nb, nu, 0]) / down) * upper
            V_1[nb, :, nu] = v_temp.squeeze()

    return V_1

def FD_Sum_rate(H, V):

    batch_size = H.shape[0]
    num_user = H.shape[1]
    Rate = torch.zeros([batch_size, 1], device=device_num)
    for nb in np.arange(0, batch_size):
        temp = torch.pow(torch.abs(H[nb, :, :] @ V[nb, :, :]), 2)
        signal_power = torch.diag(temp)
        signal_noise_power = torch.sum(temp, 1)
        for nu in np.arange(0, num_user):
            Rate[nb] = Rate[nb] + torch.log2(1 + (signal_power[nu] / (signal_noise_power[nu] - signal_power[nu] + 1)))

    return Rate