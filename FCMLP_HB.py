import numpy as np
import random
import torch
import math
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, Tanh, BatchNorm1d as BN, ReLU6 as ReLU6, Softmax as Soft
from BGNN_global_value import *

########################################################################################################################基于MLP创建conventional DNN for HBF##################################################################
device_num = device_num_get()

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]).double().to(device=device_num), BN(channels[i], eps=1e-05).double().to(device=device_num), ReLU().double().to(device=device_num))
        for i in range(1, len(channels))
    ])

class HB_FC_MLP(nn.Module):
    def __init__(self, num_ant_per_chain, num_user, num_rf, num_ant, P_pow, channel):                                            #__init__()写成__int__()会报错多余参数
        super(HB_FC_MLP, self).__init__()
        self.num_ant_per_chain = num_ant_per_chain
        self.P_pow = P_pow
        self.P_normalized = P_pow / num_ant_per_chain
        self.num_rf = num_rf
        self.MLP_channel = channel
        self.mlp = MLP(channel)
        self.mlp = Seq(*[self.mlp, Seq(Lin(channel[len(channel) - 1], num_ant * 2 + num_user).double().to(device =device_num), Tanh().double().to(device =device_num))])

    def forward(self, H):

        batch_size, num_user, num_ant = H.shape

        H_real = H.real
        H_real_1 = torch.reshape(H_real, (batch_size, -1))
        H_imag = H.imag
        H_imag_1 = torch.reshape(H_imag, (batch_size, -1))
        H_input = torch.cat([H_real_1, H_imag_1], dim=1)
        P_input = (self.P_pow/(num_user*self.num_ant_per_chain)) * torch.ones([batch_size, num_user]).to(device =device_num)
        HB_input = torch.cat([H_input, P_input], dim=1)

        HB_out = self.mlp(HB_input)
        F, V = HB_generate(HB_out, H, self.num_ant_per_chain, num_user, self.num_rf, num_ant, self.P_normalized)
        Rate = Sum_rate(H, F, V)

        return F, V, Rate

def HB_generate(HB_output, H, num_ant_per_chain, num_user, num_rf, num_ant, P_normalized):

    batch_size = HB_output.shape[0]
    F_output = HB_output[:, np.arange(0, num_ant)] + 1j * HB_output[:, np.arange(num_ant, num_ant*2)]
    Sigmoid_1 = Sigmoid().double().to(device=device_num)
    P_output = Sigmoid_1(HB_output[:, np.arange(num_ant * 2, num_ant * 2 + num_user)])

    F_output_1 = F_output / torch.abs(F_output)
    F = torch.zeros([batch_size, num_ant, num_rf], dtype=torch.complex128).to(device = device_num)
    for nr in np.arange(0, num_rf):
        F[:, np.arange(num_ant_per_chain * nr, num_ant_per_chain * (nr + 1)), nr] = F_output_1[:, np.arange(num_ant_per_chain * nr, num_ant_per_chain * (nr + 1))]

    P_output_1 = torch.sum(P_output, dim=1)
    P_output_2 = torch.zeros((batch_size, num_user), dtype=torch.float64).to(device = device_num)
    for nb in np.arange(0, batch_size):
        P_output_2[nb, :] = (P_normalized / P_output_1[nb]) * P_output[nb, :]

    H_eff = H @ F
    H_1 = torch.conj(torch.transpose(H_eff, 1, 2))
    H_inv = torch.inverse(torch.matmul(H_eff, H_1) + (num_user / P_normalized) * torch.eye(num_user, device=device_num))  ######MMSE/RZF
    # H_inv = torch.inverse(torch.matmul(H_eff, H_1))           ###############ZF
    W = torch.matmul(H_1, H_inv)
    V = torch.zeros([batch_size, num_rf, num_user], dtype=torch.complex128, device=device_num)
    for nb in np.arange(0, batch_size):
        p_vector = torch.sqrt(P_output_2[nb, :]).squeeze()
        Power_allocation_matrix_1 = torch.diag(p_vector)
        Power_allocation_matrix = Power_allocation_matrix_1.type(torch.complex128)
        W_temp = W[nb, :, :]
        W_1 = W_temp / torch.norm(W_temp, 'fro', dim=0)
        V[nb, :, :] = W_1 @ Power_allocation_matrix

    return F, V


def Sum_rate(H, F, V):

    batch_size = H.shape[0]
    num_user = H.shape[1]
    Rate = torch.zeros([batch_size, 1]).to(device =device_num)
    for nb in np.arange(0, batch_size):
        temp = torch.pow(torch.abs(H[nb, :, :] @ F[nb, :, :] @ V[nb, :, :]), 2)
        signal_power = torch.diag(temp)
        signal_noise_power = torch.sum(temp, 1)
        for nu in np.arange(0, num_user):
            Rate[nb] = Rate[nb] + torch.log2(1 + (signal_power[nu] / (signal_noise_power[nu] - signal_power[nu] + 1)))

    return Rate