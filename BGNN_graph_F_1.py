import numpy as np
import random
import torch
import math
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, Tanh, BatchNorm1d as BN, ReLU6 as ReLU6, Softmax as Soft
from BGNN_global_value import *

########################################################################################################################digital precoding采取optimal solution structure############################################
device_num = device_num_get()

def MLP(channels, batch_norm=True):
    return Seq(*[
        # Seq(Lin(channels[i - 1], channels[i]).double().to(device=device_num), BN(channels[i], eps=1e-05).double().to(device=device_num), ReLU().double().to(device=device_num))
        Seq(Lin(channels[i - 1], channels[i]).double().to(device =device_num), ReLU().double().to(device =device_num))
        for i in range(1, len(channels))
    ])

class User_message_generate(nn.Module):                                                                                 # 2*num_ant_per_chain + num_M + 2 -> M
    def __init__(self, num_ant_per_chain, user_channel, num_M, operation = 'mean'):
        super(User_message_generate, self).__init__()
        self.num_ant_per_chain = num_ant_per_chain
        self.num_M = num_M
        self.user_channel = user_channel
        self.mlp = MLP(user_channel)
        self.mlp = Seq(*[self.mlp, Seq(Lin(user_channel[len(user_channel) - 1], num_M).double().to(device =device_num), Tanh().double().to(device =device_num))])
        # self.mlp = Seq(*[self.mlp, Seq(Lin(user_channel[len(user_channel) - 1], num_M).double().to(device =device_num))])

    def forward(self, D, B, H):
        batch_size, num_user, num_ant_2 = H.size()                                                                      #num_ant_2 = num_ant*2
        num_ant = int(num_ant_2 / 2)
        num_rf = B.shape[1]
        B_norm = torch.sum(B, dim=1)
        output = torch.zeros([batch_size, num_user, num_rf, self.num_M]).to(device =device_num)

        #parallel trainning
        data_input = torch.zeros([batch_size * num_user * num_rf, self.user_channel[0]], dtype=torch.float64).to(device =device_num)
        for nu in np.arange(0, num_user):
            d_k = D[:, nu, :]
            b_k = B_norm[:, nu, :]
            for nr in np.arange(0, num_rf):
                h_kr = H[:, nu, (self.num_ant_per_chain * nr) : (self.num_ant_per_chain * (nr+1))]
                h_ki = H[:, nu, (num_ant + self.num_ant_per_chain * nr) : (num_ant + self.num_ant_per_chain * (nr+1))]
                data_input[batch_size*(nu * num_rf + nr) : batch_size*(nu * num_rf + nr + 1), :] = torch.cat([d_k, b_k, h_kr, h_ki], dim=1)
        data_output = self.mlp(data_input)
        for nu in np.arange(0, num_user):
            for nr in np.arange(0, num_rf):
                output[:, nu, nr, :] = data_output[batch_size*(nu * num_rf + nr) : batch_size*(nu * num_rf + nr + 1), :]

        return output



class Antenna_message_generate(nn.Module):                                                                              #4*num_ant_per_chain + num_M -> num_M
    def __init__(self, num_ant_per_chain, antenna_channel, num_M, operation = 'mean'):
        super(Antenna_message_generate, self).__init__()
        self.num_ant_per_chain = num_ant_per_chain
        self.num_M = num_M
        self.antenna_channel = antenna_channel
        self.mlp = MLP(antenna_channel)
        self.mlp = Seq(*[self.mlp, Seq(Lin(antenna_channel[len(antenna_channel) - 1], num_M).double().to(device =device_num), Tanh().double().to(device =device_num))])

    def forward(self, C, F, H):
        batch_size, num_user, num_ant_2 = H.size()
        num_ant = int(num_ant_2/2)
        num_rf = F.shape[2]
        output = torch.zeros([batch_size, num_rf, num_user, self.num_M]).to(device =device_num)
        C_norm = torch.sum(C, dim=1)

        data_input = torch.zeros([batch_size * num_rf * num_user, self.antenna_channel[0]], dtype=torch.float64).to(device =device_num)
        for nr in np.arange(0, num_rf):
            c_i = C_norm[:, nr, :]
            F_i = F[:, :, nr]
            for nu in np.arange(0, num_user):
                h_kr = H[:, nu, (self.num_ant_per_chain * nr) : (self.num_ant_per_chain * (nr+1))]
                h_ki = H[:, nu, (num_ant + self.num_ant_per_chain * nr) : (num_ant + self.num_ant_per_chain * (nr+1))]
                data_input[batch_size*(nr * num_user + nu) : batch_size*(nr * num_user + nu + 1), :] = torch.cat([F_i, c_i, h_kr, h_ki], dim=1)
        data_output = self.mlp(data_input)
        for nr in np.arange(0, num_rf):
            for nu in np.arange(0, num_user):
                output[:, nr, nu, :] = data_output[batch_size*(nr * num_user + nu) : batch_size*(nr * num_user + nu + 1), :]
        return output

class Analogue_update(nn.Module):                                                                                       #2*num_M -> 2*num_ant_per_chain
    def __init__(self, num_ant_per_chain, analogue_channel, num_M, operation = 'mean'):
        super(Analogue_update, self).__init__()
        self.num_ant_per_chain = num_ant_per_chain
        self.num_M = num_M
        self.analogue_channel = analogue_channel
        self.mlp = MLP(analogue_channel)
        self.mlp = Seq(*[self.mlp, Seq(Lin(analogue_channel[len(analogue_channel) - 1], 2 * num_ant_per_chain).double().to(device =device_num), Tanh().double().to(device =device_num))])
        self.weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.weight_1.data.fill_(1.0)

    def forward(self, C, F, F_0):
        batch_size = C.shape[0]
        num_rf = C.shape[2]
        F_temp_1 = torch.zeros([batch_size, 2 * self.num_ant_per_chain, num_rf], dtype=torch.float64).to(device =device_num)
        C_norm = torch.sum(C, dim=1)

        data_input = torch.zeros([batch_size * num_rf, self.analogue_channel[0]], dtype=torch.float64).to(device =device_num)
        for nr in np.arange(0, num_rf):
            data_input[batch_size*nr : batch_size*(nr+1), :] = torch.cat([C_norm[:, nr, :], F[:, :, nr]], dim=1)
            # data_input[batch_size * nr: batch_size * (nr + 1), :] = W_i[:, nr, :]
        data_output = self.mlp(data_input)
        for nr in np.arange(0, num_rf):
            F_temp_1[:, :, nr] = data_output[batch_size*nr : batch_size*(nr+1), :]
        # normalization
        F_temp = F_temp_1 + self.weight_1 * F_0
        # F_temp = F_temp_1
        F_2 = torch.abs(F_temp[:, np.arange(0, self.num_ant_per_chain), :] + 1j * F_temp[:, np.arange(0, self.num_ant_per_chain) + self.num_ant_per_chain, :])
        F_3 = torch.cat([F_2, F_2], dim=1)
        F_4 = F_temp / F_3
        return F_4

class Digital_update(nn.Module):                                                                                        #2*num_M -> 2
    def __init__(self, digital_channel, num_M, operation = 'mean'):
        super(Digital_update, self).__init__()
        self.num_M = num_M
        self.digital_channel = digital_channel
        self.mlp = MLP(digital_channel)
        self.mlp = Seq(*[self.mlp, Seq(Lin(digital_channel[len(digital_channel) - 1], 2).double().to(device =device_num), Sigmoid().double().to(device =device_num))])
        # self.weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.weight_1.data.fill_(100.0)

    def forward(self, D, B, P_pow_normalized, D_0):
        batch_size = B.shape[0]
        num_user = B.shape[2]
        D_1 = torch.zeros([batch_size, num_user, 2], dtype=torch.float64).to(device =device_num)
        D_out = torch.zeros([batch_size, num_user, 2], dtype=torch.float64).to(device =device_num)
        B_norm = torch.sum(B, dim=1)
        data_input = torch.zeros([batch_size*num_user, self.digital_channel[0]], dtype=torch.float64).to(device =device_num)
        for nu in np.arange(0, num_user):
            data_input[batch_size*nu : batch_size*(nu+1), :] = torch.cat([D[:, nu, :], B_norm[:, nu, :]], dim=1)
        dat_output = self.mlp(data_input)
        for nu in np.arange(0, num_user):
            D_1[:, nu, :] = dat_output[batch_size*nu : batch_size*(nu+1), :]
        D_temp = D_1
        D_sum = torch.sum(D_temp, dim=1)
        for nb in np.arange(0, batch_size):
            D_out[nb, :, :] = P_pow_normalized * (D_temp[nb, :, :] / D_sum[nb, :])
        return D_out

def W_generate(C):                                                                                                      #to generate W and M from C and B, respectively

    batch_size, size_1, size_2, num_M = C.size()
    # C_array = C.cpu().detach().numpy()
    W = torch.zeros([batch_size, size_1, size_2, 2*num_M]).to(device =device_num)
    for nu in np.arange(0, size_1):
        for nr in np.arange(0, size_2):
            temp = torch.sum(C[:, nu, :, :], dim=1) - C[:, nu, nr, :]
            temp_1 = temp
            W[:,nu,nr,:] = torch.cat([C[:,nu,nr,:], temp_1], dim=1)

    return W

def generalBF_generate(D, H, F_1, num_ant_per_chain):                                                                                         #to generate digital beamforming matrix from D

    batch_size, num_user, num_ant = H.shape
    num_rf = F_1.shape[2]
    F = torch.zeros([batch_size, num_ant, num_rf], dtype=torch.complex128).to(device =device_num)                                    #将DNN的输出F_1转换成正常的block diagonal的F
    for nr in np.arange(0, num_rf):
        F[:, (num_ant_per_chain * nr) : (num_ant_per_chain * (nr+1)), nr] = F_1[:, :, nr]
    V_1 = torch.empty([batch_size, num_rf, num_user], dtype=torch.complex128).to(device =device_num)
    for nb in np.arange(0, batch_size):
        temp_H = H[nb, :, :] @ F[nb, :, :]
        Noise_and_Interference = torch.eye(num_rf, dtype=torch.complex128).to(device =device_num)
        for nu in np.arange(0, num_user):
            a = temp_H[nu, :].unsqueeze(dim=0)
            d = torch.conj(a.t()) @ a
            Noise_and_Interference = Noise_and_Interference + D[nb, nu, 1] * d                 # \mathbf{I}_{N_t} + \sum_{l=1}^{K} q_{l} \hat{\mathbf{h}}_{l}^{H} \hat{\mathbf{h}}_{l}
        for nu in np.arange(0, num_user):
            a_1 = temp_H[nu, :].unsqueeze(dim=0)
            c_1 = torch.conj(a_1.t())
            upper = Noise_and_Interference.inverse() @ c_1
            down = torch.norm(upper, 'fro')
            v_temp = (torch.sqrt(D[nb, nu, 0])/down) * upper
            V_1[nb, :, nu] = v_temp.squeeze()
    return F, V_1

class BiGraph_net_F_1(nn.Module):
    def __init__(self, num_ant_per_chain, num_M, num_iter, P_pow, channel):                                            #__init__()写成__int__()会报错多余参数
        super(BiGraph_net_F_1, self).__init__()
        self.num_ant_per_chain = num_ant_per_chain
        self.num_M = num_M
        self.P_pow = P_pow
        self.num_iter = num_iter
        self.user_channel = channel[0, :]
        self.antenna_channel = channel[1, :]
        self.analogue_channel = channel[2, :]
        self.digital_channel = channel[3, :]
        self.C_func = User_message_generate(self.num_ant_per_chain, self.user_channel, num_M)
        self.B_func = Antenna_message_generate(self.num_ant_per_chain, self.antenna_channel, num_M)
        self.F_func = Analogue_update(num_ant_per_chain, self.analogue_channel, num_M)
        self.D_func = Digital_update(self.digital_channel, num_M)

    def forward(self, H, F_1, num_rf):

        batch_size, num_user, num_ant_2 = H.shape
        num_ant = int(num_ant_2/2)

        D = (self.P_pow/(num_user*self.num_ant_per_chain)) * torch.ones([batch_size, num_user, 2]).to(device =device_num)
        D_0 = D
        # 初始化使得F满足前Nt与后N_t数平方相加为1
        F_init = torch.zeros([batch_size, self.num_ant_per_chain, num_rf], dtype=torch.complex128).to(device =device_num)

        for nr in np.arange(0, num_rf):                                                                                 #将SDR_AltMin (EGT) 的F作为initial value
            F_init[:, :, nr] = F_1[:, self.num_ant_per_chain*nr : self.num_ant_per_chain*(nr+1), nr]
        F = torch.cat([F_init.real, F_init.imag], dim=1)

        F_0 = F
        B = torch.randn([batch_size, num_rf, num_user, self.num_M]).to(device =device_num)
        Rate_iter = torch.zeros([batch_size, 2, self.num_iter]).to(device =device_num)
        for ni in np.arange(0, self.num_iter):
            C = self.C_func(D, B, H)
            F = self.F_func(C, F, F_0)
            Rate_1 = Sum_rate_cal(H, F, D, self.num_ant_per_chain, num_ant, num_ant_2)[2]
            Rate_iter[:, 0, ni] = Rate_1.squeeze()
            B = self.B_func(C, F, H)
            D = self.D_func(D, B, self.P_pow/self.num_ant_per_chain, D_0)
            F_2, V, Rate_2 = Sum_rate_cal(H, F, D, self.num_ant_per_chain, num_ant, num_ant_2)
            Rate_iter[:, 1, ni] = Rate_2.squeeze()
        return F_2, V, Rate_iter, D

def Sum_rate_cal(H, F, D, num_ant_per_chain, num_ant, num_ant_2):

    F_c = F[:, np.arange(0, num_ant_per_chain), :] + 1j * F[:, np.arange(0, num_ant_per_chain) + num_ant_per_chain,:]  # transfer the output F_1 of DNN to the analogue beamforming matrix F
    F_temp = F_c.type(torch.complex128)
    H_1 = H[:, :, 0:num_ant] + 1j * H[:, :, num_ant:num_ant_2]
    F_1, V = generalBF_generate(D, H_1, F_temp, num_ant_per_chain)
    a = Sum_rate(H_1, F_1, V)

    return F_1, V, a


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