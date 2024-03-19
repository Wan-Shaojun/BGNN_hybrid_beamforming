import numpy as np
import random
import torch
import math
from BGNN_global_value import *
# from scipy.io import savemat
from BGNN_generate_channel import *
# from BGNN_graph import *
from BGNN_bentchmark import *

device_num = device_num_get()
dir = './BGNN_performance'


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

def FD_Sum_rate(H, V):

    batch_size = H.shape[0]
    num_user = H.shape[1]
    Rate = torch.zeros([batch_size, 1]).to(device =device_num)
    for nb in np.arange(0, batch_size):
        temp = torch.pow(torch.abs(H[nb, :, :] @ V[nb, :, :]), 2)
        signal_power = torch.diag(temp)
        signal_noise_power = torch.sum(temp, 1)
        for nu in np.arange(0, num_user):
            Rate[nb] = Rate[nb] + torch.log2(1 + (signal_power[nu] / (signal_noise_power[nu] - signal_power[nu] + 1)))

    return Rate

# def Loss(Rate_iter):
#
#     num_iter = Rate_iter.shape[1]
#
#     loss_temp = Rate_iter[:, num_iter, 1]
#     loss = torch.mean(loss_temp)
#
#     return loss

def Loss_1(Rate):

    num_iter = Rate.shape[2]
    ## max \sum_{t=1}^{T} Rate^{t}
    loss_iter = - torch.mean(Rate, dim=0)
    loss_iter_2 = torch.mean(loss_iter, dim=0)
    # loss_iter_2 = loss_iter[1, :]
    loss = (1/num_iter) * torch.sum(loss_iter_2)
    ## max Rate^{T}
    # loss = -torch.mean(Rate[:, num_iter-1])

    return loss

def Update_FCCNN(DNN, num_Epoch, num_batch, batch_size, num_user, num_ant, SNR):

    torch.cuda.empty_cache()
    DNN.train()
    Rate_all = torch.zeros([num_Epoch, num_batch])
    Opt = torch.optim.Adam(DNN.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(Opt, T_max=100)
    P_pow = 10**(SNR/10)
    for ne in np.arange(0, num_Epoch):
        print("Epoch number:", ne)
        for nb in np.arange(0, num_batch):
            loss = 0
            temp_channel = channel_generate(batch_size, num_user, num_ant, num_cl = 5, num_ray = 2).to(device =device_num)

            Opt.zero_grad()  # 对每个batch训练时，将梯度设为零
            V, Rate = DNN.forward(temp_channel)
            loss = -torch.mean(Rate)
            loss.backward()

            Rate_temp = torch.mean(Rate)
            Rate_all[ne, nb] = Rate_temp
            Opt.step()
            torch.cuda.empty_cache()
        scheduler.step()
        rate_epoch = torch.mean(Rate_all[ne, :])
        print('The Sum Rate in this Epoch is :', rate_epoch.cpu().detach().numpy())
        if (ne % 1 == 0):
            torch.save(DNN.state_dict(), dir + '/FCCNN_DB_User_%d_Antenna_%d_SNR_%d_dB.pth' % (num_user, num_ant, SNR))
        torch.cuda.empty_cache()
    return Rate_all

def Update_FCMLP(DNN, num_Epoch, num_batch, batch_size, num_user, num_ant, num_rf, SNR):

    torch.cuda.empty_cache()
    DNN.train()
    Rate_all = torch.zeros([num_Epoch, num_batch])
    Opt = torch.optim.Adam(DNN.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(Opt, T_max=100)
    num_ant_per_chain = int(num_ant / num_rf)
    P_pow = 10**(SNR/10)
    for ne in np.arange(0, num_Epoch):
        print("Epoch number:", ne)
        for nb in np.arange(0, num_batch):
            loss = 0
            temp_channel = channel_generate(batch_size, num_user, num_ant, num_cl = 5, num_ray = 2).to(device =device_num)

            Opt.zero_grad()  # 对每个batch训练时，将梯度设为零
            F, V, Rate = DNN.forward(temp_channel)
            loss = -torch.mean(Rate)
            loss.backward()

            Rate_temp = torch.mean(Rate)
            Rate_all[ne, nb] = Rate_temp
            Opt.step()
            torch.cuda.empty_cache()
        scheduler.step()
        rate_epoch = torch.mean(Rate_all[ne, :])
        print('The Sum Rate in this Epoch is :', rate_epoch.cpu().detach().numpy())
        if (ne % 1 == 0):
            torch.save(DNN.state_dict(), dir + '/FCMLP_HB_numAntennaPerChain_%d_User_%d_RFChain_%d_SNR_%d_dB.pth' % (num_ant_per_chain, num_user, num_rf, SNR))
        torch.cuda.empty_cache()
    return Rate_all

def Test_FCMLP(DNN, num_Epoch, num_batch, batch_size, num_user, num_ant, num_rf, P_pow):

    torch.cuda.empty_cache()
    DNN.eval()
    Rate_all = torch.zeros([num_Epoch, num_batch])
    num_ant_per_chain = int(num_ant / num_rf)
    for ne in np.arange(0, num_Epoch):
        print("Epoch number:", ne)
        for nb in np.arange(0, num_batch):
            temp_channel = channel_generate(batch_size, num_user, num_ant, num_cl = 5, num_ray = 2).to(device =device_num)

            F, V, Rate = DNN.forward(temp_channel)
            Rate_temp = torch.mean(Rate)

            Rate_all[ne, nb] = Rate_temp
            torch.cuda.empty_cache()
    return Rate_all


########################################################################################################################BGNN的训练和测试########################################################################################
def Update(DNN, num_Epoch, num_batch, batch_size, num_user, num_ant, num_rf, SNR):

    torch.cuda.empty_cache()
    DNN.train()
    Rate_all = torch.zeros([num_Epoch, num_batch])
    Opt = torch.optim.Adam(DNN.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(Opt, T_max=100)
    num_ant_per_chain = int(num_ant / num_rf)
    P_pow = 10**(SNR/10)
    for ne in np.arange(0, num_Epoch):
        print("Epoch number:", ne)
        for nb in np.arange(0, num_batch):
            loss = 0
            temp_channel = channel_generate(batch_size, num_user, num_ant, num_cl = 5, num_ray = 2).to(device =device_num)
            temp_channel_float = torch.cat([temp_channel.real, temp_channel.imag], dim=2)

            F_target_temp_2 = F_init_generate(temp_channel_float, num_rf)

            Opt.zero_grad()  # 对每个batch训练时，将梯度设为零
            F, V, Rate_iter, D = DNN.forward(temp_channel_float, F_target_temp_2, num_rf)                   ### BGNN_graph_F
            loss = Loss_1(Rate_iter)

            Rate_temp = torch.mean(Sum_rate(temp_channel, F, V))
            Rate_all[ne, nb] = Rate_temp

            loss.backward()
            Opt.step()
            torch.cuda.empty_cache()
        scheduler.step()
        rate_epoch = torch.mean(Rate_all[ne, :])
        print('The Sum Rate in this Epoch is :', rate_epoch.cpu().detach().numpy())
        if (ne % 1 == 0):
            torch.save(DNN.state_dict(), dir + '/Without_BN_BGNN_MLP_Opt_Simple_numAntennaPerChain_%d_numM_%d_numIter_%d_User_%d_RFChain_%d_SNR_%d_dB.pth' % (num_ant_per_chain, num_ant_per_chain, 10, num_user, num_rf, SNR))
        torch.cuda.empty_cache()
    return Rate_all

def Test(DNN, num_Epoch, num_batch, batch_size, num_user, num_ant, num_rf, P_pow):

    torch.cuda.empty_cache()
    DNN.eval()
    Rate_all = torch.zeros([num_Epoch, num_batch])
    num_ant_per_chain = int(num_ant / num_rf)
    # model_paramter = DNN.named_parameters()
    for ne in np.arange(0, num_Epoch):
        print("Epoch number:", ne)
        for nb in np.arange(0, num_batch):
            temp_channel = channel_generate(batch_size, num_user, num_ant, num_cl = 5, num_ray = 2).to(device =device_num)
            temp_channel_float = torch.cat([temp_channel.real, temp_channel.imag], dim=2)

            # F_target_temp, iter_num = Wei_Yu_Analogue_Precoding(temp_channel, num_rf, P_pow)
            # F_target_temp, D_temp, iter_num = Wei_Yu_ZF_Precoding(temp_channel, P_pow, num_rf)
            # rate_target = torch.mean(Sum_rate(temp_channel, F_target_temp, D_temp))
            # F_target_temp_1, D_temp_1, iter_num_1 = Wei_Yu_MMSE_Precoding(temp_channel, P_pow, num_rf)
            # rate_target_1 = torch.mean(Sum_rate(temp_channel, F_target_temp_1, D_temp_1))
            #
            F_target_temp_2 = F_init_generate(temp_channel_float, num_rf)
            # V_target_temp_2 = FD_ZF_1(temp_channel@F_target_temp_2, P_pow/num_ant_per_chain)
            # V_target_temp_3 = FD_MMSE_1(temp_channel@F_target_temp_2, P_pow/num_ant_per_chain)
            # rate_target_2 = torch.mean(Sum_rate(temp_channel, F_target_temp_2, V_target_temp_2))
            # rate_target_3 = torch.mean(Sum_rate(temp_channel, F_target_temp_2, V_target_temp_3))

            F, V, Rate_iter, D = DNN.forward(temp_channel_float, F_target_temp_2, num_rf)
            Rate_temp = torch.mean(Sum_rate(temp_channel, F, V))
            # V_1 = FD_ZF_1(temp_channel@F, P_pow/num_ant_per_chain)
            # V_2 = FD_MMSE_1(temp_channel@F, P_pow/num_ant_per_chain)
            # rate_1 = torch.mean(Sum_rate(temp_channel, F, V_1))
            # rate_2 = torch.mean(Sum_rate(temp_channel, F, V_2))

            Rate_all[ne, nb] = Rate_temp
            torch.cuda.empty_cache()
        # rate_epoch = torch.mean(Rate_all[ne, :])
        # print('The Sum Rate in this Epoch is :', rate_epoch.cpu().detach().numpy())
    return Rate_all

########################################################################################################################给定通信信道和analog_precoding初始值，直接对比两个BGNN的性能表现################################
def Test_Compare(DNN, DNN_1, num_Epoch, num_batch, batch_size, num_user, num_ant, num_rf, P_pow):

    torch.cuda.empty_cache()
    DNN.eval()
    DNN_1.eval()
    Rate_all = torch.zeros([num_Epoch, num_batch])
    Rate_all_1 = torch.zeros([num_Epoch, num_batch])
    num_ant_per_chain = int(num_ant / num_rf)
    for ne in np.arange(0, num_Epoch):
        print("Epoch number:", ne)
        for nb in np.arange(0, num_batch):
            temp_channel = channel_generate(batch_size, num_user, num_ant, num_cl = 5, num_ray = 2).to(device =device_num)
            temp_channel_float = torch.cat([temp_channel.real, temp_channel.imag], dim=2)

            # F_target_temp, iter_num = Wei_Yu_Analogue_Precoding(temp_channel, num_rf, P_pow)
            # F_target_temp, D_temp, iter_num = Wei_Yu_ZF_Precoding(temp_channel, P_pow, num_rf)
            # rate_target = torch.mean(Sum_rate(temp_channel, F_target_temp, D_temp))
            # F_target_temp_1, D_temp_1, iter_num_1 = Wei_Yu_MMSE_Precoding(temp_channel, P_pow, num_rf)
            # rate_target_1 = torch.mean(Sum_rate(temp_channel, F_target_temp_1, D_temp_1))
            #
            F_target_temp_2 = F_init_generate(temp_channel_float, num_rf)
            # V_target_temp_2 = FD_ZF_1(temp_channel@F_target_temp_2, P_pow/num_ant_per_chain)
            # V_target_temp_3 = FD_MMSE_1(temp_channel@F_target_temp_2, P_pow/num_ant_per_chain)
            # rate_target_2 = torch.mean(Sum_rate(temp_channel, F_target_temp_2, V_target_temp_2))
            # rate_target_3 = torch.mean(Sum_rate(temp_channel, F_target_temp_2, V_target_temp_3))

            F, V, Rate_iter, D = DNN.forward(temp_channel_float, F_target_temp_2, num_rf)
            Rate_temp = torch.mean(Sum_rate(temp_channel, F, V))
            # V_1 = FD_ZF_1(temp_channel@F, P_pow/num_ant_per_chain)
            # V_2 = FD_MMSE_1(temp_channel@F, P_pow/num_ant_per_chain)
            # rate_1 = torch.mean(Sum_rate(temp_channel, F, V_1))
            # rate_2 = torch.mean(Sum_rate(temp_channel, F, V_2))

            F_1, V_1, Rate_iter_1, D_1 = DNN_1.forward(temp_channel_float, F_target_temp_2, num_rf)
            Rate_temp_1 = torch.mean(Sum_rate(temp_channel, F_1, V_1))

            Rate_all[ne, nb] = Rate_temp
            Rate_all_1[ne, nb] = Rate_temp_1
            torch.cuda.empty_cache()
        # rate_epoch = torch.mean(Rate_all[ne, :])
        # print('The Sum Rate in this Epoch is :', rate_epoch.cpu().detach().numpy())
    return Rate_all, Rate_all_1


########################################################################################################################基于codebook的BGNN的训练和测试####################################################
def Update_Codebook(DNN, num_Epoch, num_batch, batch_size, num_user, num_ant, num_rf, SNR, codebook):

    torch.cuda.empty_cache()
    DNN.train()
    Rate_all = torch.zeros([num_Epoch, num_batch])
    Opt = torch.optim.Adam(DNN.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(Opt, T_max=100)
    num_ant_per_chain = int(num_ant / num_rf)
    P_pow = 10**(SNR/10)
    for ne in np.arange(0, num_Epoch):
        print("Epoch number:", ne)
        for nb in np.arange(0, num_batch):
            loss = 0
            temp_channel = channel_generate(batch_size, num_user, num_ant, num_cl = 5, num_ray = 2).to(device =device_num)
            temp_channel_float = torch.cat([temp_channel.real, temp_channel.imag], dim=2)

            F_target_temp_2 = F_init_generate(temp_channel_float, num_rf)
            F_opt = FD_MMSE_1(temp_channel, P_pow)
            F_RF, F_BB, Index_SSP = SSP_Mul_Precoding(F_opt, codebook, num_rf)
            Rate_SSP = torch.mean(Sum_rate(temp_channel, F_RF, F_BB))
            # F_target_temp_2 = F_RF

            Opt.zero_grad()  # 对每个batch训练时，将梯度设为零
            F, V, Index_BGNN, D, loss = DNN.forward(temp_channel_float, F_target_temp_2, num_rf, Index_SSP)                   ### BGNN_graph_F
            # loss = Loss_1(Rate_iter)

            Rate_temp = torch.mean(Sum_rate(temp_channel, F, V))
            Rate_all[ne, nb] = Rate_temp

            loss.backward()
            Opt.step()
            torch.cuda.empty_cache()
        scheduler.step()
        rate_epoch = torch.mean(Rate_all[ne, :])
        print('The Sum Rate in this Epoch is :', rate_epoch.cpu().detach().numpy())
        if (ne % 1 == 0):
            torch.save(DNN.state_dict(), dir + '/Supervised_SSP_BGNN_MLP_Opt_Codebook_numAntennaPerChain_%d_numM_%d_numIter_%d_User_%d_RFChain_%d_SNR_%d_dB.pth' % (num_ant_per_chain, num_ant_per_chain, 2, num_user, num_rf, SNR))
        torch.cuda.empty_cache()
    return Rate_all

def Test_Codebook(DNN, num_Epoch, num_batch, batch_size, num_user, num_ant, num_rf, SNR, codebook):

    torch.cuda.empty_cache()
    DNN.eval()
    Rate_all = torch.zeros([num_Epoch, num_batch])
    num_ant_per_chain = int(num_ant / num_rf)
    P_pow = 10**(SNR/10)
    for ne in np.arange(0, num_Epoch):
        print("Epoch number:", ne)
        for nb in np.arange(0, num_batch):
            loss = 0
            temp_channel = channel_generate(batch_size, num_user, num_ant, num_cl = 5, num_ray = 2).to(device =device_num)
            temp_channel_float = torch.cat([temp_channel.real, temp_channel.imag], dim=2)

            F_target_temp_2 = F_init_generate(temp_channel_float, num_rf)

            F, V, Rate_iter, D = DNN.forward(temp_channel_float, F_target_temp_2, num_rf)                   ### BGNN_graph_F

            Rate_temp = torch.mean(Sum_rate(temp_channel, F, V))
            Rate_all[ne, nb] = Rate_temp
            torch.cuda.empty_cache()
        rate_epoch = torch.mean(Rate_all[ne, :])
        torch.cuda.empty_cache()
    return Rate_all
