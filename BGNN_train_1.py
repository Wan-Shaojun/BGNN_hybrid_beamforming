import numpy as np
import random
import torch
import math
from BGNN_global_value import *
from BGNN_generate_channel import *
from BGNN_bentchmark import *

########################################################################################################################对BGNN进行various system的训练###################################################################
dir = './BGNN_performance'
device_num = device_num_get()

def Sum_rate(H, F, V):

    batch_size = H.shape[0]
    num_user = H.shape[1]
    Rate = torch.zeros([batch_size, 1]).to(device=device_num)
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
    Rate = torch.zeros([batch_size, 1]).to(device=device_num)
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
#     loss_temp = Rate_iter[:, num_iter-1, 1]
#     loss = - torch.mean(loss_temp)
#
#     return loss

def Loss_1(Rate):

    num_iter = Rate.shape[2]
    ## max \sum_{t=1}^{T} Rate^{t}
    loss_iter = - torch.mean(Rate, dim=0)
    loss_iter_2 = (1/2) * torch.sum(loss_iter, dim=0)
    loss = (1/num_iter) * torch.sum(loss_iter_2)
    ## max Rate^{T}
    # loss = -torch.mean(Rate[:, num_iter-1])

    return loss

def Update_Various_System(DNN, num_Epoch, num_batch, batch_size, max_user, max_rf, num_ant_per_chain, SNR, num_iter):
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
            temp_user = random.randint(2, max_rf)
            temp_rf = random.randint(2, max_rf)
            temp_ant = temp_rf * num_ant_per_chain
            temp_channel = channel_generate(batch_size, temp_user, temp_ant, num_cl=5, num_ray=2).to(device=device_num)
            temp_channel_float = torch.cat([temp_channel.real, temp_channel.imag], dim=2)

            F_target_temp_2 = F_init_generate(temp_channel_float, temp_rf)

            Opt.zero_grad()  # 对每个batch训练时，将梯度设为零
            F, V, Rate_iter, D = DNN.forward(temp_channel_float, F_target_temp_2, temp_rf)
            loss = Loss_1(Rate_iter)  # Average iteration
            loss.backward()
            Opt.step()

            Rate_temp = torch.mean(Sum_rate(temp_channel, F, V))
            Rate_all[ne, nb] = Rate_all[ne, nb] + Rate_temp
            torch.cuda.empty_cache()
        scheduler.step()
        rate_epoch = torch.mean(Rate_all[ne, :])
        print('The Sum Rate in this Epoch is :', rate_epoch.cpu().detach().numpy())
        if (ne % 1 == 0):
            torch.save(DNN.state_dict(),dir + '/BGNN_ZF_Simple_various_system_numAntennaPerChain_%d_numM_%d_numIter_%d_MaxRF_%d_MaxUser_%d_SNR_%d_dB.pth' % (num_ant_per_chain, num_ant_per_chain, num_iter, max_rf, max_user, SNR))
        torch.cuda.empty_cache()
    return Rate_all

def Update_Various_User(DNN, num_Epoch, num_batch, batch_size, num_user, num_ant, num_rf, SNR, num_iter):
    torch.cuda.empty_cache()
    DNN.train()
    Rate_all = torch.zeros([num_Epoch, num_batch])
    Opt = torch.optim.Adam(DNN.parameters(), lr=1e-3)
    # scheduler = torch.optim.lr_scheduler.StepLR(Opt, step_size=2, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(Opt, T_max=100)
    num_ant_per_chain = int(num_ant/num_rf)
    P_pow = 10 ** (SNR / 10)
    for ne in np.arange(0, num_Epoch):
        print("Epoch number:", ne)
        for nb in np.arange(0, num_batch):
            loss = 0
            for ni in np.arange(0, 6, 1):
                temp_user = num_user + ni
                temp_channel = channel_generate(batch_size, temp_user, num_ant, num_cl = 5, num_ray = 2).to(device=device_num)
                temp_channel_float = torch.cat([temp_channel.real, temp_channel.imag], dim=2)
                # F_target_temp, iter_num = Wei_Yu_Analogue_Precoding(temp_channel, num_rf, P_pow)
                # F_target_temp, D_temp, iter_num = Wei_Yu_ZF_Precoding(temp_channel, P_pow, num_rf)

                F_target_temp_2 = F_init_generate(temp_channel_float, num_rf)

                Opt.zero_grad()  # 对每个batch训练时，将梯度设为零
                F, V, Rate_iter, D = DNN.forward(temp_channel_float, F_target_temp_2, num_rf)
                loss = Loss_1(Rate_iter)              #Average iteration
                loss.backward()
                Opt.step()

                Rate_temp = torch.mean(Sum_rate(temp_channel, F, V))
                Rate_all[ne, nb] = Rate_all[ne, nb] + Rate_temp
                torch.cuda.empty_cache()
        scheduler.step()
        rate_epoch = torch.mean(Rate_all[ne, :])
        print('The Sum Rate in this Epoch is :', rate_epoch.cpu().detach().numpy())
        if (ne % 1 == 0):
            torch.save(DNN.state_dict(),dir + '/Without_BN_BGNN_MLP_Opt_Simple_various_user_numAntennaPerChain_%d_numM_%d_numIter_%d_InitialRFchain_%d_InitialUser_%d_SNR_%d_dB.pth' % (num_ant_per_chain, num_ant_per_chain, num_iter, num_rf, num_user, SNR))
        torch.cuda.empty_cache()
    return Rate_all

def Update_Various_RF_chain(DNN, num_Epoch, num_batch, batch_size, num_user, num_ant, num_rf, SNR, num_iter):
    torch.cuda.empty_cache()
    DNN.train()
    Rate_all = torch.zeros([num_Epoch, num_batch])
    Opt = torch.optim.Adam(DNN.parameters(), lr=1e-3)
    # scheduler = torch.optim.lr_scheduler.StepLR(Opt, step_size=2, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(Opt, T_max=100)
    num_ant_per_chain = int(num_ant / num_rf)
    P_pow = 10 ** (SNR / 10)
    for ne in np.arange(0, num_Epoch):
        print("Epoch number:", ne)
        for nb in np.arange(0, num_batch):
            loss = 0
            for ni in np.arange(0, 5, 1):
                temp_rf = num_rf + ni
                temp_ant = num_ant_per_chain * temp_rf
                temp_channel = channel_generate(batch_size, num_user, temp_ant, num_cl = 5, num_ray = 2).to(device=device_num)
                temp_channel_float = torch.cat([temp_channel.real, temp_channel.imag], dim=2)

                # F_target_temp, iter_num = Wei_Yu_Analogue_Precoding(temp_channel, temp_rf, P_pow)
                F_target_temp_2 = F_init_generate(temp_channel_float, temp_rf)

                Opt.zero_grad()  # 对每个batch训练时，将梯度设为零
                F, V, Rate_iter, D = DNN.forward(temp_channel_float, F_target_temp_2, temp_rf)
                loss = Loss_1(Rate_iter)              #Average iteration
                loss.backward()
                Opt.step()

                Rate_temp = torch.mean(Sum_rate(temp_channel, F, V))
                Rate_all[ne, nb] = Rate_all[ne, nb] + Rate_temp
                torch.cuda.empty_cache()
        scheduler.step()
        rate_epoch = torch.mean(Rate_all[ne, :])
        print('The Sum Rate in this Epoch is :', rate_epoch.cpu().detach().numpy())
        if (ne % 1 == 0):
            torch.save(DNN.state_dict(),dir + '/Without_BN_BGNN_MLP_Opt_Simple_various_RF_chain_numAntennaPerChain_%d_numM_%d_numIter_%d_InitialRFchain_%d_InitialUser_%d_SNR_%d_dB.pth' % (num_ant_per_chain, num_ant_per_chain * 1, num_iter, num_rf, num_user, SNR))
        torch.cuda.empty_cache()
    return Rate_all


