import numpy as np
import random
import torch
import math
from BGNN_generate_channel import *
from BGNN_global_value import *
from BGNN_graph_F_1 import *
from BGNN_graph_F_2 import *
from BGNN_graph_ZF import *
# from BGNN_graph_CNN import *
from BGNN_Train import *
from BGNN_train_1 import *
import matplotlib.pyplot as plt
import time
from torch.nn.parallel import DataParallel
# import scipy.io as io

# dir = './BGNN_performance_codebook'
dir = './BGNN_performance'
device_num = device_num_get()

num_Epoch, num_batch, batch_size = 100, 100, 32
# num_Epoch, num_batch, batch_size = 2, 2, 8
SNR = -5
P_pow = 10**(SNR/10)
num_cl, num_ray,  = 5, 2

torch.cuda.empty_cache()
num_iter = 10
num_ant_per_chain = 4
num_M = num_ant_per_chain * 1

num_user = 4
num_rf = 5
num_ant = num_ant_per_chain * num_rf

codebook = DFT_codebook_generate(num_ant_per_chain).to(device = device_num)

# batch_size_1 = 3000
# temp_channel = channel_generate(batch_size_1, num_user, num_ant, num_cl, num_ray).to(device = device_num)
# F_opt = FD_MMSE_1(temp_channel, P_pow)
# F_RF, F_BB = SSP_Mul_Precoding(F_opt, codebook, num_rf)
# Rate_SSP = torch.mean(Sum_rate(temp_channel, F_RF, F_BB))

# hidden_channel = hidden_channel_generate_1(num_ant_per_chain, num_M).to(device=device_num)
# DNN = BiGraph_net_F_2(num_ant_per_chain, num_M, num_iter, P_pow, hidden_channel, codebook).to(device=device_num)
# loss_all = Update_Codebook(DNN, num_Epoch, num_batch, batch_size, num_user, num_ant, num_rf, SNR, codebook)
hidden_channel = hidden_channel_generate_1(num_ant_per_chain, num_M).to(device=device_num)
DNN = BiGraph_net_F_1(num_ant_per_chain, num_M, num_iter, P_pow, hidden_channel).to(device=device_num)
# hidden_channel = hidden_channel_generate_ZF(num_ant_per_chain, num_M).to(device=device_num)
# DNN = BiGraph_net_ZF(num_ant_per_chain, num_M, num_iter, P_pow, hidden_channel).to(device=device_num)
# hidden_channel = hidden_channel_generate_CNN(num_ant_per_chain, num_M).to(device=device_num)
# DNN = BiGraph_CNN(num_ant_per_chain, num_M, num_iter, P_pow, hidden_channel).to(device=device_num)
# loss_all = Update_Various_RF_chain(DNN, num_Epoch, num_batch, batch_size, num_user, num_ant, num_rf, SNR, num_iter)
# loss_all = Update_Various_User(DNN, num_Epoch, num_batch, batch_size, num_user, num_ant, num_rf, SNR, num_iter)
loss_all = Update(DNN, num_Epoch, num_batch, batch_size, num_user, num_ant, num_rf, SNR)

# torch.cuda.empty_cache()
# num_iter = 2
# num_user = 4
# num_rf = 5
# num_ant = 40
# num_ant_per_chain = int(num_ant / num_rf)
# num_M = num_ant_per_chain
# hidden_channel = hidden_channel_generate_Nt(num_M).cuda()
# DNN = BiGraph_net_F_Nt(num_rf, num_M, num_iter, P_pow, hidden_channel).cuda()
# loss_all = Update(DNN, num_Epoch, num_batch, batch_size, num_user, num_ant, num_rf, P_pow)
# torch.save(DNN.state_dict(), dir + '/BGNN_F_Nt_numAntennaPerChain_%d_numM_%d_numIter_%d_User_%d_RFChain_%d_Power_%d.pth' % (num_ant_per_chain, num_M, num_iter, num_user, num_rf, P_pow))

loss_epoch = torch.mean(loss_all, dim=1)
x = torch.linspace(1, num_Epoch, num_Epoch)

plt.figure(1)
plt.plot(x, loss_epoch.cpu().detach().numpy(), label = 'PC, BGNN')
plt.xlabel('Epoch number')
plt.ylabel('Sum Rate(bps/Hz)')
plt.legend(loc = 'best')
plt.show()

x = x + 1