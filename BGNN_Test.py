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
import matplotlib.pyplot as plt
import time
# import scipy.io as io

dir = './BGNN_performance/'
device_num = device_num_get()

num_Epoch, num_batch, batch_size = 1, 3, 1000
Epoch_size = int(num_batch * batch_size)

SNR = -5
P_pow = 10**(SNR/10)

num_cl, num_ray,  = 5, 2
num_iter = 10
num_ant_per_chain = 4
num_M = num_ant_per_chain * 1

num_user = 4
num_rf = 5
num_ant = num_ant_per_chain * num_rf

codebook = DFT_codebook_generate(num_ant_per_chain).to(device = device_num)
# batch_size_1 = 100
# temp_channel = channel_generate(batch_size_1, num_user, num_ant, num_cl, num_ray).to(device = device_num)
# F_opt = FD_MMSE_1(temp_channel, P_pow)
# F_RF, F_BB, Index = SSP_Mul_Precoding(F_opt, codebook, num_rf)
# Rate_SSP = torch.mean(Sum_rate(temp_channel, F_RF, F_BB))

torch.cuda.empty_cache()
# hidden_channel = hidden_channel_generate_1(num_ant_per_chain, num_M).to(device=device_num)
# DNN = BiGraph_net_F_2(num_ant_per_chain, num_M, num_iter, P_pow, hidden_channel, codebook).to(device=device_num)
# hidden_channel = hidden_channel_generate_ZF(num_ant_per_chain, num_M).to(device=device_num)
# DNN = BiGraph_net_ZF(num_ant_per_chain, num_M, num_iter, P_pow, hidden_channel).to(device=device_num)
hidden_channel = hidden_channel_generate_1(num_ant_per_chain, num_M).to(device=device_num)
DNN = BiGraph_net_F_1(num_ant_per_chain, num_M, num_iter, P_pow, hidden_channel).to(device=device_num)
# hidden_channel = hidden_channel_generate_CNN(num_ant_per_chain, num_M).to(device=device_num)
# DNN = BiGraph_CNN(num_ant_per_chain, num_M, num_iter, P_pow, hidden_channel).to(device=device_num)
DNN.load_state_dict(torch.load(dir + "Without_BN_BGNN_MLP_Opt_Simple_numAntennaPerChain_4_numM_4_numIter_10_User_4_RFChain_5_SNR_-5_dB.pth"))

torch.cuda.synchronize()
t0 = time.time()
# loss_test = Test_F(DNN, num_Epoch, num_batch, batch_size, test_channel, num_rf, P_pow, F_1)
# loss_test = Test_Codebook(DNN, num_Epoch, num_batch, batch_size, num_user, num_ant, num_rf, P_pow, codebook)
loss_test = Test(DNN, num_Epoch, num_batch, batch_size, num_user, num_ant, num_rf, P_pow)
torch.cuda.synchronize()
t1 = time.time()
test_time = t1 - t0
time_per_sample = test_time / (num_Epoch * num_batch * batch_size)

torch.cuda.empty_cache()
a = torch.mean(torch.mean(loss_test, dim=1))
loss_epoch = -torch.mean(loss_test, dim=1)
# loss_epoch_2 = -torch.mean(loss_all_2, dim=1)
x = torch.linspace(1, num_Epoch, num_Epoch)

