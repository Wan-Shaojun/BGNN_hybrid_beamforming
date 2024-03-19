import numpy as np
import random
import torch
import math
from BGNN_generate_channel import *
from BGNN_global_value import *
from BGNN_graph_F_1 import *
from BGNN_graph_ZF import *
from BGNN_graph_CNN import *
from BGNN_Train import *
import matplotlib.pyplot as plt
import time
# import scipy.io as io

########################################################################################################################直接对两个BGNN进行对比############################################################
dir = './BGNN_performance/'
device_num = device_num_get()

num_Epoch, num_batch, batch_size = 1, 2, 1000
Epoch_size = int(num_batch * batch_size)

SNR = 0
P_pow = 10**(SNR/10)

num_cl, num_ray,  = 5, 2
num_iter, num_ant_per_chain = 2, 4
num_M = num_ant_per_chain * 1

num_user = 7
num_rf = 8
num_ant = num_ant_per_chain * num_rf

torch.cuda.empty_cache()
# hidden_channel = hidden_channel_generate_ZF(num_ant_per_chain, num_M).to(device=device_num)
# DNN = BiGraph_net_ZF(num_ant_per_chain, num_M, num_iter, P_pow, hidden_channel).to(device=device_num)
hidden_channel = hidden_channel_generate_1(num_ant_per_chain, num_M).to(device=device_num)
DNN = BiGraph_net_F_1(num_ant_per_chain, num_M, num_iter, P_pow, hidden_channel).to(device=device_num)
DNN.load_state_dict(torch.load(dir + "[100_60_32]_BGNN_MLP_Opt_Simple_various_user_numAntennaPerChain_4_numM_4_numIter_2_InitialRFchain_8_InitialUser_2_SNR_0_dB.pth"))
hidden_channel_1 = hidden_channel_generate_1(num_ant_per_chain, num_M).to(device=device_num)
DNN_1 = BiGraph_net_F_1(num_ant_per_chain, num_M, num_iter, P_pow, hidden_channel_1).to(device=device_num)
DNN_1.load_state_dict(torch.load(dir + "[100_60_48]_BGNN_MLP_Opt_Simple_various_user_numAntennaPerChain_4_numM_4_numIter_2_InitialRFchain_8_InitialUser_2_SNR_0_dB.pth"))

loss_test, loss_test_1 = Test_Compare(DNN, DNN_1, num_Epoch, num_batch, batch_size, num_user, num_ant, num_rf, P_pow)
rate = torch.mean(torch.mean(loss_test, dim=1))
rate_1 = torch.mean(torch.mean(loss_test_1, dim=1))

x = 1

