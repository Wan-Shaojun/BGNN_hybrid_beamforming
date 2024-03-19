import numpy as np
import random
import torch
import math
from BGNN_generate_channel import *
from BGNN_global_value import *
from FCMLP_HB import *
from BGNN_Train import *
import matplotlib.pyplot as plt
import time
# import scipy.io as io

dir = './BGNN_performance/'
device_num = device_num_get()

num_Epoch, num_batch, batch_size = 1, 3, 1000
Epoch_size = int(num_batch * batch_size)

SNR = -10
P_pow = 10**(SNR/10)

num_cl, num_ray,  = 5, 2
num_ant_per_chain = 8

num_user = 4
num_rf = 5
num_ant = num_ant_per_chain * num_rf

torch.cuda.empty_cache()                                                                                                     # trainning with num_user = 4, num_rf = 8, num_ant = 16
hidden_channel = hidden_channel_generate_FCMLP(num_ant, num_user, num_rf).to(device=device_num)
DNN = HB_FC_MLP(num_ant_per_chain, num_user, num_rf, num_ant, P_pow, hidden_channel).to(device=device_num)
DNN.load_state_dict(torch.load(dir + "FCMLP_HB_numAntennaPerChain_8_User_4_RFChain_5_SNR_-10_dB.pth"))

torch.cuda.synchronize()
t0 = time.time()
loss_test = Test_FCMLP(DNN, num_Epoch, num_batch, batch_size, num_user, num_ant, num_rf, P_pow)
torch.cuda.synchronize()
t1 = time.time()
test_time = t1 - t0
time_per_sample = test_time / (num_Epoch * num_batch * batch_size)

torch.cuda.empty_cache()
a = torch.mean(torch.mean(loss_test, dim=1))
loss_epoch = -torch.mean(loss_test, dim=1)
# loss_epoch_2 = -torch.mean(loss_all_2, dim=1)
x = torch.linspace(1, num_Epoch, num_Epoch)

