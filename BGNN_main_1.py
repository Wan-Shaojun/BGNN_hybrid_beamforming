import numpy as np
import random
import torch
import math
from BGNN_generate_channel import *
from BGNN_global_value import *
from BGNN_graph_F_1 import *
from BGNN_graph_ZF import *
from BGNN_Train import *
from BGNN_train_1 import *
import matplotlib.pyplot as plt
import time
from torch.nn.parallel import DataParallel
# import scipy.io as io

########################################################################################################################对应BGNN_train_1, 对BGNN进行various_system训练###############################
dir = './BGNN_performance'
device_num = device_num_get()

num_Epoch, num_batch, batch_size = 100, 60, 48
# num_Epoch, num_batch, batch_size = 2, 2, 4
SNR = 0
P_pow = 10**(SNR/10)
num_cl, num_ray,  = 5, 2

torch.cuda.empty_cache()
num_iter, num_ant_per_chain = 2, 4
num_M = num_ant_per_chain * 1

# max_user = 8
# max_rf = 8
num_user = 2
num_rf = 8
num_ant = num_ant_per_chain * num_rf

hidden_channel = hidden_channel_generate_1(num_ant_per_chain, num_M).to(device=device_num)
DNN = BiGraph_net_F_1(num_ant_per_chain, num_M, num_iter, P_pow, hidden_channel).to(device=device_num)
# hidden_channel = hidden_channel_generate_ZF(num_ant_per_chain, num_M).to(device=device_num)
# DNN = BiGraph_net_ZF(num_ant_per_chain, num_M, num_iter, P_pow, hidden_channel).to(device=device_num)
# loss_all = Update_Various_System(DNN, num_Epoch, num_batch, batch_size, max_user, max_rf, num_ant_per_chain, SNR, num_iter)
loss_all = Update_Various_User(DNN, num_Epoch, num_batch, batch_size, num_user, num_ant, num_rf, SNR, num_iter)

loss_epoch = torch.mean(loss_all, dim=1)
x = torch.linspace(1, num_Epoch, num_Epoch)

plt.figure(1)
plt.plot(x, loss_epoch.cpu().detach().numpy(), label = 'PC, BGNN')
plt.xlabel('Epoch number')
plt.ylabel('Sum Rate(bps/Hz)')
plt.legend(loc = 'best')
plt.show()

x = x + 1


