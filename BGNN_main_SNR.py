import numpy as np
import random
import torch
import math
from BGNN_generate_channel import *
from BGNN_global_value import *
# from BGNN_graph import *
from BGNN_graph_F_1 import *
from BGNN_graph_ZF import *
from BGNN_Train import *
from BGNN_train_1 import *
import matplotlib.pyplot as plt
import time
from torch.nn.parallel import DataParallel
# import scipy.io as io

dir = './BGNN_performance'
device_num = device_num_get()

num_Epoch, num_batch, batch_size = 100, 300, 32
# num_Epoch, num_batch, batch_size = 2, 2, 4
num_cl, num_ray,  = 5, 2

num_iter, num_ant_per_chain = 2, 4
num_M = num_ant_per_chain * 1

num_user = 4
num_rf = 5
num_ant = num_ant_per_chain * num_rf

SNR_low = 10
SNR_gap = -5
num_dot = 7

loss_all = torch.zeros([num_dot, num_Epoch, num_batch])
for nu in np.arange(0, num_dot):
    if (nu != 4):
        snr = SNR_low + SNR_gap * nu
        P_pow = 10**(snr/10)
        print("SNR:", snr)

        torch.cuda.empty_cache()
        hidden_channel = hidden_channel_generate_ZF(num_ant_per_chain, num_M).to(device=device_num)
        DNN = BiGraph_net_ZF(num_ant_per_chain, num_M, num_iter, P_pow, hidden_channel).to(device=device_num)
        loss_all[nu, :, :] = Update(DNN, num_Epoch, num_batch, batch_size, num_user, num_ant, num_rf, snr)


loss_epoch = torch.mean(loss_all[0, :, :], dim=1)
x = torch.linspace(1, num_Epoch, num_Epoch)

plt.figure(1)
plt.plot(x, loss_epoch.cpu().detach().numpy(), label = 'PC, BGNN')
plt.xlabel('Epoch number')
plt.ylabel('Sum Rate(bps/Hz)')
plt.legend(loc = 'best')
plt.show()

x = 0 + 1


