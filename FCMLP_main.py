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
from torch.nn.parallel import DataParallel
# import scipy.io as io


########################################################################################################################训练Conventional DNN for HBF##############################################################
dir = './BGNN_performance'
device_num = device_num_get()

num_Epoch, num_batch, batch_size = 100, 300, 32
# num_Epoch, num_batch, batch_size = 2, 2, 4l
SNR = -10
P_pow = 10**(SNR/10)
num_cl, num_ray,  = 5, 2

torch.cuda.empty_cache()
num_ant_per_chain = 5

num_user = 4
num_rf = 5
num_ant = num_ant_per_chain * num_rf

hidden_channel = hidden_channel_generate_FCMLP(num_ant, num_user, num_rf).to(device=device_num)
DNN = HB_FC_MLP(num_ant_per_chain, num_user, num_rf, num_ant, P_pow, hidden_channel).to(device=device_num)
loss_all = Update_FCMLP(DNN, num_Epoch, num_batch, batch_size, num_user, num_ant, num_rf, SNR)

loss_epoch = torch.mean(loss_all, dim=1)
x = torch.linspace(1, num_Epoch, num_Epoch)

plt.figure(1)
plt.plot(x, loss_epoch.cpu().detach().numpy(), label = 'PC, BGNN')
plt.xlabel('Epoch number')
plt.ylabel('Sum Rate(bps/Hz)')
plt.legend(loc = 'best')
plt.show()

x = x + 1


