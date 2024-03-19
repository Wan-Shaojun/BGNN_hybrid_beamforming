import numpy as np
import random
import torch
import math
# import scipy.io as io
# from scipy.io import savemat
from BGNN_generate_channel import *
from BGNN_bentchmark import *
from Benchmark_SDR import *
import time

batch_size = 1
num_ant_per_chain = 4
num_user, num_rf = 4, 5
P_pow = 0.1

num_dot = 5
N_init = 4

time_SDR = torch.zeros(1, num_dot).cuda()
rate_SDR = torch.zeros(1, num_dot).cuda()
for nu in np.arange(0, num_dot):
    num_ant_per_chain = N_init + nu
    num_ant = num_rf * num_ant_per_chain
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    t0 = time.time()
    temp_channel = channel_generate(batch_size, num_user, num_ant, num_cl=5, num_ray=2).cuda()
    F_RF_1, F_BB_1, Rate, Useless_index_sdr = SDR_AltMin_Precoding(temp_channel, num_user, num_ant, num_rf, P_pow)
    rate_SDR[0, nu] = torch.mean(Rate)
    torch.cuda.synchronize()
    t1 = time.time()
    time_SDR[0, nu] = (t1-t0) / batch_size

time_MM = torch.zeros(1, num_dot).cuda()
rate_MM = torch.zeros(1, num_dot).cuda()
for nr in np.arange(0, num_dot):
    num_ant_per_chain = N_init + nu
    num_ant = num_rf * num_ant_per_chain
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    t4 = time.time()
    temp_channel = channel_generate(batch_size, num_user, num_ant, num_cl=5, num_ray=2).cuda()
    V_FD = FD_MMSE_1(temp_channel, P_pow)
    F_RF_4, F_BB_4, use_less_index_mm = MM_HBF(V_FD, num_rf)
    sum_rate_mm = Sum_rate(temp_channel, F_RF_4, F_BB_4)
    rate_MM[0, nr] = torch.mean(sum_rate_mm)
    torch.cuda.synchronize()
    t5 = time.time()
    time_MM[0, nr] = (t5-t4) / batch_size

time_YuWei = torch.zeros(1, num_dot).cuda()
rate_YuWei = torch.zeros(1, num_dot).cuda()
for nu in np.arange(0, num_dot):
    num_ant_per_chain = N_init + nu
    num_ant = num_rf * num_ant_per_chain
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    t4 = time.time()
    temp_channel = channel_generate(batch_size, num_user, num_ant, num_cl=5, num_ray=2).cuda()
    F, D, use_less_index_yw = YuWei_hybrid_precoding(temp_channel, P_pow, num_rf)
    sum_rate_yw = Sum_rate(temp_channel, F, D)
    rate_YuWei[0, nu] = torch.mean(sum_rate_yw)
    torch.cuda.synchronize()
    t5 = time.time()
    time_YuWei[0, nu] = (t5-t4) / batch_size

x = 0