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

batch_size = 1000
num_ant_per_chain = 4
# num_user, num_rf = 4, 8
# num_ant = num_ant_per_chain * num_rf
P_pow = 1

# temp_channel = channel_generate(batch_size, num_user, num_ant, num_cl=5, num_ray=2).cuda()
# # temp_channel = Rayleigh_channel_generate(batch_size, num_user, num_ant).cuda()
# temp_channel_float = torch.cat([temp_channel.real, temp_channel.imag], dim=2)
#
# t1 = time.time()
# F_target_temp, D_temp, iter_num = Wei_Yu_ZF_Precoding(temp_channel, P_pow, num_rf)
# rate_target = torch.mean(Sum_rate(temp_channel, F_target_temp, D_temp))
# t2 = time.time()
# t_ZF = (t2 - t1) / batch_size

#
# t3 = time.time()
# F_target_temp_1, D_temp_1 = Simple_ZF(temp_channel, P_pow, num_rf)
# rate_target_0 = torch.mean(Sum_rate(temp_channel, F_target_temp_1, D_temp_1))
# t4 = time.time()
# t_simple = (t4 - t3) / batch_size

# x=0

# rate_ZF = torch.zeros(1, 4).cuda()
# num_user_1, num_rf = 4, 8
# num_ant = num_rf * num_ant_per_chain
# for nu in np.arange(0, 4):
#     num_user = num_user_1 + nu
#     temp_channel = channel_generate(batch_size, num_user, num_ant, num_cl=5, num_ray=2).cuda()
#     V_FD = FD_ZF_1(temp_channel, P_pow)
#     rate_FD = FD_Sum_rate(temp_channel, V_FD)
#     rate_ZF[0, nu] = torch.mean(rate_FD)



# time_SDR = torch.zeros(1, 5).cuda()
# rate_SDR = torch.zeros(1, 5).cuda()
# # num_user, num_rf_1 = 4, 4
# num_user_1, num_rf = 3, 8
# for nu in np.arange(0, 5):
#     num_user = num_user_1 + nu
#     # num_rf = num_rf_1 + nu
#     num_ant = num_rf * num_ant_per_chain
#     torch.cuda.empty_cache()
#     torch.cuda.synchronize()
#     t0 = time.time()
#     temp_channel = channel_generate(batch_size, num_user, num_ant, num_cl=5, num_ray=2).cuda()
#     F_RF_1, F_BB_1, Rate, Useless_index = SDR_AltMin_Precoding(temp_channel, num_user, num_ant, num_rf, P_pow)
#     rate_SDR[0, nu] = torch.mean(Rate)
#     torch.cuda.synchronize()
#     t1 = time.time()
#     time_SDR[0, nu] = (t1-t0) / batch_size

# torch.cuda.empty_cache()
# # DFT_codebook = DFT_codebook_generate(num_ant).cuda()
# # Q_codebook = Q_resolution_codebook(num_ant, q=10).cuda()
# torch.cuda.synchronize()
# t2 = time.time()
# temp_channel, A_t = channel_generate(batch_size, num_user, num_ant, num_cl=5, num_ray=2)
# temp_channel = temp_channel.cuda()
# A_t = A_t.cuda()
# V_FD = FD_ZF_1(temp_channel, P_pow)
# # F_RF_2, F_BB_2 = SSP_Mul_Precoding(V_FD, DFT_codebook, num_rf)
# # F_RF_2, F_BB_2 = SSP_Mul_Precoding(V_FD, Q_codebook, num_rf)
# F_RF_2, F_BB_2 = SSP_Mul_Precoding(V_FD, A_t, num_rf)
# rate_SSP = torch.mean(Sum_rate(temp_channel, F_RF_2, F_BB_2))
# torch.cuda.synchronize()
# t3 = time.time()
# time_SSP = (t3-t2) / batch_size

# rate_FD = torch.mean(FD_Sum_rate(temp_channel, V_FD))

# time_MM = torch.zeros(1, 5).cuda()
# rate_MM = torch.zeros(1, 5).cuda()
# # num_user, num_rf_1 = 4, 4
# num_user_1, num_rf = 3, 8
# for nr in np.arange(0, 5):
#     num_user = num_user_1 + nr
#     # num_rf = num_rf_1 + nr
#     num_ant = num_rf * num_ant_per_chain
#     torch.cuda.empty_cache()
#     torch.cuda.synchronize()
#     t4 = time.time()
#     temp_channel = channel_generate(batch_size, num_user, num_ant, num_cl=5, num_ray=2).cuda()
#     # V_FD = FD_ZF_1(temp_channel, P_pow)
#     V_FD = FD_MMSE_1(temp_channel, P_pow)
#     F_RF_4, F_BB_4 = MM_HBF(V_FD, num_rf)
#     rate_MM[0, nr] = torch.mean(Sum_rate(temp_channel, F_RF_4, F_BB_4))
#     torch.cuda.synchronize()
#     t5 = time.time()
#     time_MM[0, nr] = (t5-t4) / batch_size

time_YuWei = torch.zeros(1, 5).cuda()
rate_YuWei = torch.zeros(1, 5).cuda()
num_user_1, num_rf = 6, 8
# num_user, num_rf_1 = 4, 5
for nu in np.arange(0, 2):
    num_user = num_user_1 + nu
    # num_rf = num_rf_1 + nu
    num_ant = num_rf * num_ant_per_chain
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    t4 = time.time()
    temp_channel = channel_generate(batch_size, num_user, num_ant, num_cl=5, num_ray=2).cuda()
    F, D, use_less_index = YuWei_hybrid_precoding(temp_channel, P_pow, num_rf)
    sum_rate_temp = Sum_rate(temp_channel, F, D)
    rate_YuWei[0, nu] = torch.mean(sum_rate_temp)
    torch.cuda.synchronize()
    t5 = time.time()
    time_YuWei[0, nu] = (t5-t4) / batch_size


# rate_FD = torch.mean(FD_Sum_rate(temp_channel, V_FD))

x = 0