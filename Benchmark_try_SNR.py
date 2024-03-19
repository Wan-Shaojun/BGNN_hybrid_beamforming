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

batch_size = 10000
num_ant_per_chain = 4
num_user, num_rf = 4, 4
num_ant = num_ant_per_chain * num_rf

device_num = device_num_get()
temp_channel = Rayleigh_channel_generate(batch_size, num_user, num_ant).to(device = device_num)

SNR_low = 30
SNR_gap = 10
num_dot = 5

rate_RZF = torch.zeros(1, num_dot)
rate_ZF = torch.zeros(1, num_dot)
rate_MRT = torch.zeros(1, num_dot)

for nu in np.arange(0, num_dot):
    snr = SNR_low + SNR_gap * nu
    P_pow = 10 ** (snr / 10)
    D_rzf = FD_MMSE_1(temp_channel, P_pow)
    rate_RZF[0, nu] = torch.mean(FD_Sum_rate(temp_channel, D_rzf))
    Gian_rzf = torch.mean(torch.abs(temp_channel @ D_rzf), dim=0)
    D_zf = FD_ZF_1(temp_channel, P_pow)
    rate_ZF[0, nu] = torch.mean(FD_Sum_rate(temp_channel, D_zf))
    Gian_zf = torch.mean(torch.abs(temp_channel @ D_zf), dim=0)
    # D_mrt = FD_MRT_1(temp_channel, P_pow)
    # rate_MRT[0, nu] = torch.mean(FD_Sum_rate(temp_channel, D_mrt))
    # Gian_mrt = torch.mean(torch.abs(temp_channel @ D_mrt), dim=0)
    x = 0

# time_SDR = torch.zeros(1, num_dot).cuda()
# rate_SDR = torch.zeros(1, num_dot).cuda()
# for nu in np.arange(0, num_dot):
#     snr = SNR_low + SNR_gap * nu
#     P_pow = 10**(snr/10)
#     torch.cuda.empty_cache()
#     torch.cuda.synchronize()
#     t0 = time.time()
#     F_RF_1, F_BB_1, Rate, Useless_index_sdr = SDR_AltMin_Precoding(temp_channel, num_user, num_ant, num_rf, P_pow)
#     rate_SDR[0, nu] = torch.mean(Rate)
#     torch.cuda.synchronize()
#     t1 = time.time()
#     time_SDR[0, nu] = (t1-t0) / batch_size
#
# time_MM = torch.zeros(1, num_dot).cuda()
# rate_MM = torch.zeros(1, num_dot).cuda()
# for nu in np.arange(0, num_dot):
#     snr = SNR_low + SNR_gap * nu
#     P_pow = 10**(snr/10)
#     torch.cuda.empty_cache()
#     torch.cuda.synchronize()
#     t4 = time.time()
#     V_FD = FD_MMSE_1(temp_channel, P_pow)
#     F_RF_4, F_BB_4, use_less_index_mm = MM_HBF(V_FD, num_rf)
#     sum_rate_MM = Sum_rate(temp_channel, F_RF_4, F_BB_4)
#     rate_MM[0, nu] = torch.mean(sum_rate_MM)
#     torch.cuda.synchronize()
#     t5 = time.time()
#     time_MM[0, nu] = (t5-t4) / batch_size
#
# time_YuWei = torch.zeros(1, num_dot).cuda()
# rate_YuWei = torch.zeros(1, num_dot).cuda()
# for nu in np.arange(0, num_dot):
#     snr = SNR_low + SNR_gap * nu
#     P_pow = 10**(snr/10)
#     torch.cuda.empty_cache()
#     torch.cuda.synchronize()
#     t4 = time.time()
#     F, D, use_less_index_yw = YuWei_hybrid_precoding(temp_channel, P_pow, num_rf)
#     sum_rate_YuWei = Sum_rate(temp_channel, F, D)
#     rate_YuWei[0, nu] = torch.mean(sum_rate_YuWei)
#     torch.cuda.synchronize()
#     t5 = time.time()
#     time_YuWei[0, nu] = (t5-t4) / batch_size

x = 0