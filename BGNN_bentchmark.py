import numpy as np
import random
import torch
import math
from BGNN_global_value import *
# from scipy.io import savemat
# from BGNN_generate_channel import *

########################################################################################################################对比算法###########################################################
device_num = device_num_get()

def FD_Sum_rate(H, V):

    batch_size = H.shape[0]
    num_user = H.shape[1]
    Rate = torch.zeros([batch_size, 1], device=device_num)
    for nb in np.arange(0, batch_size):
        temp = torch.pow(torch.abs(H[nb, :, :] @ V[nb, :, :]), 2)
        signal_power = torch.diag(temp)
        signal_noise_power = torch.sum(temp, 1)
        for nu in np.arange(0, num_user):
            Rate[nb] = Rate[nb] + torch.log2(1 + (signal_power[nu] / (signal_noise_power[nu] - signal_power[nu] + 1)))

    return Rate

def Sum_rate(H, F, V):

    batch_size = H.shape[0]
    num_user = H.shape[1]
    Rate = torch.zeros([batch_size, 1], device=device_num)
    for nb in np.arange(0, batch_size):
        temp = torch.pow(torch.abs(H[nb, :, :] @ F[nb, :, :] @ V[nb, :, :]), 2)
        signal_power = torch.diag(temp)
        signal_noise_power = torch.sum(temp, 1)
        for nu in np.arange(0, num_user):
            Rate[nb] = Rate[nb] + torch.log2(1 + (signal_power[nu] / (signal_noise_power[nu] - signal_power[nu] + 1)))

    return Rate

def Water_Power_allocation(H, D, P_pow):                                        ######只适用与完美实现orthogonality的情况

    batch_size = H.shape[0]
    num_user = H.shape[1]
    # H_eff = torch.matmul(H, D)
    Pow_allo_matrix = torch.zeros([batch_size, num_user, num_user], dtype=torch.complex128, device=device_num)
    for nb in np.arange(0, batch_size):
        # temp_channel = H_eff[nb]
        ZF_matrix = D[nb, :, :]
        Pow_allo_matrix[nb, :, :] = Water_Power_allocation_per_sample(ZF_matrix, P_pow)

    return Pow_allo_matrix

def Water_Power_allocation_per_sample(F_hat, P_pow):                            #F_hat为ZF matrix, 部分连接结构时为digital beamforming matrix

    num_user = F_hat.shape[1]
    Q = torch.conj(torch.transpose(F_hat, 0, 1)) @ F_hat
    D_power = torch.zeros([1, num_user], device=device_num)
    for nu in np.arange(0, num_user):
        D_power[0, nu] = Q[nu, nu].real
    # torch.sort 默认从小到大排序
    hight, hight_index = torch.sort(D_power)
    hight_inver = 1 / hight

    for nu in np.arange(1, num_user):                                           #遍历找出max_index, user数为max_index + 1
        water_level = hight[0, nu]
        P_test = ((water_level - hight) > 0) * (water_level - hight)
        test_power = torch.sum(P_test)
        max_index = nu
        if (test_power - P_pow) > 0:
            max_index = max_index - 1
            break

    temp_level = hight[0, max_index]
    a = ((temp_level - hight) > 0) * (temp_level - hight)
    temp_power = torch.sum(a)
    final_level = temp_level + (P_pow - temp_power) / (max_index + 1)
    final_power = ((final_level - hight) > 0) * (final_level - hight) * hight_inver

    Pow_matrix = torch.zeros([num_user, num_user], dtype=torch.complex128)
    for nu in np.arange(0, num_user):
        user_index = int(hight_index[0, nu])
        temp_power_1 = final_power[0, nu]
        Pow_matrix[user_index, user_index] = torch.sqrt(temp_power_1)

    # Pow_matrix = Pow_matrix.cuda()
    Pow_matrix = Pow_matrix.to(device=device_num)

    return Pow_matrix

def FD_MMSE_1(H, P_pow):

    batch_size, num_user, num_ant = H.shape

    H_1 = torch.conj(torch.transpose(H, 1, 2))
    H_2 = H @ H_1 + (num_user / P_pow) * torch.eye(num_user, device=device_num)
    H_inv = torch.inverse(H_2)
    W = torch.matmul(H_1, H_inv)                                                        # W为ZF beamforming matrix
    Power_allocation_matrix = Water_Power_allocation(H, W, P_pow)
    D = W @ Power_allocation_matrix

    return D

def FD_MRT_1(H, P_pow):

    batch_size, num_user, num_ant = H.shape

    H_1 = torch.conj(torch.transpose(H, 1, 2))
    W = H_1
    Power_allocation_matrix = Water_Power_allocation(H, W, P_pow)                   #假设完美orthogonality能够被实现
    D = W @ Power_allocation_matrix


    return D

def FD_ZF_1(H, P_pow):

    batch_size, num_user, num_ant = H.shape

    H_1 = torch.conj(torch.transpose(H, 1, 2))
    H_inv = torch.inverse(torch.matmul(H, H_1))
    W = torch.matmul(H_1, H_inv)                                                        # W为ZF beamforming matrix
    Power_allocation_matrix = Water_Power_allocation(H, W, P_pow)
    D = W @ Power_allocation_matrix
    # W_norm = torch.norm(W, 'fro', 1, keepdim=True)
    # D = torch.zeros([batch_size, num_ant, num_user], dtype=torch.complex128, device=device_num)
    # for nb in np.arange(0, batch_size):
    #     for nu in np.arange(0, num_user):
    #         # profit_1 = Power_allocation_matrix[nb, nu, nu]
    #         # profit = profit_1 / W_norm[nb, :, nu]
    #         profit = Power_allocation_matrix[nb, nu, nu]
    #         D[nb, :, nu] = profit * W[nb, :, nu]

    return D

def PC_EGT_ZF_precoding(H, P):                                                                                          # num_rf is defaulted to be the same as num_user

    batch_size, num_user, num_ant = H.shape
    num_ant_per_chain = int(num_ant / num_user)

    F = torch.zeros([batch_size, num_ant, num_user], dtype=torch.complex128, device=device_num)
    H_1 = torch.conj(torch.transpose(H, 1, 2))
    for nr in np.arange(0, num_user):
        phase_temp = torch.angle(H_1[:, nr * num_ant_per_chain : nr * num_ant_per_chain + num_ant_per_chain, nr])
        F[:, nr*num_ant_per_chain : nr*num_ant_per_chain+num_ant_per_chain, nr] = torch.exp(1j * phase_temp)
    # H_eff = torch.matmul(H, F)                                                                                        #下面这两种求digital precoding的方法都是正确的， 第一种相当于把HF作为等效信道，第二种把FD作为FD的beamforming matrix, 第二种没有用F^H * F = M * I的性质
    # D = FD_ZF_1(H_eff, P/num_ant_per_chain)

    H_eff = H @ F
    W = torch.conj(torch.transpose(H_eff, 1, 2)) @ torch.inverse(H_eff @ torch.conj(torch.transpose(H_eff, 1, 2)))
    P_matrix = Water_Power_allocation(H, F @ W, P)
    D = W @ P_matrix


    return F, D


def PC_Random_Phase_ZF_precoding(H, P, num_rf):  # num_rf is defaulted to be the same as num_user

    batch_size, num_user, num_ant = H.shape
    num_ant_per_chain = int(num_ant / num_rf)
    F = torch.zeros([batch_size, num_ant, num_rf], dtype=torch.complex128, device=device_num)
    for nr in np.arange(0, num_rf):
        phase_temp = (torch.rand([batch_size, num_ant_per_chain], device=device_num) * 2 - 1) * math.pi
        F[:, nr * num_ant_per_chain: nr * num_ant_per_chain + num_ant_per_chain, nr] = torch.exp(1j * phase_temp)
    H_eff = torch.matmul(H, F)
    D = FD_ZF_1(H_eff, P / num_ant_per_chain)

    return F, D

def PC_EGT_ZF_analytical(num_user, num_ant, P):

    Rate = num_user * math.log2(1 + (math.pi * P * num_ant) / (4 * num_user * num_user))

    return Rate

def EGT_precoding(H):

    batch_size, num_user, num_ant = H.shape
    num_ant_per_chain = int(num_ant / num_user)

    F = torch.zeros([batch_size, num_ant, num_user], dtype=torch.complex128, device=device_num)
    H_1 = torch.conj(torch.transpose(H, 1, 2))
    for nr in np.arange(0, num_user):
        phase_temp = torch.angle(H_1[:, nr * num_ant_per_chain: nr * num_ant_per_chain + num_ant_per_chain, nr])
        F[:, nr * num_ant_per_chain: nr * num_ant_per_chain + num_ant_per_chain, nr] = torch.exp(1j * phase_temp)

    return F

#################################################################################################################################################################YuWei Analogue Precoding###############################################################
################################################################################################################################################################################################################################################

def Wei_Yu_ZF_Precoding(H, P_pow, num_rf):

    batch_size, num_user, num_ant = H.shape
    num_ant_per_chain = num_ant / num_rf
    F, iter_num = Wei_Yu_Analogue_Precoding(H, num_rf, P_pow)
    H_eff = H @ F
    D = FD_ZF_1(H_eff, P_pow / num_ant_per_chain)

    return F, D, iter_num

def Wei_Yu_MMSE_Precoding(H, P_pow, num_rf):

    batch_size, num_user, num_ant = H.shape
    num_ant_per_chain = num_ant / num_rf
    F, iter_num = Wei_Yu_Analogue_Precoding(H, num_rf, P_pow)
    H_eff = H @ F
    D = FD_MMSE_1(H_eff, P_pow / num_ant_per_chain)

    return F, D, iter_num

def Wei_Yu_Analogue_Precoding(H, num_rf, P_pow):

    batch_size, num_user, num_ant = H.shape
    F = torch.zeros([batch_size, num_ant, num_rf], dtype=torch.complex128, device=device_num)
    iter_num = torch.zeros([batch_size, 1])

    for nb in np.arange(0, batch_size):
        temp_channel = H[nb, :, :]
        F[nb, :, :], iter_num[nb, :] = Wei_Yu_Analogue_Precoding_Persample(temp_channel, num_rf, P_pow)

    return F, iter_num

def Wei_Yu_Analogue_Precoding_Persample(H, num_rf, P_pow):

    num_user, num_ant = H.shape
    F_1 = (P_pow/num_ant) * torch.conj(torch.transpose(H, 0, 1)) @ H
    # F_1 = torch.conj(torch.transpose(H, 0, 1)) @ H
    num_ant_per_chain = int(num_ant/num_rf)

    ####initialize###########
    F_update = torch.zeros([num_ant, num_rf], dtype=torch.complex128, device=device_num)
    for nr in  np.arange(0, num_rf):
        F_update[np.arange(num_ant_per_chain * nr, num_ant_per_chain*(nr + 1)), nr] = torch.ones(num_ant_per_chain, dtype=torch.complex128, device=device_num)
    capacity_before = 0.0
    A_0 = torch.eye(num_rf, dtype=torch.complex128, device=device_num) + torch.conj(torch.transpose(F_update, 0, 1)) @ F_1 @ F_update
    capacity_after = torch.linalg.det(A_0).real

    iter_num = 0
    ratio = 1

    # while (capacity_after / capacity_before) >= 2 ** (0.01):
    while(ratio>0.01):
        Output_F = F_update
        capacity_before = capacity_after
        for nr in np.arange(0, num_rf):
            temp_F = F_update[:, np.arange(0, num_rf) != nr]
            C_j = torch.eye(num_rf-1, dtype=torch.complex128, device=device_num) + torch.conj(torch.transpose(temp_F, 0, 1)) @ F_1 @ temp_F
            G_j = F_1 - F_1 @ temp_F @ torch.inverse(C_j) @ torch.conj(torch.transpose(temp_F, 0, 1)) @ F_1
            for na in np.arange(num_ant_per_chain * nr, num_ant_per_chain * (nr+1)):
                temp_mu = G_j[na, :] @ F_update[:, nr] - G_j[na, na] * F_update[na, nr]
                if temp_mu == 0:
                    F_update[na, nr] = 1.0 + 1j * 0.0
                else:
                    F_update[na, nr] = temp_mu / abs(temp_mu)
        iter_num = iter_num + 1
        A = torch.eye(num_rf, dtype=torch.complex128, device=device_num) + torch.conj(torch.transpose(F_update, 0, 1)) @ F_1 @ F_update
        capacity_after = torch.linalg.det(A).real
        ratio = torch.abs(capacity_after - capacity_before) / capacity_after
        # print('capacity_after', capacity_after)
        # a = capacity_after / capacity_before

    return Output_F, iter_num

#################################################################################################################################################################SSP Precoding###############################################################
################################################################################################################################################################################################################################################

def SSP_Mul_Precoding(Fopt, Codebook, num_rf):

    batch_size, num_ant, num_user = Fopt.shape
    F_RF = torch.zeros([batch_size, num_ant, num_rf], dtype=torch.complex128, device=device_num)
    F_BB = torch.zeros([batch_size, num_rf, num_user], dtype=torch.complex128, device=device_num)
    Index = torch.zeros([batch_size, num_rf], dtype=torch.int64, device=device_num)
    for nb in np.arange(0, batch_size):
        # if (nb % 1000 == 0):
        #     print("channel num", nb)
        F_opt = Fopt[nb, :, :]
        FRF, FBB, index = SSP_precoding(F_opt, Codebook, num_rf)
        # FRF, FBB = SSP_precoding(F_opt, Codebook[nb, :, :], num_rf)
        F_RF[nb, :, :] = FRF
        F_BB[nb, :, :] = FBB
        Index[nb, :] = index

    return F_RF, F_BB, Index

def SSP_precoding(Fopt, Codebook, num_rf):

    num_ant, num_user = Fopt.shape
    codebook_size = Codebook.shape[1]
    num_ant_per_chain = int(num_ant/num_rf)
    F_RF = torch.zeros([num_ant, 1], dtype=torch.complex128, device=device_num)
    F_res = Fopt
    P_pow = torch.norm(Fopt, 'fro')**2
    index = torch.zeros([1, num_rf], dtype=torch.int64).to(device = device_num)

    for nr in np.arange(0, num_rf):
        # Q = torch.zeros([num_ant, codebook_size], device=device_num)
        # Q[np.arange(num_ant_per_chain * nr, num_ant_per_chain * (nr+1)), :] = torch.ones([num_ant_per_chain, codebook_size], device=device_num)
        # Codebook_1 = Codebook * Q
        Codebook_1 = torch.zeros([num_ant, codebook_size], dtype=torch.complex128, device=device_num)
        Codebook_1[np.arange(num_ant_per_chain * nr, num_ant_per_chain * (nr+1)), :] = Codebook
        # Codebook_1 = Codebook * Q
        Phi = torch.conj(torch.transpose(Codebook_1, 0, 1)) @ F_res
        gain_vector = torch.diag(Phi @ torch.conj(torch.transpose(Phi, 0, 1))).real
        num = torch.max(gain_vector, dim=0)[1]
        index[0, nr] = num
        if (nr == 0):
            F_RF = Codebook_1[:, num].unsqueeze(dim=1)
        else:
            F_RF = torch.cat([F_RF, Codebook_1[:, num].unsqueeze(dim=1)], dim=1)
        F_RF_H = torch.conj(torch.transpose(F_RF, 0, 1))
        F_BB = torch.inverse(F_RF_H @ F_RF) @ F_RF_H @ Fopt
        F_res_unnorm = Fopt - F_RF @ F_BB
        F_res = F_res_unnorm / torch.norm(F_res_unnorm, 'fro')

    F_BB = (torch.sqrt(P_pow) / torch.norm(F_RF @ F_BB, 'fro')) * F_BB

    return F_RF, F_BB, index


#################################################################################################################################################################Yu Wei Precoding###############################################################
################################################################################################################################################################################################################################################

def YuWei_hybrid_precoding(channel, P_pow, num_rf):

    num_batch, num_user, num_ant = channel.shape
    F = torch.zeros([num_batch, num_ant, num_rf], dtype=torch.complex128, device=device_num)
    D = torch.zeros([num_batch, num_rf, num_user], dtype=torch.complex128, device=device_num)
    use_less_index = []

    for nb in np.arange(0, num_batch):
        if (nb % 1000 == 0):
            print("channel num", nb)
        temp_channel = channel[nb, :, :]
        V_RF, V_D, r = YuWei_hybrid_precoding_Per_sample(temp_channel, P_pow, num_rf)
        if (r>50):
            use_less_index.append(nb)
        F[nb, :, :] = V_RF
        D[nb, :, :] = V_D

    use_less_index = np.asarray(use_less_index, dtype=int)

    return F, D, use_less_index

def YuWei_hybrid_precoding_Per_sample(channel, P_pow, num_rf):

    num_user, num_ant = channel.shape
    num_ant_per_chain = int(num_ant / num_rf)

    ######initialize the P_matrix and V_RF
    P_matrix = torch.eye(num_user, dtype=torch.complex128, device=device_num)
    A = torch.zeros([num_ant_per_chain, 1], dtype=torch.complex128, device=device_num)
    for nr in np.arange(0, num_rf):
        temp_angle = 1 - 2 * torch.rand([num_ant_per_chain, 1], device=device_num)
        A = torch.block_diag(A, torch.exp(1j * math.pi * temp_angle))
    V_RF = A[num_ant_per_chain:, 1:]

    y_0 = 10000
    P_inv_half = torch.pinverse(P_matrix) ** (1/2)                                                                                      ####torch.sqrt()与**(1/2)都是对元素求根号，但是由于P_matrix始终是diagonal square matrix, 点乘与矩阵乘法相等
    H_hat = P_inv_half @ channel
    y_1 = trace_cal(H_hat, V_RF, num_ant_per_chain)
    ratio = 1

    r = 0
    # while ((torch.abs(y_0 - y_1) >= 0.01) & (r <= 100)):
    while ((ratio > 0.01) & (r <= 50)):

        r = r + 1
        y_0 = y_1

        V_RF = YuWei_analogue_precoding(channel, V_RF, P_matrix, num_rf)
        P_matrix = YuWei_digital_precoding(channel, V_RF, P_pow)
        P_inv_half = torch.pinverse(P_matrix) ** (1/2)
        H_hat = P_inv_half @ channel

        y_1 = trace_cal(H_hat, V_RF, num_ant_per_chain)
        ratio = torch.abs(y_0 - y_1) / y_0

    H_eff = channel @ V_RF
    V_D_normal = torch.conj(H_eff.T) @ torch.pinverse(H_eff @ torch.conj(H_eff.T))
    V_D = V_D_normal @ torch.sqrt(P_matrix)

    return V_RF, V_D, r

def YuWei_analogue_precoding(channel, V_RF, P_matrix, num_rf):

    num_ant = channel.shape[1]
    num_ant_per_chain = int(num_ant/num_rf)

    # A = torch.zeros([num_ant_per_chain, 1], dtype=torch.complex128, device=device_num)
    # for nr in np.arange(0, num_rf):
    #     temp_angle = 1 - 2 * torch.rand([num_ant_per_chain, 1], device=device_num)
    #     A = torch.block_diag(A, torch.exp(1j * math.pi * temp_angle))
    # V_RF = A[num_ant_per_chain:, 1:]

    y_0 = 10000
    P_inv_half = torch.pinverse(P_matrix) ** (1/2)
    H_hat = P_inv_half @ channel
    y_1 = trace_cal(H_hat, V_RF, num_ant_per_chain)

    r = 0
    ratio = 1

    # while (torch.abs(y_0 - y_1) >= 0.01):
    while ((ratio > 0.01) & (r <= 100)):

        r = r + 1
        y_0 = y_1

        for nr in np.arange(0, num_rf):
            V_RF_j = V_RF[:, np.arange(0, num_rf) != nr]
            A_j = H_hat @ V_RF_j @ torch.conj(V_RF_j.T) @ torch.conj(H_hat.T)


            B_j = torch.conj(H_hat.T) @ torch.pinverse(A_j) @ torch.pinverse(A_j) @ H_hat
            D_j = torch.conj(H_hat.T) @ torch.pinverse(A_j) @ H_hat

            for nt in np.arange(num_ant_per_chain * nr, num_ant_per_chain * (nr + 1)):

                B_j_nt_0 = B_j[:, np.arange(0, num_ant) != nt]
                B_j_nt = B_j_nt_0[np.arange(0, num_ant) != nt, :]
                V_rf_nr_nt = V_RF[np.arange(0, num_ant) != nt, nr]
                Complex_num_B = torch.conj(V_rf_nr_nt.T) @ B_j_nt @ V_rf_nr_nt
                ita_ij_B = B_j[nt, nt].real + 2 * Complex_num_B.real

                D_j_nt_0 = D_j[:, np.arange(0, num_ant) != nt]
                D_j_nt = D_j_nt_0[np.arange(0, num_ant) != nt, :]
                Complex_num_D = torch.conj(V_rf_nr_nt.T) @ D_j_nt @ V_rf_nr_nt
                ita_ij_D = D_j[nt, nt].real + 2 * Complex_num_D.real

                mu_ij_B = B_j[nt, :] @ V_RF[:, nr] - B_j[nt, nt].real * V_RF[nt, nr]
                mu_ij_D = D_j[nt, :] @ V_RF[:, nr] - D_j[nt, nt].real * V_RF[nt, nr]

                c_ij = (1 + ita_ij_D) * mu_ij_B - ita_ij_B * mu_ij_D
                z_ij_complex = 2 * torch.conj(mu_ij_B) * mu_ij_D
                z_ij = z_ij_complex.imag

                if (c_ij.real >= 0):
                    phi_ij = torch.arcsin(c_ij.imag / torch.abs(c_ij))
                else:
                    phi_ij = math.pi - torch.arcsin(c_ij.imag / torch.abs(c_ij))

                theta_ij_1 = -phi_ij + torch.arcsin(z_ij / torch.abs(c_ij))
                theta_ij_2 = math.pi - phi_ij - torch.arcsin(z_ij / torch.abs(c_ij))

                # complex_theta_1_B = torch.exp(-1 * 1j * theta_ij_1) * mu_ij_B
                # complex_theta_1_D = torch.exp(-1 * 1j * theta_ij_1) * mu_ij_D
                # f_V_RF_theta_1 = num_ant_per_chain * (torch.trace(torch.inverse(A_j)).real - (ita_ij_B + 2 * complex_theta_1_B.real) / (1 + ita_ij_D + 2 * complex_theta_1_D.real))
                #
                # complex_theta_2_B = torch.exp(-1 * 1j * theta_ij_2) * mu_ij_B
                # complex_theta_2_D = torch.exp(-1 * 1j * theta_ij_2) * mu_ij_D
                # f_V_RF_theta_2 = num_ant_per_chain * (torch.trace(torch.inverse(A_j)).real - (ita_ij_B + 2 * complex_theta_2_B.real) / (1 + ita_ij_D + 2 * complex_theta_2_D.real))
                #
                # if (f_V_RF_theta_1 < f_V_RF_theta_2):
                #     V_RF[nt, nr] = torch.exp(-1 * 1j * theta_ij_1)
                # else:
                #     V_RF[nt, nr] = torch.exp(-1 * 1j * theta_ij_2)

                V_RF_1 = V_RF
                V_RF_1[nt, nr] = torch.exp(-1 * 1j * theta_ij_1)
                f_V_RF_1 = trace_cal(H_hat, V_RF_1, num_ant_per_chain)

                V_RF_2 = V_RF
                V_RF_2[nt, nr] = torch.exp(-1 * 1j * theta_ij_2)
                f_V_RF_2 = trace_cal(H_hat, V_RF_2, num_ant_per_chain)

                if (f_V_RF_1 < f_V_RF_2):
                    V_RF[nt, nr] = torch.exp(-1 * 1j * theta_ij_1)
                else:
                    V_RF[nt, nr] = torch.exp(-1 * 1j * theta_ij_2)

        y_1 = trace_cal(H_hat, V_RF, num_ant_per_chain)
        ratio = torch.abs(y_0 - y_1) / y_0

    return V_RF

def YuWei_digital_precoding(channel, V_RF, P_pow):

    H_eff = channel @ V_RF
    W = torch.conj(H_eff.T) @ torch.pinverse(H_eff @ torch.conj(H_eff.T))

    P_matrix_half = Water_Power_allocation_per_sample(V_RF @ W, P_pow)
    P_matrix = P_matrix_half @ P_matrix_half

    return P_matrix

def trace_cal(H_hat, V_RF, num_ant_per_chain):

    H_eff = H_hat @ V_RF
    target = torch.pinverse(H_eff @ torch.conj(H_eff.T))
    output = num_ant_per_chain * torch.trace(target).real

    return output

#######################################################################################################################MM-based HBF##########################################################

def MM_HBF(F_opt, num_rf):

    batch_size, num_ant, num_user = F_opt.shape
    F_BB = torch.zeros([batch_size, num_rf, num_user], dtype=torch.complex128, device=device_num)
    F_RF = torch.zeros([batch_size, num_ant, num_rf], dtype=torch.complex128, device=device_num)
    use_less_index = []

    for nb in np.arange(0, batch_size):
        if (nb % 1000 == 0):
            print("channel num", nb)
        F_opt_nb = F_opt[nb, :, :]
        F_RF[nb, :, :], F_BB[nb, :, :], r = MM_HBF_per_sample(F_opt_nb, num_rf)
        if (r>50):
            use_less_index.append(nb)

    use_less_index = np.asarray(use_less_index, dtype=int)

    return  F_RF, F_BB, use_less_index

def MM_HBF_per_sample(F_opt, num_rf):

    num_ant, num_user = F_opt.shape
    P_pow = torch.norm(F_opt, 'fro')**2
    num_ant_per_chain = int(num_ant/num_rf)
    F_mask = torch.zeros([num_ant, num_rf], dtype=torch.complex128, device=device_num)
    for nr in np.arange(0, num_rf):
        a = torch.ones([num_ant_per_chain, 1], dtype=torch.complex128, device=device_num)
        F_mask[np.arange(num_ant_per_chain*nr, num_ant_per_chain*(nr+1)), nr] = a.squeeze(dim=1)
    F_RF = F_mask
    F_BB = torch.rand([num_rf, num_user], dtype=torch.complex128, device=device_num)
    F_BB = (math.sqrt(P_pow) / torch.norm(F_RF @ F_BB, 'fro')) * F_BB

    y_0 = 1000
    y_1 = torch.norm(F_opt - F_RF @ F_BB, 'fro')**2
    ratio = 1
    r = 0
    # while (abs(y_0 - y_1) > 0.01):
    # while (ratio > 0.01):
    while ((ratio > 0.01) & (r <= 50)):

        r = r+1
        y_0 = y_1

        F_RF_pinv = torch.inverse(torch.conj(torch.transpose(F_RF, 1, 0)) @ F_RF) @ torch.conj(torch.transpose(F_RF, 1, 0))
        F_BB = F_RF_pinv @ F_opt
        F_BB = (math.sqrt(P_pow) / torch.norm(F_RF @ F_BB, 'fro')) * F_BB

        y_0_in = 1000
        y_1_in = torch.norm(F_opt - F_RF @ F_BB, 'fro')**2
        ratio_in = 1

        # while (abs(y_0_in - y_1_in) > 0.01):
        while (ratio_in > 0.01):

            y_0_in = y_1_in

            target = torch.zeros([num_rf * num_ant, 1], dtype=torch.complex128, device=device_num)
            f_RF = torch.reshape(F_RF.T, (-1, 1))

            Q = torch.zeros(num_rf, num_rf, device=device_num)
            C = torch.zeros(num_ant, num_rf, device=device_num)
            for nu in np.arange(0, num_user):
                f_BB_k = F_BB[:, nu].unsqueeze(dim=1)
                Q = Q + torch.conj(f_BB_k) @ f_BB_k.T
                C = C + F_opt[:, nu].unsqueeze(dim=1) @ torch.conj(f_BB_k.T)
            P = torch.kron(Q, torch.eye(num_ant, device=device_num))
            Z = torch.reshape(C.T, (-1, 1))
            Eig = torch.linalg.eigvals(P)
            lamda_max = max(torch.abs(Eig))
            V = (P - lamda_max * torch.eye(num_ant * num_rf, device=device_num)) @ f_RF - Z
            f_RF_1 = -torch.exp(1j * torch.angle(V))
            F_RF_2 = torch.reshape(f_RF_1, (num_rf, num_ant))


            # for nu in np.arange(0, num_user):
            #     F_BB_k = F_BB[:, nu].unsqueeze(dim=1)
            #     Q_k = torch.kron(torch.conj(F_BB_k) @ F_BB_k.T, torch.eye(num_ant, device=device_num))
            #     Eig = torch.linalg.eigvals(Q_k)
            #     lamda_k = max(torch.abs(Eig))
            #     E_k = F_opt[:, nu].unsqueeze(dim=1) @ torch.conj(F_BB_k.T)
            #     e_k = torch.reshape(E_k.T, (-1, 1))
            #     target = target + (1/num_user) * ((Q_k - lamda_k * torch.eye(num_rf * num_ant, dtype=torch.complex128, device=device_num)) @ f_RF - e_k )
            # f_RF_1 = -torch.exp(1j * torch.angle(target))
            # F_RF_2 = torch.reshape(f_RF_1, (num_rf, num_ant))

            F_RF = F_RF_2.T * F_mask
            # F_BB = (math.sqrt(P_pow) / torch.norm(F_RF @ F_BB, 'fro')) * F_BB
            y_1_in = torch.norm(F_opt - F_RF @ F_BB, 'fro')**2

            ratio_in = torch.abs(y_0_in - y_1_in) / y_0_in

        y_1 = y_1_in
        ratio = torch.abs(y_0 - y_1) / y_0

    F_BB = (math.sqrt(P_pow) / torch.norm(F_RF @ F_BB, 'fro')) * F_BB

    return F_RF, F_BB, r


#############################################################################Simple Initialization################################################################
##################################################################################################################################################################

def Simple_ZF(H, P_pow, num_rf):
    batch_size, num_user, num_ant = H.shape
    num_ant_per_chain = num_ant / num_rf
    H_float = torch.cat([H.real, H.imag], dim=2)
    F = F_init_generate(H_float, num_rf)
    H_eff = H @ F
    D = FD_ZF_1(H_eff, P_pow / num_ant_per_chain)

    return F, D

def F_init_generate(H, num_rf):
    batch_size, num_user, num_ant_2 = H.shape
    num_ant = int(num_ant_2 / 2)
    num_ant_per_chain = int(num_ant / num_rf)

    F_init = torch.zeros([batch_size, num_ant, num_rf], dtype=torch.complex128).to(device=device_num)
    H_complex = H[:, :, 0:num_ant] + 1j * H[:, :, num_ant:num_ant_2]
    H_equal = torch.sum(H_complex, dim=1)
    for nr in np.arange(0, num_rf):
        # h_k = H_equal[:, np.arange(num_ant_per_chain*nr, num_ant_per_chain*(nr+1))]
        h_k = torch.conj(H_equal[:, np.arange(num_ant_per_chain*nr, num_ant_per_chain*(nr+1))])
        F_init[:, num_ant_per_chain * nr : num_ant_per_chain * (nr + 1), nr] = h_k / torch.abs(h_k)

    return F_init

# def F_init_generate_1(H, num_rf):           #求F_norm max的channel
#     batch_size, num_user, num_ant_2 = H.shape
#     num_ant = int(num_ant_2 / 2)
#     num_ant_per_chain = int(num_ant / num_rf)
#
#     F_init = torch.zeros([batch_size, num_ant, num_rf], dtype=torch.complex128).to(device=device_num)
#     H_complex = H[:, :, 0:num_ant] + 1j * H[:, :, num_ant:num_ant_2]
#     for nr in np.arange(0, num_rf):
#         H_complex_partial = H_complex[:, :, num_ant_per_chain*nr : num_ant_per_chain*(nr+1)]
#         H_complex_partial_norm = torch.norm(H_complex_partial, 'fro', dim=2)**2
#         h_index = torch.max(H_complex_partial_norm, dim=1)[1]
#         h_k = torch.zeros([batch_size, num_ant_per_chain], dtype=torch.complex128, device=device_num)
#         for nb in np.arange(0, batch_size):
#             h_k[nb, :] = H_complex_partial[nb, h_index[nb], :]
#         # h_k = H_complex_partial[:, h_index, :]
#         F_init[:, num_ant_per_chain * nr: num_ant_per_chain * (nr + 1), nr] = h_k / torch.abs(h_k)
#
#     return F_init


