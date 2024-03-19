import numpy as np
import random
import torch
import math
# from scipy.io import savemat
from BGNN_generate_channel import *
from BGNN_bentchmark import *
# import cvxpy as cp
# from cvxpylayers.torch import CvxpyLayer

############################################################################################Benchmark_YuWei_Precoding###########################################################################
def YuWei_hybrid_precoding(channel, P_pow, num_rf):

    num_batch, num_user, num_ant = channel.shape
    F = torch.zeros([num_batch, num_ant, num_rf], dtype=torch.complex128).cuda()
    D = torch.zeros([num_batch, num_rf, num_user], dtype=torch.complex128).cuda()
    for nb in np.arange(0, num_batch):
        print("channel num", nb)
        temp_channel = channel[nb, :, :]
        V_RF, V_D = YuWei_hybrid_precoding_Per_sample(temp_channel, P_pow, num_rf)
        F[nb, :, :] = V_RF
        D[nb, :, :] = V_D

    return F, D

def YuWei_hybrid_precoding_Per_sample(channel, P_pow, num_rf):

    num_user, num_ant = channel.shape
    num_ant_per_chain = int(num_ant / num_rf)

    ######initialize the P_matrix and V_RF
    P_matrix = torch.eye(num_user, dtype=torch.complex128).cuda()
    A = torch.zeros([num_ant_per_chain, 1], dtype=torch.complex128).cuda()
    for nr in np.arange(0, num_rf):
        temp_angle = 1 - 2 * torch.rand([num_ant_per_chain, 1]).cuda()
        A = torch.block_diag(A, torch.exp(1j * math.pi * temp_angle))
    V_RF = A[num_ant_per_chain:, 1:]

    y_0 = 10000
    P_inv_half = torch.inverse(P_matrix) ** (1/2)                                                                                      ####torch.sqrt()与**(1/2)都是对元素求根号，但是由于P_matrix始终是diagonal square matrix, 点乘与矩阵乘法相等
    H_hat = P_inv_half @ channel
    y_1 = trace_cal(H_hat, V_RF, num_ant_per_chain)

    while (torch.abs(y_0 - y_1) >= 0.01):

        y_0 = y_1
        V_RF = YuWei_analogue_precoding(channel, V_RF, P_matrix, num_rf)
        P_matrix = YuWei_digital_precoding(channel, V_RF, P_pow)
        P_inv_half = torch.inverse(P_matrix) ** (1/2)
        H_hat = P_inv_half @ channel
        y_1 = trace_cal(H_hat, V_RF, num_ant_per_chain)

    H_eff = channel @ V_RF
    V_D_normal = torch.conj(H_eff.T) @ torch.inverse(H_eff @ torch.conj(H_eff.T))
    V_D = V_D_normal @ torch.sqrt(P_matrix)

    return V_RF, V_D




def YuWei_analogue_precoding(channel, V_RF, P_matrix, num_rf):

    num_ant = channel.shape[1]
    num_ant_per_chain = int(num_ant/num_rf)

    # A = torch.zeros([num_ant_per_chain, 1], dtype=torch.complex128).cuda()
    # for nr in np.arange(0, num_rf):
    #     temp_angle = 1 - 2 * torch.rand([num_ant_per_chain, 1]).cuda()
    #     A = torch.block_diag(A, torch.exp(1j * math.pi * temp_angle))
    # V_RF = A[num_ant_per_chain:, 1:]

    y_0 = 10000
    P_inv_half = torch.inverse(P_matrix) ** (1/2)
    H_hat = P_inv_half @ channel
    y_1 = trace_cal(H_hat, V_RF, num_ant_per_chain)

    r = 0

    while (torch.abs(y_0 - y_1) >= 0.01):

        r = r + 1
        y_0 = y_1
        for nr in np.arange(0, num_rf):
            V_RF_j = V_RF[:, np.arange(0, num_rf) != nr]
            A_j = H_hat @ V_RF_j @ torch.conj(V_RF_j.T) @ torch.conj(H_hat.T)


            B_j = torch.conj(H_hat.T) @ torch.inverse(A_j) @ torch.inverse(A_j) @ H_hat
            D_j = torch.conj(H_hat.T) @ torch.inverse(A_j) @ H_hat

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

    return V_RF

def YuWei_digital_precoding(channel, V_RF, P_pow):

    H_eff = channel @ V_RF
    W = torch.conj(H_eff.T) @ torch.inverse(H_eff @ torch.conj(H_eff.T))

    P_matrix_half = Water_Power_allocation_per_sample(V_RF @ W, P_pow)
    P_matrix = P_matrix_half @ P_matrix_half

    return P_matrix

def trace_cal(H_hat, V_RF, num_ant_per_chain):

    H_eff = H_hat @ V_RF
    target = torch.inverse(H_eff @ torch.conj(H_eff.T))
    output = num_ant_per_chain * torch.trace(target).real

    return output