import numpy as np
import random
import torch
import math
from BGNN_global_value import *
# from scipy.io import savemat
# from BGNN_generate_channel import *
from BGNN_bentchmark import *
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

#################################################################################################################################################################SDR Precoding###############################################################
################################################################################################################################################################################################################################################

def SDR_AltMin_Precoding(channel, num_user, num_ant, num_rf, P_pow):

    # channel = channel_generate(num_channel, num_user, num_ant, num_cl=5, num_ray=2)
    num_channel = channel.shape[0]
    # F_opt = FD_ZF_1(channel, P_pow).cpu()
    F_opt = FD_MMSE_1(channel, P_pow).cpu()
    F_RF = torch.zeros([num_channel, num_ant, num_rf], dtype=torch.complex128, device=device_num)
    F_BB = torch.zeros([num_channel, num_rf, num_user], dtype=torch.complex128, device=device_num)
    use_less_index = []

    for nc in np.arange(0, num_channel):
        if (nc % 1000 == 0):
            print("channel num", nc)
        Fopt = F_opt[nc, :, :]
        FRF, FBB, r = SDR_AltMin(Fopt, num_rf)
        if (r>50):
            use_less_index.append(nc)
        F_RF[nc, :, :] = FRF.to(device=device_num)
        F_BB[nc, :, :] = FBB.to(device=device_num)

    Rate = Sum_rate(channel, F_RF, F_BB)
    use_less_index = np.asarray(use_less_index, dtype=int)

    return F_RF, F_BB, Rate, use_less_index

def SDR_AltMin(F_opt, num_rf):

    num_ant, num_user = F_opt.shape
    num_ant_per_chain = int(num_ant/num_rf)
    P_pow = torch.norm(F_opt, 'fro')**2

    A = torch.zeros([num_ant_per_chain, 1], dtype=torch.complex128)
    for nr in np.arange(0, num_rf):
        temp_angle = 1 - 2 * torch.rand([num_ant_per_chain, 1])
        #temp_angle = torch.zeros([num_ant_per_chain, 1])
        A = torch.block_diag(A, torch.exp(1j * math.pi * temp_angle))
    F_RF = A[num_ant_per_chain:, 1:]
    F_BB = torch.randn([num_rf, num_user], dtype=torch.complex128)

    y_before = 0
    y_after = torch.norm(F_opt - F_RF @ F_BB, 'fro')**2
    r = 0
    ratio = 1
    # while (torch.abs(y_before - y_after) > 0.01):
    while ((ratio > 0.01) & (r <= 50)):

        r = r + 1

        y_before = y_after

        a1 = torch.cat([torch.ones([1, num_user * num_rf], dtype=torch.complex128), torch.zeros([1, 1], dtype=torch.complex128)], dim=1).squeeze()
        A1 = torch.diag(a1)
        a2 = torch.cat([torch.zeros([1, num_user * num_rf], dtype=torch.complex128), torch.ones([1, 1], dtype=torch.complex128)], dim=1).squeeze()
        A2 = torch.diag(a2)

        temp = torch.kron(torch.eye(num_user), F_RF)
        F_opt_vec = torch.reshape(torch.transpose(F_opt, 0, 1), [-1, 1])
        C_upper = torch.cat([torch.conj(torch.transpose(temp, 0, 1)) @ temp, -torch.conj(torch.transpose(temp, 0, 1)) @ F_opt_vec], dim=1)
        C_down = torch.cat([-torch.conj(torch.transpose(F_opt_vec, 0, 1)) @ temp, torch.conj(torch.transpose(F_opt_vec, 0, 1)) @ F_opt_vec], dim=1)
        C = torch.cat([C_upper, C_down], dim=0)

        X = cp.Variable([num_rf * num_user + 1, num_rf * num_user + 1], hermitian = True)
        cost = cp.real(cp.trace(C @ X))
        constraints = [cp.real(cp.trace(A1 @ X)) - num_rf * P_pow / num_ant == 0]
        constraints += [cp.real(cp.trace(A2 @ X)) - 1 == 0]
        constraints += [X >> 0]
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve()

        X_out = torch.from_numpy(X.value)
        V, D = torch.linalg.eig(X_out)
        num = torch.max(torch.abs(V).unsqueeze(dim=1), dim=0)[1]
        x = torch.sqrt(V[num]) * D[:, num]
        F_temp = torch.reshape(x[np.arange(0, num_rf * num_user), 0], [num_user, num_rf])
        F_BB = torch.transpose(F_temp, 0, 1)

        for na in np.arange(0, num_ant):
            m = math.floor(na * num_rf/num_ant)
            a = F_opt[na, :] @ torch.conj(torch.transpose(F_BB[m, :].unsqueeze(dim=0), 0, 1))
            F_RF[na, m] = a / torch.abs(a)
            # F_RF[na, m] = math.exp(1j * np.angle(F_opt[na, :] @ torch.conj(torch.transpose(F_BB[m, :].unsqueeze(dim=0), 0, 1))))

        y_after = torch.norm(F_opt - F_RF @ F_BB, 'fro')**2
        ratio = abs(y_before - y_after) / y_before
        # print(y_after)

    return F_RF, F_BB, r