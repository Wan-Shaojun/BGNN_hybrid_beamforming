import numpy as np
import random
import torch
import math

####################################################生成通信信道，BGNN隐藏层size###############################################################################################################
# ctrl + / 批量注释
# ctrl + r 批量替换
def DFT_codebook_generate(num_ant_per_chain):

    DFT_codebook = torch.zeros([num_ant_per_chain, num_ant_per_chain], dtype=torch.complex128)
    for nr in np.arange(0, num_ant_per_chain):
        for nc in np.arange(0, num_ant_per_chain):
            a = -1 * 2 * math.pi * nr * nc / num_ant_per_chain
            DFT_codebook[nr, nc] = math.cos(a) + 1j * math.sin(a)

    return DFT_codebook

def Q_resolution_codebook(num_ant, q):

    size = 2**q
    size_1 = 2**(q-1)
    resolution = (2 * math.pi) / size
    Q_codebook = torch.zeros([num_ant, size], dtype=torch.complex128)
    for nt in np.arange(0, num_ant):
        for ns in np.arange(0, size):
            a = -1 * resolution * (nt * ns - size_1)
            Q_codebook[nt, ns] = math.cos(a) + 1j * math.sin(a)

    return Q_codebook

def Rayleigh_channel_generate(batch_size, num_user, num_ant):

    channel = (1/math.sqrt(2)) * (torch.randn([batch_size, num_user, num_ant]) + 1j * torch.randn([batch_size, num_user, num_ant])).type(torch.complex128)
    return channel

def mu_mmWave_channel_generate(num_user, num_ant, num_cl, num_ray, Nr=1, d=0.5, array_type=1):

    channel = torch.empty([num_user, num_ant], dtype= torch.complex128)
    num_path = num_cl * num_ray
    # At_user = torch.zeros([num_ant, num_user * num_path], dtype=torch.complex128)
    for nu in np.arange(0, num_user):
        # channel[nu,:], At_user[:, np.arange(num_path * nu, num_path * (nu + 1))] = mmWave_channel_generate(num_ant, num_cl, num_ray, Nr, d, array_type)
        channel[nu, :] = mmWave_channel_generate(num_ant, num_cl, num_ray, Nr, d, array_type)

    return channel
    # return channel, At_user


def channel_generate(batch_size, num_user, num_ant, num_cl, num_ray):

    mmWave_channel = torch.empty([batch_size, num_user, num_ant], dtype=torch.complex128)
    # At_sum = torch.zeros([batch_size, num_ant, num_user*num_cl*num_ray], dtype=torch.complex128)
    for batch_index in np.arange(0, batch_size):
        # mmWave_channel[batch_index, :, :], At_sum[batch_index, :, :] = mu_mmWave_channel_generate(num_user, num_ant, num_cl, num_ray)
        mmWave_channel[batch_index, :, :] = mu_mmWave_channel_generate(num_user, num_ant, num_cl, num_ray)
    # return mmWave_channel, At_sum
    return mmWave_channel

def mmWave_channel_generate(num_ant, num_cl, num_ray, Nr=1, d=0.5, array_type=1):

    num_path = num_cl * num_ray
    if array_type == 1:
        sigGain = np.sqrt(num_ant * Nr/num_path) * np.ones([num_cl, 1])
        sigAngle = (7.5 * math.pi / 180) * np.ones([2,1])
        DirTX = np.array([-1 * math.pi, math.pi])
        DirRX = np.array([-1 * math.pi, math.pi])

    GainAlpha = np.zeros([num_path, 1], dtype=np.complex128)                                                            #如果不加np.complex128, 下面就会报错
    AOD_angle = np.zeros([1, num_path])
    AOA_angle = np.zeros([1, num_path])

    if array_type == 1:
        meanAOA = (2 * np.random.rand(num_cl, 1) - 1) * (DirTX[1] - DirTX[0])/2 + (DirTX[1] - DirTX[0])/2               #若为np.random.rand(num_cl,1) * (DirTx[1] - DirTx[0])会报错，np.array数据无法和tensor的数据相乘
        meanAOD = (2 * np.random.rand(num_cl, 1) - 1) * (DirRX[1] - DirRX[0])/2 + (DirTX[1] - DirRX[0])/2
        meanAngle = np.concatenate((meanAOA, meanAOD), axis=1)                                                          #array的拼接尽量用concatenate((ab,b),axis),axis=0,行增加；axis=1,列增加
        #meanAngle = np.array([meanAOA, meanAOD])                                                                       #不加np.array(),直接A=[a,b]是list数据类型，索引不能用A[i,j]
        At = np.zeros([num_ant, num_path], dtype=np.complex128)
        Ar = np.zeros([Nr, num_path], dtype=np.complex128)
        iN = 0
        while iN < num_path:
            icls = math.floor(iN/num_ray)                                                                               #np.floor() 默认返回浮点数， math.floor返回整数
            sig = sigGain[icls, 0]
            GainAlpha[iN, 0] = sig * (np.random.randn(1) + 1j * np.random.randn(1)) / np.sqrt(2)                        #创建array,tensor时，默认都是实数，如果要定义为复数，需要在初始化的时候就定义好

            meanDepart = meanAngle[icls, 0]
            meanArrival = meanAngle[icls, 1]
            sigDepart = sigAngle[0]
            sigArrival = sigAngle[1]

            AngD = genLaplacian(meanDepart, sigDepart)
            AngA = genLaplacian(meanArrival, sigArrival)
            AOD_angle[0, iN] = AngD
            AOA_angle[0,iN] = AngA

            Attmp = np.exp(1j * math.pi * d * np.sin(AngD) * np.arange(0, num_ant)) / np.sqrt(num_ant)                  # np.exp()函数只能对一个数求指数，np就可以对整个array求指数！其他函数类似sqrt,floor等是同样的道理
            At[:, iN] = Attmp.transpose()                                                                               #permute实际上是定义顺序， 如果permute(0,1)则不会对矩阵进行任何变化
            Artmp = np.exp(1j * math.pi * d * np.sin(AngA) * np.arange(0, Nr)) / np.sqrt(Nr)
            Ar[:, iN] = Artmp.transpose()
            iN = iN + 1

    GainAlpha = np.squeeze(GainAlpha)                                                                                   #此处不用squeeze(),GainAlpha为而二维数组，diag()输出为第一个元素
    channel_array = Ar @ np.diag(GainAlpha) @ np.conj(At.transpose())
    channel = torch.tensor(channel_array, dtype=torch.complex128)
    # At_1 = torch.tensor(At, dtype=torch.complex128) * np.sqrt(num_ant)

    return channel


def genLaplacian(mu, sig):
    b = sig / np.sqrt(2)
    u = np.random.rand(1) - 1/2
    x = mu - b * np.sign(u) * np.log(1 - 2 * np.abs(u)) / np.log(np.exp(1))                                       #matlab有的数学运算，np一般都有，且都是一样写法

    return x

# def hidden_channel_generate(num_ant, num_M):
#
#     user_channel = torch.tensor([2 * num_ant + num_M + 2, 200, 200]).reshape(1, -1)
#     antenna_channel = torch.tensor([4 * num_ant + num_M, 200, 200]).reshape(1, -1)
#     # analogue_channel = torch.tensor([2 * num_M, 200, 200]).reshape(1, -1)
#     analogue_channel = torch.tensor([2 * num_M + 2 * num_ant, 200, 200]).reshape(1, -1)
#     digital_channel = torch.tensor([2 * num_M + 2, 200, 200]).reshape(1, -1)
#     # digital_channel = torch.tensor([2 * num_M, 200, 200]).reshape(1, -1)
#
#     hidden_channel = torch.cat([user_channel, antenna_channel, analogue_channel, digital_channel], dim=0)
#     return hidden_channel

def hidden_channel_generate_1(num_ant, num_M):

    user_channel = torch.tensor([2 * num_ant + num_M + 2, 200, 200]).reshape(1, -1)
    antenna_channel = torch.tensor([4 * num_ant + num_M, 200, 200]).reshape(1, -1)
    # analogue_channel = torch.tensor([2 * num_M, 200, 200]).reshape(1, -1)
    analogue_channel = torch.tensor([num_M + 2 * num_ant, 200, 200]).reshape(1, -1)
    digital_channel = torch.tensor([num_M + 2, 200, 200]).reshape(1, -1)

    hidden_channel = torch.cat([user_channel, antenna_channel, analogue_channel, digital_channel], dim=0)
    return hidden_channel

def hidden_channel_generate_CNN(num_ant, num_M):

    cnn_channel_num = 8

    user_channel = torch.tensor([2 * num_ant + num_M + 2, 200, 200]).reshape(1, -1)
    antenna_channel = torch.tensor([4 * num_ant + num_M, 200, 200]).reshape(1, -1)
    # analogue_channel = torch.tensor([2 * num_M, 200, 200]).reshape(1, -1)
    analogue_channel = torch.tensor([num_M + 2 * num_ant, 200, 200]).reshape(1, -1)
    digital_channel = torch.tensor([(num_M + 2) * 1, 200, 200]).reshape(1, -1)
    cnn_channel = torch.tensor([1, cnn_channel_num, cnn_channel_num]).reshape(1, -1)

    hidden_channel = torch.cat([user_channel, antenna_channel, analogue_channel, digital_channel, cnn_channel], dim=0)
    return hidden_channel

def hidden_channel_generate_ZF(num_ant, num_M):

    user_channel = torch.tensor([2 * num_ant + num_M + 1, 200, 200]).reshape(1, -1)
    antenna_channel = torch.tensor([4 * num_ant + num_M, 200, 200]).reshape(1, -1)
    # analogue_channel = torch.tensor([2 * num_M, 200, 200]).reshape(1, -1)
    analogue_channel = torch.tensor([num_M + 2 * num_ant, 200, 200]).reshape(1, -1)
    digital_channel = torch.tensor([num_M + 1, 200, 200]).reshape(1, -1)

    hidden_channel = torch.cat([user_channel, antenna_channel, analogue_channel, digital_channel], dim=0)
    return hidden_channel

def hidden_channel_generate_FCMLP(num_ant, num_user, num_rf):

    hidden_size = 200 * num_user
    # hidden_size = 200
    channel = torch.tensor([num_user * num_ant * 2 + num_user, hidden_size, hidden_size])

    return channel

def hidden_channel_generate_FCCNN(num_ant, num_user):

    hidden_size = 200 * num_user
    cnn_dimension = 8
    cnn_channel = torch.tensor([1, cnn_dimension, cnn_dimension])
    # user_channel = torch.tensor([num_user * num_ant * 2, hidden_size, hidden_size])

    return cnn_channel