# -*- coding: utf-8 -*-
# @Time    : 2018/3/27 18:00
# @Author  : timothy
# susceptible infected removed model (individual)

import scipy.integrate as spi
import numpy as np
import pylab as pl
import networkx as nx
import random as rd

# 初始化
BETA = 0.5  # 感染率
GAMMA = 0.2  # 治愈率
N = 50  # 网络节点数
t_range = np.arange(0, 50, 10e-2)  # 常微分方程迭代步数

# 开始随机选不同两个节点染毒（Internet_state: 网络状态，source_1, source_2 都是node的id）
Internet_state = np.zeros(N)
source_1 = rd.randint(0, N)
source_2 = source_1
while source_2 == source_1:
    source_2 = rd.randint(0, N)
# 修改source 1，2 为感染态
Internet_state[source_1] = 1
Internet_state[source_2] = 1

# 随机初始化网络
scale_free_network = nx.random_graphs.barabasi_albert_graph(N, 1)
adjacent_Matrix = nx.to_numpy_matrix(scale_free_network)

# 按照文件初始化网络
karate_G = nx.read_gml('.\data\\karate.gml')


# Y = dict()
# Y['susceptible'] = np.zeros(N)
# Y['infected'] = np.zeros(N)
# Y['removed'] = np.zeros(N)


# 常微分方程组
def diff_eqs(net_state, t):
    Y = np.zeros(N)
    for i in range(N):
        # 节点i邻居的感染情况那个sigma
        neighbor_sum = 0
        for j in range(N):
            # 得到邻居节点的感染状况
            neighbor_sum += adjacent_Matrix[i, j] * net_state[j]
        Y[i] = (1 - net_state[i]) * BETA * neighbor_sum - GAMMA * net_state[i]
    return Y


# 开始计算,迭代t_range次
def run_ode():
    result = spi.odeint(func=diff_eqs, y0=Internet_state, t=t_range)
    return result


# 出图
def plot(result):
    pl.plot(result, '-rs', label='Susceptible infected removed')
    pl.legend(loc=0)
    pl.xlabel('Time')
    pl.ylabel('Ratio')
    pl.show()


def main():
    result = run_ode()
    result_mean = np.mean(result, axis=1)  # 染毒节点平均占比
    plot(result_mean)


if __name__ == '__main__':
    main()
