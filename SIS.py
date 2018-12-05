# -*- coding: utf-8 -*-
# @Time    : 2018/3/27 18:00
# @Author  : timothy
# susceptible infected susceptible model

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

# 初始化网络
scale_free_network = nx.random_graphs.barabasi_albert_graph(N, 1)
adjacent_Matrix = nx.to_numpy_matrix(scale_free_network)


# 常微分方程组
def diff_eqs(net_state, t):
    # 初始化
    y = np.zeros(N)
    # 迭代一轮
    for i in range(N):
        # 节点i邻居的感染情况那个sigma
        neighbor_sum = 0
        for j in range(N):
            # 得到邻居节点的感染状况之和
            neighbor_sum += adjacent_Matrix[i, j] * net_state[j]
        y[i] = (1 - net_state[i]) * BETA * neighbor_sum - GAMMA * net_state[i]
    # 输出迭代结果 y : array, shape (len(y0)，)
    return y


# 开始计算,迭代t_range次
def run_ode():
    result = spi.odeint(func=diff_eqs, y0=Internet_state, t=t_range)
    return result


# 出图
def plot(infected, _infected):
    pl.plot(infected, '-rs', label='infected')
    pl.plot(_infected, 'o', label='_infected')
    pl.legend(loc=0)
    pl.xlabel('Time')
    pl.ylabel('Ratio')
    pl.show()


def main():
    result = run_ode()
    print('shape of result:', result.shape)
    # axis = 0 :
    result_mean = np.mean(result, axis=1)  # 染毒节点平均占比
    for j in range(result.shape[1]):
        for i in range(result.shape[0]):
            _rd = np.random.RandomState()
            rd_mean = _rd.uniform(0, 1)
            print(rd_mean)
            if result[i][j] >= rd_mean:
                result[i][j] = 1
            else:
                result[i][j] = 0
    _result_mean = np.mean(result, axis=1)
    plot(result_mean, _result_mean)


if __name__ == '__main__':
    main()
