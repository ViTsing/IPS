# -*- coding: utf-8 -*-
# @Time    : 2018/3/27 18:00
# @Author  : timothy
# susceptible infected susceptible model

import scipy.integrate as spi
import numpy as np
import pylab as pl
import networkx as nx
import random as rd


class SIS:
    def __init__(self, beta, gamma, n_nodes, t_range):
        # 初始化
        self.BETA = beta  # 感染率
        self.GAMMA = gamma  # 治愈率
        self.N = n_nodes  # 网络节点数
        self.t_range = t_range  # 常微分方程迭代步数 np.arange(0, 50, 10e-2)
        self.Internet_state = np.zeros(self.N)
        self.adjacent_Matrix = None

    def random_source(self):
        # 开始随机选不同两个节点染毒（Internet_state: 网络状态，source_1, source_2 都是node的id）
        source_1 = rd.randint(0, self.N - 1)
        source_2 = source_1
        while source_2 == source_1:
            source_2 = rd.randint(0, self.N - 1)
        # 修改source 1，2 为感染态
        self.Internet_state[source_1] = 1
        self.Internet_state[source_2] = 1
        print('source_1', source_1)
        print('source_2', source_2)
        return

    def init_Graph(self, file, label, r_flag=False):
        '''
        初始化图结构
        :param r_flag: 随机初始化标志
        :param file:
        :return: 邻接矩阵
        '''
        if r_flag is False:
            scale_free_network = nx.random_graphs.barabasi_albert_graph(self.N, 1)
            self.adjacent_Matrix = nx.to_numpy_matrix(scale_free_network)
        else:
            karate_G = nx.read_gml(file, label=label)
            self.adjacent_Matrix = nx.to_numpy_matrix(karate_G)
            # nx.draw(karate_G)
            # plt.show()
        return

    # 常微分方程组
    def diff_eqs(self, net_state, t):
        # 初始化
        y = np.zeros(self.N)
        # 迭代一轮
        for i in range(self.N):
            # 节点i邻居的感染情况那个sigma
            neighbor_sum = 0
            for j in range(self.N):
                # 得到邻居节点的感染状况之和
                neighbor_sum += self.adjacent_Matrix[i, j] * net_state[j]
            y[i] = (1 - net_state[i]) * self.BETA * neighbor_sum - self.GAMMA * net_state[i]
        # 输出迭代结果 y : array, shape (len(y0)，)
        return y

    # 开始计算,迭代t_range次
    def run_ode(self):
        result = spi.odeint(func=self.diff_eqs, y0=self.Internet_state, t=self.t_range)
        return result

    def sample_result(self, result):
        for j in range(result.shape[1]):
            for i in range(result.shape[0]):
                _rd = np.random.RandomState()
                rd_mean = _rd.uniform(0, 1)
                if result[i][j] >= rd_mean:
                    result[i][j] = 1
                else:
                    result[i][j] = 0
        return result


def main():
    sis_range = np.arange(0, 50, 10e-2)
    sis = SIS(0.5, 0.2, 34, sis_range)
    sis.random_source()
    sis.init_Graph('data\\karate.gml', 'id')
    result = sis.run_ode()
    print('shape of result:', result.shape)
    # axis = 0 :
    result_mean = np.mean(result, axis=1)  # 染毒节点平均占比

    _result = sis.sample_result(result)
    _result_mean = np.mean(_result, axis=1)

    # 出图
    def plot(infected, _infected):
        pl.plot(infected, '-rs', label='infected')
        pl.plot(_infected, 'o', label='_infected')
        pl.legend(loc=0)
        pl.xlabel('Time')
        pl.ylabel('Ratio')
        pl.show()

    plot(result_mean, _result_mean)


if __name__ == '__main__':
    main()
