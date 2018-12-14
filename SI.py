# -*- coding: utf-8 -*-
# @Time    : 2018/3/27 18:00
# @Author  : timothy
# susceptible infected susceptible model

import scipy.integrate as spi
import numpy as np
import pylab as pl
import networkx as nx
import random as rd


class SI:
    def __init__(self, beta, n_nodes, t_range):
        # 初始化
        self.BETA = beta  # 感染率
        self.N = n_nodes  # 网络节点数
        self.t_range = t_range  # 常微分方程迭代步数 np.arange(0, 50, 10e-2)
        self.Internet_state = np.zeros(self.N)
        self.adjacent_Matrix = None
        self.Graph = None
        self.final_state = list()

    def random_source(self, K):
        list_source = list()
        while len(list_source) < K:
            # 开始随机选不同节点染毒（Internet_state: 网络状态，source_i是node的id）
            source_i = rd.randint(0, self.N - 1)
            if source_i not in list_source:
                list_source.append(source_i)
                # 修改source_i为感染态
                self.Internet_state[source_i] = 1
            else:
                continue
        # for i in list_source:
        #     print("infected source id: ", i)
        return list_source

    def init_Graph(self, file, label, r_flag=False):
        '''
        初始化图结构
        :param r_flag: 随机初始化标志
        :param file:
        :return: 邻接矩阵
        '''
        if r_flag is False:
            self.Graph = nx.random_graphs.barabasi_albert_graph(self.N, 1)
            self.adjacent_Matrix = nx.to_numpy_matrix(self.Graph)
        else:
            self.Graph = nx.read_gml(file, label=label)
            self.adjacent_Matrix = nx.to_numpy_matrix(self.Graph)
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
            y[i] = (1 - net_state[i]) * self.BETA * neighbor_sum
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

        self.final_state = result[-1, :]

        return result

    def show(self, labels={}):
        node_color = list()
        for i in self.final_state:
            if i == 0:
                node_color.append('g')
            else:
                node_color.append('r')

        pos = nx.kamada_kawai_layout(self.Graph)

        nx.draw_networkx(self.Graph, pos, arrows=True, with_labels=True, nodelist=self.Graph.nodes(),  # 基本参数

                         node_color=node_color, node_size=280, alpha=1,  # 结点参数,alpha是透明度

                         width=1, style='solid',
                         # 边参数(solid|dashed|dotted,dashdot)

                         labels=labels, font_size=10, font_weight='normal',

                         label=['Final state']

                         )

        pl.show()
        return


def main():
    sis_range = np.arange(0, 50, 10e-2)
    sis = SI(0.5, 34, sis_range)
    sis.random_source(2)
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

    # plot(result_mean, _result_mean)
    sis.show()


if __name__ == '__main__':
    main()
