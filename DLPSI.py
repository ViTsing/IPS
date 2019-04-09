from LPSI import LPSI
from SI import SI
from SIS import SIS
from SIR import SIR
import numpy as np
import pylab as pl
from operator import itemgetter
from COS import cosine_dist
from math import e as E


def evaluate():
    dataset = 'karate'

    if dataset == 'karate':
        N_node = 34
        # K = 2,3,5
        num_source = 5
        iteration_range = np.arange(0, 50, 10e-2)
        ipm = SIR(0.5, 0.2, N_node, iteration_range)
        list_source = ipm.random_source(num_source)
        ipm.init_Graph('data\\karate.gml', 'id')
    else:
        N_node = 198
        num_source = 10
        iteration_range = np.arange(0, 50, 10e-2)
        ipm = SIR(0.5, 0.2, N_node, iteration_range)
        list_source = ipm.random_source(num_source)
        ipm.init_Graph('data\\jazz.net', 'id')
    result = ipm.run_ode()
    sample_result = ipm.sample_result(result)
    net_state = sample_result[-1, :]

    # # 正向推演
    t_net_state = []
    for item in net_state:
        if item == 0.:
            t_net_state.append(0.0)
        else:
            t_net_state.append(1.0)
    lpsi = LPSI(ipm.adjacent_Matrix, 0.5, net_state)
    c = lpsi.get_converge()
    c = np.array(c)
    c = c.T[0, :]

    # 反向推演
    _net_state = []
    for item in net_state:
        if item == 0.:
            _net_state.append(1.0)
        else:
            _net_state.append(0.0)
    _lpsi = LPSI(ipm.adjacent_Matrix, 0.5, _net_state)
    _c = _lpsi.get_converge()
    _c = np.array(_c)
    _c = _c.T[0, :]

    # 结合
    all_c = np.true_divide(c, _c)

    # 相似度
    c = enumerate(list(all_c))
    l_c = []
    node_labels = dict()
    for i, j in c:
        l_c.append((i, j))
        node_labels[i] = str(round(j, 2))
    # 标签迭代结果
    r_c = sorted(l_c, key=itemgetter(1), reverse=True)
    # 按顺序选取top-k构成候选节点集
    list_predict = list()
    for j, score in r_c[0:num_source]:
        list_predict.append(j)
    count = 0
    # 如果命中count自增
    for item in list_predict:
        if item in list_source:
            count += 1
    print('precision', count / num_source)
    # print('Result:', list_source, list_predict)
    # ipm.show(labels=node_labels)
    precision = count / num_source
    recall = count / num_source
    f_score = (2 * precision * recall) / (precision + recall + 0.001)
    return f_score


if __name__ == '__main__':
    run_times = 500
    sum_p = 0
    for i in range(run_times):
        sum_p += evaluate()
        i += 1
        print(i)
    average = sum_p / run_times
    print(average)
