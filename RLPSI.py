from LPSI import LPSI
from SIS import SIS
import numpy as np
import pylab as pl
from operator import itemgetter
from COS import cosine_dist
from math import e as E


def RLPSI(j, net_state, sis_range):
    sis_cov = SIS(0.5, 0.2, 34, sis_range)
    sis_cov.Internet_state[j] = 1
    sis_cov.init_Graph('data\\karate.gml', 'id')
    result_cov = sis_cov.run_ode()
    sample_cov = sis_cov.sample_result(result_cov)
    net_state_cov = sample_cov[-1, :]
    score_cov = cosine_dist(net_state, net_state_cov)

    sis_half = SIS(0.5, 0.2, 34, sis_range / 2)
    sis_half.Internet_state[j] = 1
    sis_half.init_Graph('data\\karate.gml', 'id')
    result_half = sis_half.run_ode()
    sample_half = sis_half.sample_result(result_half)
    net_state_half = sample_half[-1, :]
    score_half = cosine_dist(net_state, net_state_half)

    return (score_cov + score_half) / 2


def main():
    top_k = 5
    beta = 1
    sis_range = np.arange(0, 50, 10e-2)
    sis = SIS(0.5, 0.2, 34, sis_range)
    list_source = sis.random_source(top_k)
    sis.init_Graph('data\\karate.gml', 'id')
    result = sis.run_ode()
    # print('shape of result:', result.shape)
    # axis = 0 :
    result_mean = np.mean(result, axis=1)  # 染毒节点平均占比

    sample_result = sis.sample_result(result)
    _result_mean = np.mean(sample_result, axis=1)

    # plot(result_mean, _result_mean)

    net_state = sample_result[-1, :]

    lpsi = LPSI(sis.adjacent_Matrix, 0.4, net_state)
    c = lpsi.get_converge()
    c = np.array(c)
    c = enumerate(list(c.T[0, :]))
    l_c = []
    node_labels = dict()
    for i, j in c:
        l_c.append((i, j))
        node_labels[i] = str(round(j, 2))
    # 标签迭代结果
    r_c = sorted(l_c, key=itemgetter(1), reverse=True)

    f_c = list()

    for j, score in r_c:
        # 可多次计算取平均
        R_score = RLPSI(j, net_state, sis_range)
        F_score = 0.3 * R_score + 0.7 / (1 + E ** (-score))
        t = (j, F_score)
        f_c.append(t)

    f_c = sorted(f_c, key=itemgetter(1), reverse=True)
    # 按顺序选取top-k构成候选节点集
    list_predict = list()
    for j, score in f_c[0:top_k]:
        list_predict.append(j)
    count = 0
    # 如果命中count自增
    for item in list_predict:
        if item in list_source:
            count += 1
    print('precision', count / top_k)
    # print('Result:', list_source, list_predict)
    # sis.show(labels=node_labels)
    precision = count / top_k
    recall = count / 2
    f_score = (2 * precision * recall) / (precision + recall + 0.001)
    return f_score

    # 出图


def plot(infected, _infected):
    pl.plot(infected, '-rs', label='infected')
    pl.plot(_infected, 'o', label='_infected')
    pl.legend(loc=0)
    pl.xlabel('Time')
    pl.ylabel('Ratio')
    pl.show()


if __name__ == '__main__':
    run_times = 30
    sum_p = 0
    for i in range(run_times):
        sum_p += main()
        i += 1
        print(i)
    average = sum_p / run_times
    print(average)
