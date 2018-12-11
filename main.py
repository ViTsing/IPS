from LPSI import LPSI
from SIS import SIS
import numpy as np
import pylab as pl
from operator import itemgetter


def main():
    top_k = 5
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

    lpsi = LPSI(sis.adjacent_Matrix, 0.5, net_state)
    c = lpsi.get_converge()
    c = np.array(c)
    c = enumerate(list(c.T[0, :]))
    l_c = []
    node_labels = dict()
    for i, j in c:
        l_c.append((i, j))
        node_labels[i] = str(round(j, 2))
    r_c = sorted(l_c, key=itemgetter(1), reverse=True)
    list_predict = list()
    for j, score in r_c[0:top_k]:
        list_predict.append(j)
    count = 0
    for item in list_predict:
        if item in list_source:
            count += 1
    print('precision', count / top_k)
    # print('Result:', list_source, list_predict)
    # sis.show(labels=node_labels)
    return count / top_k


# 出图
def plot(infected, _infected):
    pl.plot(infected, '-rs', label='infected')
    pl.plot(_infected, 'o', label='_infected')
    pl.legend(loc=0)
    pl.xlabel('Time')
    pl.ylabel('Ratio')
    pl.show()


if __name__ == '__main__':
    sum_p = 0
    for i in range(500):
        sum_p += main()
        i += 1
        print(i)
    average = sum_p / 500
    print(average)
