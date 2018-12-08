from LPSI import LPSI
from SIS import SIS
import numpy as np
import pylab as pl


def main():
    sis_range = np.arange(0, 50, 10e-2)
    sis = SIS(0.5, 0.2, 34, sis_range)
    sis.random_source()
    sis.init_Graph('data\\karate.gml', 'id')
    result = sis.run_ode()
    print('shape of result:', result.shape)
    # axis = 0 :
    result_mean = np.mean(result, axis=1)  # 染毒节点平均占比

    sample_result = sis.sample_result(result)
    _result_mean = np.mean(sample_result, axis=1)

    # plot(result_mean, _result_mean)

    net_state = sample_result[-1, :]

    lpsi = LPSI(sis.adjacent_Matrix, 0.5, net_state)
    c = lpsi.get_converge()
    c = np.ndarray.reshape(c, -1, 1)
    print(c, type(c))


# 出图
def plot(infected, _infected):
    pl.plot(infected, '-rs', label='infected')
    pl.plot(_infected, 'o', label='_infected')
    pl.legend(loc=0)
    pl.xlabel('Time')
    pl.ylabel('Ratio')
    pl.show()


if __name__ == '__main__':
    main()
