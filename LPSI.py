import numpy as np


class LPSI:

    def __init__(self, adjacent_matrix, alpha, net_state):
        self.adjacent_matrix = adjacent_matrix
        self.alpha = alpha
        self.net_state = net_state

        self.matrix_S = self.produce_S()

    def produce_S(self):
        _matrix_d = np.zeros(self.adjacent_matrix.shape)
        dim = len(self.adjacent_matrix[0])
        for i in range(dim):
            _matrix_d[i][i] = np.sqrt(np.sum(self.adjacent_matrix[i]), -1 / 2)
        _matrix_s = np.multiply(_matrix_d, self.adjacent_matrix)
        matrix_s = np.multiply(_matrix_s, _matrix_d)
        return matrix_s
