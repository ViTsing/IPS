import numpy as np
from scipy.spatial.distance import pdist


def cosine_dist(x, y):
    '''
    :param x:
    :param y:
    :return: 余弦距离
    '''
    sim = pdist(np.vstack([x, y]), 'cosine')
    return sim[0]


if __name__ == '__main__':
    a = [2, 2, 0]
    b = [0, 0, 1]
    print(cosine_dist(a, b))
