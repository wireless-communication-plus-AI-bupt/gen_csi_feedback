import numpy as np


def cos_sim(vector_a, vector_b):
    """
    vector_a, vector_b:预编码向量, (B, Tx) B是样本个数, Tx是天线个数, 每一个元素是复数
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = vector_a * vector_b.H
    num1 = np.sqrt(vector_a * vector_a.H)
    num2 = np.sqrt(vector_b * vector_b.H)
    cos = (num / (num1 * num2))
    return cos


if __name__ == '__main__':
    pass