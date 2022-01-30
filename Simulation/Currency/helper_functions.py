import numpy as np
from RandomCorrMat import RandomCorrMat


def generate_random_currency_variance(rates):
    exchange_variance = np.random.uniform(low=rates / 100, high=rates / 50, size=[len(rates)] * 2)
    currency_variance = np.mean(exchange_variance / rates, axis=1)
    return currency_variance


def generate_random_correlation_matrix(size):
    corr_mat = RandomCorrMat.randCorr(size)
    return corr_mat


def generate_covariance_matrix(currency_variance, corr_mat):
    cov_matrix = np.zeros(corr_mat.shape)
    for i in range(cov_matrix.shape[0]):
        for j in range(cov_matrix.shape[1]):
            cov_matrix[i, j] = corr_mat[i, j] * currency_variance[i] * currency_variance[j]
    return cov_matrix
