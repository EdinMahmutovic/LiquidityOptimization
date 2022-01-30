import numpy as np
from helper_functions import generate_covariance_matrix, generate_random_currency_variance, generate_random_correlation_matrix
from RandomCorrMat import RandomCorrMat
from visualizer import ForexVisualizer

'''
This class builds a simulation of a forex exchange with n currencies. A list of currencies must be given to
the constructor of the class to make a valid instance of this class. If rates, mean_returns and cov_matrix are
given then it must be given w.r.t to the first currency which is DKK. Also the rate between the same currency,
e.g. DKK/DKK is constant at 1 and thus mean_returns[0]=0 and cov_matrix[i,i]=0, i=0:n-1. 
'''


class ForexExchangeSimulator(object):
    def __init__(self, currencies, rates=None, mean_returns=None, cov_matrix=None, corr_matrix=None):
        """
        :param currencies: list of n string elements that represents different currencies used in the simulation.
        :param rates: 1d numpy array of length n that represents the rates w.r.t the first currency which must be DKK.
        :param mean_returns: 1d numpy array of length n that represents the daily geometric mean return for each currency.
        :param cov_matrix: 2d numpy array of size n x n that represents the covariance between all the daily returns.

        :method step:
        :method simulate_rate:
        :method simulate_fee:
        :method normalize_rate:
        """
        assert len(currencies) > 1, "List of currencies must contain more than 1 element."
        assert currencies[0] == 'DKK', "First element in currencies must be 'DKK'"

        if rates is not None:
            assert rates[0] == 1, "Rates are relative to the first currency. " \
                                  "Thus first element in mean_rates must be 1."
            assert rates.shape[0] == len(currencies), "rates must have same number " \
                                                      "of elements as currencies."

        if mean_returns is not None:
            assert len(currencies) == mean_returns.shape[0], "mean_returns must have same number " \
                                                             "of elements as currencies."
            if mean_returns[0] != 0:
                print("First element in mean_returns will be modified to 0. "
                      "Because {}/{} rate must stay constant".format(currencies[0], currencies[0]))

        if cov_matrix is not None:
            assert len(currencies) == cov_matrix.shape[0], "Both dimensions of cov_matrix must " \
                                                           "be equal to len(currencies)."
            assert cov_matrix.shape[0] == cov_matrix.shape[1], "cov_matrix must be square."

            if not np.all(cov_matrix.diagonal() == 0):
                print("All elements on the diagonal in cov_matrix will be modified to 0 since"
                      "the rate between the same currency stays constant, i.e. 1. ")

        self.currencies = currencies
        self.num_currencies = len(currencies)

        self.rates = rates if rates is not None else np.random.uniform(low=0.01, high=10, size=self.num_currencies)
        self.rates[0] = 1

        self.currency_var = cov_matrix.diagonal() if cov_matrix is not None \
            else generate_random_currency_variance(self.rates)

        self.correlation_matrix = corr_matrix if corr_matrix is not None \
            else generate_random_correlation_matrix(self.num_currencies)

        self.Sigma = cov_matrix if cov_matrix is not None \
            else generate_covariance_matrix(self.currency_var, self.correlation_matrix)

        self.exchange_rate = self.__build_fx_rate_matrix(self.rates)
        self.lending_rate = np.random.normal(loc=0.09, scale=0.01, size=self.num_currencies)
        self.deposit_rate = np.random.normal(loc=0.03, scale=0.01, size=self.num_currencies)

        self.exchange_rate_history = np.expand_dims(self.exchange_rate.copy(), 2)
        self.lending_rate_history = np.expand_dims(self.lending_rate.copy(), 1)
        self.deposit_rate_history = np.expand_dims(self.deposit_rate.copy(), 1)

    def step(self, t=1):
        for time_step in range(t):
            self.__update_correlation_matrix()
            self.__update_currency_variance()
            self.__update_covariance_matrix()
            self.__update_exchange_rate()
            self.__update_lending_rate()
            self.__update_deposit_rate()

            self.exchange_rate_history = np.concatenate((self.exchange_rate_history,
                                                         np.expand_dims(self.exchange_rate, 2)), axis=2)
            self.lending_rate_history = np.concatenate((self.lending_rate_history,
                                                        np.expand_dims(self.lending_rate, 1)), axis=1)
            self.deposit_rate_history = np.concatenate((self.deposit_rate_history,
                                                        np.expand_dims(self.deposit_rate, 1)), axis=1)

    def __update_correlation_matrix(self):
        updated_corr_matrix = self.correlation_matrix + np.random.normal(loc=0, scale=0.01, size=self.correlation_matrix.shape)
        self.correlation_matrix = RandomCorrMat.nearcorr(updated_corr_matrix)

    def __update_currency_variance(self):
        self.currency_var *= 1 + np.random.multivariate_normal(mean=np.zeros(self.num_currencies), cov=self.Sigma)

    def __update_covariance_matrix(self):
        self.Sigma = generate_covariance_matrix(currency_variance=self.currency_var, corr_mat=self.correlation_matrix)

    def __update_exchange_rate(self):
        rates = self.exchange_rate[0, :]
        rates += np.random.multivariate_normal(mean=np.zeros(self.num_currencies), cov=self.Sigma)
        self.exchange_rate = self.__build_fx_rate_matrix(rates)

    def __update_lending_rate(self):
        self.lending_rate *= 1 + np.random.normal(loc=0, scale=0.001, size=self.num_currencies)

    def __update_deposit_rate(self):
        self.deposit_rate *= 1 + np.random.normal(loc=0, scale=0.001, size=self.num_currencies)

    def __build_fx_rate_matrix(self, rates):
        rate_matrix = np.eye(self.num_currencies)
        for i in range(self.num_currencies):
            for j in range(self.num_currencies):
                rate_matrix[i, j] = rates[j] / rates[i]

        return rate_matrix


exchange = ForexExchangeSimulator(currencies=['DKK', 'EUR', 'USD', 'SEK'])
exchange.step(t=1000)
visual = ForexVisualizer(exchange)
print(visual.lending_rates.shape)
'''visual.plot_interest_rate()

import matplotlib.pyplot as plt
plt.plot(1,1)
plt.show()'''