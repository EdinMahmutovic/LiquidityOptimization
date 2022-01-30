import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ForexVisualizer:
    def __init__(self, forex_exchange):
        self.currencies = forex_exchange.currencies

        self.lending_rates = pd.DataFrame(data=forex_exchange.lending_rate_history.T, columns=self.currencies)
        self.deposit_rates = pd.DataFrame(data=forex_exchange.deposit_rate_history.T, columns=self.currencies)

        self.exchange_rates = {}
        for idx, currency in enumerate(self.currencies):
            currency_df = pd.DataFrame(data=forex_exchange.exchange_rate_history[idx, :, :].T, columns=self.currencies)
            self.exchange_rates.update({currency: currency_df})

    def plot_interest_rate(self):
        plt.plot(self.lending_rates, label="lending rate")
        plt.plot(self.deposit_rates, label="deposit rate")
        plt.legend()
        plt.show()

    def plot_currency_rate(self, currency):
        plt.plot(self.exchange_rates[currency], label=self.exchange_rates[currency].columns)
        plt.legend()
        plt.grid()
        plt.plot()




