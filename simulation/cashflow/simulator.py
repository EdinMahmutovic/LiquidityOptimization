import numpy as np
import matplotlib.pyplot as plt
import random
import warnings
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)


class RandomCashFlow:
    def __init__(self, num_suppliers, num_customers, mean_income, mean_outgoings, domestic_currency, t0_fx_rates):

        self.num_suppliers = num_suppliers
        self.num_customers = num_customers

        self.avg_ingoing_cash = mean_income
        self.avg_outgoing_cash = mean_outgoings

        self.currencies = t0_fx_rates.columns.tolist()
        self.balances = pd.DataFrame(data=np.zeros((1, len(self.currencies)+1)), columns=['Date'] + self.currencies, index=[0])
        self.cashflows = pd.DataFrame(columns=['Date', 'Type', 'Counterpart', 'Amount', 'Currency'])
        self.balances.iloc[0]['Date'] = -1

        self.supplier_currencies = [random.choice(self.currencies) for _ in range(self.num_suppliers)]
        avg_supplier_cost_yearly = np.zeros(self.num_suppliers)
        avg_supplier_orders_yearly = np.zeros(self.num_suppliers)
        self.avg_supplier_cost = np.zeros(self.num_suppliers)
        self.supplier_prob = np.zeros(self.num_suppliers)
        outgoing_cash_leftover = self.avg_outgoing_cash
        for supplier_id in range(self.num_suppliers):
            currency = self.supplier_currencies[supplier_id]

            domestic_yearly_cost = outgoing_cash_leftover if supplier_id == (
                        self.num_suppliers - 1) else np.random.uniform(low=0, high=outgoing_cash_leftover)
            avg_supplier_cost_yearly[supplier_id] = t0_fx_rates.loc[domestic_currency, currency] * domestic_yearly_cost

            avg_supplier_orders_yearly[supplier_id] = np.random.uniform(low=1, high=24)
            self.avg_supplier_cost[supplier_id] = avg_supplier_cost_yearly[supplier_id] / avg_supplier_orders_yearly[
                supplier_id]
            self.supplier_prob[supplier_id] = avg_supplier_orders_yearly[supplier_id] / 365
            outgoing_cash_leftover -= domestic_yearly_cost

        self.customer_currencies = [random.choice(self.currencies) for _ in range(self.num_customers)]
        avg_customer_income_yearly = np.zeros(self.num_customers)
        avg_customer_orders_yearly = np.zeros(self.num_customers)
        self.avg_customer_income = np.zeros(self.num_customers)
        self.customer_prob = np.zeros(self.num_customers)
        ingoing_cash_leftover = self.avg_ingoing_cash
        for buyer_id in range(self.num_customers):
            currency = self.customer_currencies[buyer_id]

            domestic_yearly_income = ingoing_cash_leftover if buyer_id == (
                        self.num_customers - 1) else np.random.uniform(low=0, high=ingoing_cash_leftover)
            avg_customer_income_yearly[buyer_id] = t0_fx_rates.loc[domestic_currency, currency] * np.random.uniform(
                low=0, high=ingoing_cash_leftover)

            avg_customer_orders_yearly[buyer_id] = np.random.uniform(low=1, high=24)
            self.avg_customer_income[buyer_id] = avg_customer_income_yearly[buyer_id] / avg_customer_orders_yearly[
                buyer_id]
            self.customer_prob[buyer_id] = avg_customer_orders_yearly[buyer_id] / 365
            ingoing_cash_leftover -= domestic_yearly_income

    def generate_history(self, t):
        for i in range(t):
            self.step(i)

    def step(self, t):
        balance = self.balances.iloc[-1, :].copy()
        balance['Date'] = t
        for supplier_id in range(self.num_suppliers):
            currency = self.supplier_currencies[supplier_id]
            if np.random.uniform(low=0, high=1) < self.supplier_prob[supplier_id]:
                transaction_amount = np.random.normal(loc=self.avg_supplier_cost[supplier_id],
                                                      scale=self.avg_supplier_cost[supplier_id] * 0.1)

                transaction = pd.Series({'Date': t, 'Type': 'outgoing', 'Counterpart': 'supplier_' + str(supplier_id),
                                         'Amount': transaction_amount, 'Currency': currency})
                self.cashflows = self.cashflows.append(transaction, ignore_index=True)
                balance[currency] -= transaction_amount

        for buyer_id in range(self.num_customers):
            currency = self.customer_currencies[buyer_id]
            if np.random.uniform(low=0, high=1) < self.customer_prob[buyer_id]:
                transaction_amount = np.random.normal(loc=self.avg_customer_income[buyer_id],
                                                      scale=self.avg_customer_income[buyer_id] * 0.1)
                transaction = pd.Series({'Date': t, 'Type': 'incoming', 'Counterpart': 'costumer_' + str(buyer_id),
                                         'Amount': transaction_amount, 'Currency': currency})
                self.cashflows = self.cashflows.append(transaction, ignore_index=True)
                balance[currency] += transaction_amount

        self.balances = self.balances.append(balance, ignore_index=True)


'''
rates = pd.DataFrame(columns=["DKK", "SEK", "NOK"], index=["DKK", "SEK", "NOK"])
rates.loc["DKK", "DKK"] = 1
rates.loc["SEK", "SEK"] = 1
rates.loc["NOK", "NOK"] = 1

rates.loc["DKK", "SEK"] = 1.4
rates.loc["DKK", "NOK"] = 1.1

rates.loc["SEK", "DKK"] = 1 / rates.loc["DKK", "SEK"]
rates.loc["NOK", "DKK"] = 1 / rates.loc["DKK", "NOK"]

rates.loc["SEK", "NOK"] = 1 / (rates.loc["DKK", "SEK"] * rates.loc["NOK", "DKK"])
rates.loc["NOK", "SEK"] = 1 / rates.loc["SEK", "NOK"]

cf = RandomCashFlow(5, 100, 10000, 10000, "DKK", rates)
cf.generate_history(t=1000)
cf.balances.iloc[:, 1:].plot()
plt.show()

print(cf.cashflows.head(20))
print(cf.balances.head(20))
'''