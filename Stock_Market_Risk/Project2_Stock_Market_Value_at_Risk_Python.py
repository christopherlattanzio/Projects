#normal imports
import pandas as pd
from pandas import Series , DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from pandas.io.data import DataReader
from pandas_datareader import data

from datetime import datetime

#list of stocks
tech_list = ['AAPL','GOOG','MSFT','AMZN']

#time frame
end = datetime.now()

start = datetime(end.year-1,end.month,end.day)

#building dataframe for each stock
for stock in tech_list:
    globals()[stock] = DataReader(stock,'yahoo',start,end)

closing_df = DataReader(tech_list,'yahoo',start,end)['Adj Close']

tech_rets = closing_df.pct_change()

#basic Risk Analyst
rets = tech_rets.dropna()

area = np.pi*20

plt.scatter(rets.mean(),rets.std(),s = area)

plt.xlabel('Expected Return')
plt.ylabel('Risk')

for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(
        label,
        xy = (x, y), xytext = (30, 20),
        textcoords = 'offset points', ha = 'left', va = 'top',
        arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3, rad=-0.3'))

plt.show()


# Value at Risk - value of money that can be lost
# Bootstrap method

#print(rets['AAPL'].quantile(0.05)*1000000*-1)
#at a 95% certainty expected daily losses are:

# Value at risk Monte Carlo method
days = 365

dt = 1/days

mu = rets.mean()['GOOG']

sigma = rets.std()['GOOG']

def stock_monte_carlo(start_price,days,mu,sigma):
    
    price = np.zeros(days)
    price[0] = start_price
    
    shock = np.zeros(days)
    drift = np.zeros(days)
    
    for x in range(1,days):
        
        shock[x] = np.random.normal(loc=mu*dt,scale=sigma*np.sqrt(dt))
        
        drift[x] = mu * dt
        
        price[x] = price[x-1] + (price[x-1] * (drift[x] + shock[x]))
        
    return price

start_price = 752.84

for run in range(100):
    plt.plot(stock_monte_carlo(start_price,days,mu,sigma))
    
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Monte Carlo Analysis for Google')
plt.show()

runs = 10000

sims = np.zeros(runs)

for run in range(runs):
    sims[run] = stock_monte_carlo(start_price,days,mu,sigma)[days-1]


q = np.percentile(sims,1)

plt.hist(sims,bins=200)

#start price
plt.figtext(0.6, 0.8, s="Start price: $%.2f" %start_price)

#mean end price
plt.figtext(0.6, 0.7, "Mean final price: $%.2f" %sims.mean())

#variance of the price (within 99% confidence interval)
plt.figtext(0.6, 0.6, "VaR(0.99): $%.2f" % (start_price - q,))

#display 1% quantile
plt.figtext(0.15, 0.6, "q(0.99): $%.2f" %q)

#plot a line at the 1% quantile result
plt.axvline(x=q, linewidth=4, color='r')

#title
plt.title("Final price distribution for Google Stock after %s days" %days, weight='bold')

plt.show()