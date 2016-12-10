
# coding: utf-8

# In[2]:

import pandas as pd
from pandas import Series , DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().magic('matplotlib inline')


# In[3]:

from pandas.io.data import DataReader


# In[4]:

from datetime import datetime


# In[5]:

from __future__ import division


# In[6]:

tech_list = ['AAPL','GOOG','MSFT','AMZN']


# In[7]:

end = datetime.now()

start = datetime(end.year-1,end.month,end.day)


# In[8]:

for stock in tech_list:
    globals()[stock] = DataReader(stock,'yahoo',start,end)


# In[9]:

AAPL.describe()


# In[10]:

AAPL.info()


# In[11]:

AAPL['Adj Close'].plot(legend=True,figsize=(10,4))


# In[12]:

ma_day = [10,20,50]
for ma in ma_day:
    column_name = "MA for %s days" %(str(ma))
    
    AAPL[column_name] = Series.rolling(AAPL['Adj Close'],window=ma).mean()


# In[13]:

AAPL['Volume'].plot(legend=True,figsize=(10,4))


# In[14]:

AAPL[['Adj Close','MA for 10 days','MA for 20 days','MA for 50 days']].plot(subplots=False,figsize=(10,4))


# In[15]:

AAPL['Daily Return'] = AAPL['Adj Close'].pct_change()

AAPL['Daily Return'].plot(figsize=(10,4),legend=True,linestyle='--',marker='o')


# In[16]:

AAPL['Daily Return'] = AAPL['Adj Close'].pct_change()

AAPL['Daily Return'].plot(figsize=(10,4),legend=True)


# In[17]:

sns.distplot(AAPL['Daily Return'].dropna(),bins=100,color='purple')


# In[18]:

AAPL['Daily Return'].hist(bins=100)


# In[19]:

closing_df = DataReader(tech_list,'yahoo',start,end)['Adj Close']


# In[20]:

closing_df.head()


# In[21]:

tech_rets = closing_df.pct_change()


# In[22]:

tech_rets.head()


# In[23]:

sns.jointplot('GOOG','GOOG',tech_rets,kind = 'scatter',color='seagreen')


# In[24]:

sns.jointplot('GOOG','MSFT',tech_rets,kind = 'scatter',color='seagreen')


# In[25]:

sns.pairplot(tech_rets.dropna())


# In[26]:

returns_fig = sns.PairGrid(tech_rets.dropna())

returns_fig.map_upper(plt.scatter,color='purple')

returns_fig.map_lower(sns.kdeplot,cmap='cool_d')

returns_fig.map_diag(plt.hist,bins=30)


# In[27]:

returns_fig = sns.PairGrid(closing_df)

returns_fig.map_upper(plt.scatter,color='purple')

returns_fig.map_lower(sns.kdeplot,cmap='cool_d')

returns_fig.map_diag(plt.hist,bins=30)


# In[28]:

sns.heatmap(tech_rets.dropna())


# In[29]:

sns.heatmap(closing_df.dropna())


# In[30]:

# Load the example flights dataset and conver to long-form
flights_long = sns.load_dataset("flights")
flights = flights_long.pivot("month", "year", "passengers")

# Draw a heatmap with the numeric values in each cell
sns.heatmap(flights, annot=True, fmt="d", linewidths=.5)


# In[31]:

#basic Risk Analyst
rets = tech_rets.dropna()


# In[32]:

area = np.pi*20


# In[33]:

plt.scatter(rets.mean(),rets.std(),s = area)

plt.xlabel('Expected Return')
plt.ylabel('Risk')

for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(
        label,
        xy = (x, y), xytext = (30, 20),
        textcoords = 'offset points', ha = 'left', va = 'top',
        arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3, rad=-0.3'))


# In[35]:

# Value at Risk - value of money that can be lost
# Bootstrap method

sns.distplot(AAPL['Daily Return'].dropna(),bins=100,color='purple')



# In[36]:

rets.head()


# In[41]:

rets['AAPL'].quantile(0.05)*1000000*-1
#at a 95% certainty expected daily losses are:


# In[42]:

# Value at risk Monte Carlo method
days = 365

dt = 1/days

mu = rets.mean()['GOOG']

sigma = rets.std()['GOOG']


# In[43]:

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


# In[44]:

GOOG.head()


# In[45]:

start_price = 752.84

for run in range(100):
    plt.plot(stock_monte_carlo(start_price,days,mu,sigma))
    
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Monte Carlo Analysis for Google')


# In[46]:

runs = 10000

sims = np.zeros(runs)

for run in range(runs):
    sims[run] = stock_monte_carlo(start_price,days,mu,sigma)[days-1]


# In[47]:

q = np.percentile(sims,1)


# In[56]:

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
plt.title(u"Final price distribution for Google Stock after %s days" %days, weight='bold');




# In[ ]:



