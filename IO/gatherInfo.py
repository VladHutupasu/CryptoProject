import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


# get market info for bitcoin from the start of April, 2013 to the current day
bitcoin_market_info = pd.read_html("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130428&end="+time.strftime("%Y%m%d"))[0]

# convert the date string to date yyyy-mm-dd
bitcoin_market_info = bitcoin_market_info.assign(Date=pd.to_datetime(bitcoin_market_info['Date']))

# if Volume is equal to '-' convert it to 0
bitcoin_market_info.loc[bitcoin_market_info['Volume']=="-",'Volume']=0

# convert to int
bitcoin_market_info['Volume'] = bitcoin_market_info['Volume'].astype('int64')


# look at the first few rows
print(bitcoin_market_info.head())

dateCol      = np.array(bitcoin_market_info['Date'])[::-1]
openCol      = np.array(bitcoin_market_info['Open'])[::-1]
highCol      = np.array(bitcoin_market_info['High'])[::-1]
lowCol       = np.array(bitcoin_market_info['Low'])[::-1]
closeCol     = np.array(bitcoin_market_info['Close'])[::-1]
volumeCol    = np.array(bitcoin_market_info['Volume'])[::-1]
marketCapCol = np.array(bitcoin_market_info['Market Cap'])[::-1]


years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month

fig1 = plt.subplot(2, 1, 1)
plt.plot(dateCol, closeCol)
plt.title("Evolution of bitcoin over time")
plt.ylabel("Close price $")
fig1.xaxis.set_major_locator(years)
fig1.xaxis.set_minor_locator(months)
fig1.grid()

fig2 = plt.subplot(2, 1, 2)
plt.bar(dateCol, volumeCol)
plt.ylabel("Volume $ (bn)", fontsize=12)
fig2.xaxis.set_major_locator(years)
fig2.xaxis.set_minor_locator(months)
fig2.set_yticks([int('%d000000000'%i) for i in range(25)])
fig2.set_yticklabels(range(25))
fig2.grid()

plt.gcf().autofmt_xdate()
plt.show()