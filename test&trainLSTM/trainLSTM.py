import pandas as pd
import time
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates
import numpy as np

# import the relevant Keras modules
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout

def build_model(inputs, output_size, neurons, activ_func="linear",
                dropout=0.25, loss="mae", optimizer="adam"):
    model = Sequential()

    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model



#--Get market info for bitcoin from the start of April, 2013 to the current day
bitcoin_market_info = pd.read_html("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130428&end="+time.strftime("%Y%m%d"))[0]
bitcoin_market_info = bitcoin_market_info.assign(Date=pd.to_datetime(bitcoin_market_info['Date']))
bitcoin_market_info.loc[bitcoin_market_info['Volume']=="-",'Volume']=0
bitcoin_market_info['Volume'] = bitcoin_market_info['Volume'].astype('int64')


#--Create Day Diff column with values--
market_info = bitcoin_market_info
market_info = market_info[market_info['Date']>='2016-01-01']
kwargs = { 'Day Diff': lambda x: (x['Close']-x['Open'])/x['Open']}
market_info = market_info.assign(**kwargs)


#--Create Close Off High and Volatility column--
#Close_Off_High represents the gap between the closing price and price high for that day, where values
#of -1 and 1 mean the closing price was equal to the daily low or daily high, respectively
#Volatility column is simply the difference between high and low price divided by the opening price
kwargs = { 'Close Off High': lambda x: 2*(x['High']- x['Close'])/(x['High']-x['Low'])-1,
        'Volatility': lambda x: (x['High']- x['Low'])/(x['Open'])}
market_info = market_info.assign(**kwargs)


#drop rest keep only Close,Volume,COH & Vola
model_data = market_info[['Date']+[metric for metric in ['Close','Volume','Close Off High','Volatility']]]
#reverse array for ascending order
model_data = model_data.sort_values(by='Date')


#Deleting date columne, as everything is sorted out and we'll not use date again
split_date = '2017-06-01'
#Useful graph could be drawn
training_set, test_set = model_data[model_data['Date']<split_date], model_data[model_data['Date']>=split_date]
training_set = training_set.drop('Date', 1)
test_set = test_set.drop('Date', 1)

window_len = 10
norm_cols = [metric for metric in ['Close','Volume']]

#TRAINING INPUTS
LSTM_training_inputs = []
for i in range(len(training_set)-window_len):
    temp_set = training_set[i:(i+window_len)].copy()
    for col in norm_cols:
        temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
    LSTM_training_inputs.append(temp_set)
LSTM_training_outputs = (training_set['Close'][window_len:].values/training_set['Close'][:-window_len].values)-1

#TEST INPUTS
LSTM_test_inputs = []
for i in range(len(test_set)-window_len):
    temp_set = test_set[i:(i+window_len)].copy()
    for col in norm_cols:
        temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
    LSTM_test_inputs.append(temp_set)
LSTM_test_outputs = (test_set['Close'][window_len:].values/test_set['Close'][:-window_len].values)-1

print("LSTM test outputs : ")
print(LSTM_test_outputs[0:5])
print("LSTM training outputs : ")
print(LSTM_training_outputs[0:5])



#Convert to np arrays from pandas dataframes
LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
LSTM_training_inputs = np.array(LSTM_training_inputs)

LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]
LSTM_test_inputs = np.array(LSTM_test_inputs)


# random seed for reproducibility
np.random.seed(202)
# initialise model architecture
eth_model = build_model(LSTM_training_inputs, output_size=1, neurons = 20)
# model output is next price normalised to 10th previous closing price
LSTM_training_outputs = (training_set['Close'][window_len:].values/training_set['Close'][:-window_len].values)-1
# train model on data
# note: eth_history contains information on the training error per epoch
eth_history = eth_model.fit(LSTM_training_inputs, LSTM_training_outputs,
                            epochs=50, batch_size=1, verbose=2, shuffle=True)


#Plot the error
fig, ax1 = plt.subplots(1,1)

ax1.plot(eth_history.epoch, eth_history.history['loss'])
ax1.set_title('Training Error')

if eth_model.loss == 'mae':
    ax1.set_ylabel('Mean Absolute Error (MAE)',fontsize=12)
# just in case you decided to change the model loss calculation
else:
    ax1.set_ylabel('Model Loss',fontsize=12)
ax1.set_xlabel('# Epochs',fontsize=12)
plt.show()