# import streamlit as st
# import pandas as pd
# import numpy as np
# # from keras.models import load_model
# from tensorflow.keras.models import load_model
# import matplotlib.pyplot as plt
# import yfinance as yf

# st.title("Stock Price Predictor App")

# stock = st.text_input("Enter the Stock ID", "GOOG")

# from datetime import datetime
# end = datetime.now()
# start = datetime(end.year-20,end.month,end.day)

# google_data = yf.download(stock, start, end)

# model = load_model("Latest_stock_price_model.keras")
# st.subheader("Stock Data")
# st.write(google_data)

# splitting_len = int(len(google_data)*0.7)
# x_test = pd.DataFrame(google_data.Close[splitting_len:])

# def plot_graph(figsize, values, full_data, extra_data = 0, extra_dataset = None):
#     fig = plt.figure(figsize=figsize)
#     plt.plot(values,'Orange')
#     plt.plot(full_data.Close, 'b')
#     if extra_data:
#         plt.plot(extra_dataset)
#     return fig

# st.subheader('Original Close Price and MA for 250 days')
# google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
# st.pyplot(plot_graph((15,6), google_data['MA_for_250_days'],google_data,0))

# st.subheader('Original Close Price and MA for 200 days')
# google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()
# st.pyplot(plot_graph((15,6), google_data['MA_for_200_days'],google_data,0))

# st.subheader('Original Close Price and MA for 100 days')
# google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()
# st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'],google_data,0))

# st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
# st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'],google_data,1,google_data['MA_for_250_days']))

# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler(feature_range=(0,1))
# scaled_data = scaler.fit_transform(x_test[['Close']])

# x_data = []
# y_data = []

# for i in range(100,len(scaled_data)):
#     x_data.append(scaled_data[i-100:i])
#     y_data.append(scaled_data[i])

# x_data, y_data = np.array(x_data), np.array(y_data)

# predictions = model.predict(x_data)

# inv_pre = scaler.inverse_transform(predictions)
# inv_y_test = scaler.inverse_transform(y_data)

# ploting_data = pd.DataFrame(
#  {
#   'original_test_data': inv_y_test.reshape(-1),
#     'predictions': inv_pre.reshape(-1)
#  } ,
#     index = google_data.index[splitting_len+100:]
# )
# st.subheader("Original values vs Predicted values")
# st.write(ploting_data)

# st.subheader('Original Close Price vs Predicted Close price')
# fig = plt.figure(figsize=(15,6))
# plt.plot(pd.concat([google_data.Close[:splitting_len+100],ploting_data], axis=0))
# plt.legend(["Data- not used", "Original Test data", "Predicted Test data"])
# st.pyplot(fig)


import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

st.title("Stock Price Predictor App")

# Stock input
stock = st.text_input("Enter the Stock ID", "GOOG")

# Date range
end = datetime.now()
start = datetime(end.year-20, end.month, end.day)

# Download stock data
google_data = yf.download(stock, start, end, auto_adjust=False)

# Load model
model = load_model("Latest_stock_price_model.keras")

# Show stock data
st.subheader("Stock Data")
st.write(google_data)

# Split data for testing
splitting_len = int(len(google_data) * 0.7)
# FIXED: ensure x_test is a DataFrame with column 'Close'
x_test = google_data[['Close']].iloc[splitting_len:]

# Function to plot
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'Orange')
    plt.plot(full_data['Close'], 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

# Moving averages
st.subheader('Original Close Price and MA for 250 days')
google_data['MA_for_250_days'] = google_data['Close'].rolling(250).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_250_days'], google_data))

st.subheader('Original Close Price and MA for 200 days')
google_data['MA_for_200_days'] = google_data['Close'].rolling(200).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_200_days'], google_data))

st.subheader('Original Close Price and MA for 100 days')
google_data['MA_for_100_days'] = google_data['Close'].rolling(100).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'], google_data))

st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'], google_data, 1, google_data['MA_for_250_days']))

# Scale data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test)  # x_test already has 'Close' column

# Prepare x_data and y_data for prediction
x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Make predictions
predictions = model.predict(x_data)

# Inverse transform
inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

# Plotting data
ploting_data = pd.DataFrame(
    {
        'original_test_data': inv_y_test.reshape(-1),
        'predictions': inv_pre.reshape(-1)
    },
    index=google_data.index[splitting_len+100:]
)

st.subheader("Original values vs Predicted values")
st.write(ploting_data)

# Plot original vs predicted
st.subheader('Original Close Price vs Predicted Close price')
fig = plt.figure(figsize=(15,6))
plt.plot(pd.concat([google_data['Close'][:splitting_len+100], ploting_data['predictions']], axis=0))
plt.legend(["Data- not used", "Original Test data", "Predicted Test data"])
st.pyplot(fig)
