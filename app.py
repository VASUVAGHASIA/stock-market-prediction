import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("Stock Price Predictor App")

# ------------------ Stock Input ------------------ #
stock = st.text_input("Enter the Stock ID", "GOOG")

# ------------------ Date Range ------------------ #
end = datetime.now()
start = datetime(end.year-20, end.month, end.day)

# ------------------ Download Stock Data ------------------ #
google_data = yf.download(stock, start, end, auto_adjust=False)

if google_data.empty:
    st.error("No data found for this stock. Please check the ticker.")
    st.stop()

# ------------------ Load Model ------------------ #
try:
    model = load_model("Latest_stock_price_model.keras")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ------------------ Show Stock Data ------------------ #
st.subheader("Stock Data")
st.write(google_data)

# ------------------ Split Data ------------------ #
splitting_len = int(len(google_data) * 0.7)

if len(google_data) - splitting_len <= 100:
    st.error("Not enough data to make predictions. Need more than 100 rows after splitting.")
    st.stop()

x_test = google_data[['Close']].iloc[splitting_len:]

# ------------------ Plotting Function ------------------ #
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'Orange')
    plt.plot(full_data['Close'], 'b')
    if extra_data and extra_dataset is not None:
        plt.plot(extra_dataset)
    return fig

# ------------------ Moving Averages ------------------ #
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

# ------------------ Scale Data ------------------ #
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test)

# ------------------ Prepare x_data and y_data ------------------ #
x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

if len(x_data) == 0:
    st.error("Not enough data to generate predictions. Need at least 100 rows for the sliding window.")
    st.stop()

# ------------------ Predictions ------------------ #
predictions = model.predict(x_data)

# ------------------ Inverse Transform ------------------ #
inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

# ------------------ Plotting Predictions ------------------ #
ploting_data = pd.DataFrame(
    {
        'original_test_data': inv_y_test.reshape(-1),
        'predictions': inv_pre.reshape(-1)
    },
    index=google_data.index[splitting_len+100:]
)

st.subheader("Original values vs Predicted values")
st.write(ploting_data)

st.subheader('Original Close Price vs Predicted Close price')
fig = plt.figure(figsize=(15,6))
plt.plot(pd.concat([google_data['Close'][:splitting_len+100], ploting_data['predictions']], axis=0))
plt.legend(["Data- not used", "Original Test data", "Predicted Test data"])
st.pyplot(fig)
