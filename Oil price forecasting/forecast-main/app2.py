import pandas as pd
import numpy as np
import streamlit as st
import tensorflow
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler 
import base64

# Set page title and favicon
st.set_page_config(page_title='Crude Oil Price Prediction App', page_icon=':oil_drum:')

# Load the trained model
model = load_model('my_model.h5')

# Load the data
df = pd.read_csv('brent-daily.csv')

# Preprocess the data
df = df.set_index('Date')
df = df.rename(columns={'Date': 'ds', 'Price': 'y'})

# Define a function to create sequences for the model
seq_len = 30
def create_sequences(X, y, time_steps=seq_len):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

# Set page title and subtitle
st.title('Crude Oil Price Prediction')

# Set background and text color
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #F4F4F4;
    }
    .stButton>button {
        background-color: #2E4053;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set logo and background image
from PIL import Image
logo = Image.open("logo.png")
logo = logo.resize((50,50))
st.sidebar.image(logo, use_column_width=False)
bg_image = Image.open("bg.png").convert('RGBA')
bg_image = bg_image.resize((250,250))
bg_image.putalpha(128)
st.image(bg_image, use_column_width=False)
# Set forecast horizon using a slider
forecast_horizon = st.slider(
    "Enter the number of days to be forecasted", 
    min_value=1, max_value=365, step=1
)

# Add a progress bar
latest_iteration = st.empty()
bar = st.progress(0)

# Add a delay to simulate long-running computation
import time
for i in range(100):
    latest_iteration.text(f'Calculating the forecast... {i+1}%')
    bar.progress(i + 1)
    time.sleep(0.01)

# Make predictions
latest_data = df.tail(seq_len).values.reshape(-1, 1)
scaler = StandardScaler()
latest_data_scaled = scaler.fit_transform(latest_data)
forecast = []
for i in range(forecast_horizon):
    X = latest_data_scaled.reshape(1, seq_len, 1)
    y_pred = model.predict(X)
    y_pred_rescaled = scaler.inverse_transform(y_pred)
    forecast.append(y_pred_rescaled[0][0])
    latest_data = np.append(latest_data[1:], y_pred_rescaled)
    latest_data_scaled = scaler.transform(latest_data.reshape(-1, 1))
    
# Display the forecasted values in a table and a chart
forecast_dates = pd.date_range(start=df.index[-1], periods=forecast_horizon+1, freq='D')[1:]
forecast_df = pd.DataFrame(forecast, index=forecast_dates, columns=['forecast'])
st.subheader('Forecasted Prices :money_with_wings:')
st.dataframe(forecast_df)
st.line_chart(forecast_df)
# Add a button to download the forecasted values as a CSV file
if st.button('Download Forecast as CSV'):
    csv = forecast_df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="forecast.csv">Download Forecast as CSV</a>'
    st.markdown(href, unsafe_allow_html=True)
