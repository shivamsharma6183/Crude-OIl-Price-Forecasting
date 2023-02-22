
import pandas as pd
import numpy as np
import streamlit as st
import tensorflow
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler 



model=load_model('my_model.h5')

df = pd.read_csv('brent-daily.csv')
scaler = StandardScaler()
df=df.set_index('Date')
#df['Date'] = pd.to_datetime(df['Date'])
df = df.rename(columns={'Date': 'ds', 'Price': 'y'})

seq_len=30
def create_sequences(X, y, time_steps=30):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)



st.title('Crude Oil Price Prediction')
#last_data_points = st.text_input('No of days (comma-separated)', )
forecast_horizon = st.number_input('Enter the Number of Days to be Forecast : ',
                                       min_value=1,max_value=50,
                                       step=1)

if st.button('forecast'):

    #forecast_horizon = [float(i) for i in last_data_points.split(',')]


    # Scale the most recent data using StandardScaler

    latest_data = df.tail(seq_len).values.reshape(-1, 1)
    latest_data_scaled = scaler.fit_transform(latest_data)

    # Create a list to store the forecasted values
    forecast = []

    # Loop through the forecast horizon and make predictions
    for i in range(forecast_horizon):
        # Reshape the data into the expected input shape for the model
        X = latest_data_scaled.reshape(1, seq_len, 1)
        
        # Make a prediction using the model
        y_pred = model.predict(X)
        
        # Rescale the predicted value back to the original range
        y_pred_rescaled = scaler.inverse_transform(y_pred)
        
        # Append the forecasted value to the list
        forecast.append(y_pred_rescaled[0][0])
        
        # Update the latest data with the new forecasted value
        latest_data = np.append(latest_data[1:], y_pred_rescaled)
        latest_data_scaled = scaler.transform(latest_data.reshape(-1, 1))
        
    # Create a pandas dataframe to store the forecasted values
    forecast_dates = pd.date_range(start=df.index[-1], periods=forecast_horizon+1, freq='D')[1:]
    forecast_df = pd.DataFrame(forecast, index=forecast_dates, columns=['forecast'])

    # Print the forecasted values
    st.write(forecast_df)
    st.line_chart(forecast_df)

