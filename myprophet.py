import streamlit as st
from fbprophet import Prophet
import pandas as pd
import numpy as np
from fbprophet.plot import plot_plotly
import plotly.offline as py

# Function to load data
@st.cache
def load_data():
    # Replace with your own data loading mechanism
    df = pd.read_csv('path_to_your_time_series_data.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    return df

# Function to run Prophet and make future predictions
def run_prophet(df):
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=365)  # Adjust the periods as needed
    forecast = m.predict(future)
    return forecast

# Title of the app
st.title('Time Series Forecasting with Prophet')

# File uploader allows user to add their own data
uploaded_file = st.file_uploader("Upload your time series CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    data['ds'] = pd.to_datetime(data['ds'])
    st.write(data)

    # Checkbox to show raw data
    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(data)

    # Button to run the forecast
    if st.button('Run Forecast'):
        st.subheader('Forecast data')
        forecast = run_prophet(data)
        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

        # Plot the forecast
        fig1 = plot_plotly(m, forecast)
        st.plotly_chart(fig1)

        # Show components of the forecast
        fig2 = m.plot_components(forecast)
        st.write(fig2)
