import streamlit as st
from myprophet import Prophet
from myprophet.plot import plot_plotly
import pandas as pd
from plotly import graph_objs as go

# Title and introduction
st.title('Time Series Forecasting with Prophet')
st.write('Upload a CSV file to get started. The CSV file can have multiple columns, but you need to select one column as `ds` (date) and one as `y` (value to predict).')

# File uploader
uploaded_file = st.file_uploader("Choose a file")

# If a file is uploaded, read it as a dataframe
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Display the dataframe
    st.write('Raw data')
    st.write(df)

    # Let the user select the column for 'ds' and 'y'
    date_column = st.selectbox('Which column is the date?', df.columns)
    value_column = st.selectbox('Which column is the value?', df.columns)

    # Rename the selected columns to 'ds' and 'y'
    df.rename(columns={date_column: 'ds', value_column: 'y'}, inplace=True)

    # Prophet requires columns to be named 'ds' and 'y'
    # Ensure the 'ds' column is datetime type
    df['ds'] = pd.to_datetime(df['ds'])

    # Ask user for the number of periods to forecast
    n_periods = st.number_input('Enter the number of periods to forecast', value=30)

    # Create and fit the model
    model = Prophet()
    model.fit(df)

    # Make future dataframe
    future = model.make_future_dataframe(periods=n_periods)

    # Forecast
    forecast = model.predict(future)

    # Show and plot forecast
    st.write('Forecast data')
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(n_periods))

    # Plotting the forecast
    fig1 = plot_plotly(model, forecast)
    st.plotly_chart(fig1)

    # Plotting forecast components
    fig2 = model.plot_components(forecast)
    st.write(fig2)

else:
    st.info('Awaiting CSV file to be uploaded. Once uploaded, select the date and value columns for forecasting.')
