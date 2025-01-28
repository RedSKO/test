import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prophet import Prophet
import matplotlib.pyplot as plt

# Load pre-processed invoice data
@st.cache_data
def load_data():
    return pd.read_csv("invoices.csv")

# Function to prioritize payments
def prioritize_payments(df):
    df['Due_In_Days'] = (pd.to_datetime(df['Due_Date']) - datetime.now()).dt.days
    df['Priority'] = np.where(df['Due_In_Days'] < 5, "High", 
                             np.where(df['Due_In_Days'] < 10, "Medium", "Low"))
    return df.sort_values(by=['Priority', 'Due_In_Days'])

# Function to detect anomalies
def detect_anomalies(df):
    anomalies = []
    # Example: Detect duplicate invoices
    duplicates = df[df.duplicated(subset=['Invoice_ID'], keep=False)]
    if not duplicates.empty:
        anomalies.append("Duplicate invoices detected: " + ", ".join(duplicates['Invoice_ID'].astype(str)))
    # Example: Detect unusually high amounts
    high_amounts = df[df['Amount'] > df['Amount'].quantile(0.95)]
    if not high_amounts.empty:
        anomalies.append("Unusually high amounts detected in invoices: " + ", ".join(high_amounts['Invoice_ID'].astype(str)))
    return anomalies

# Function to forecast cash flow risks
def forecast_cash_flow_risks(df):
    total_pending = df[df['Status'] == 'Pending']['Amount'].sum()
    due_soon = df[(df['Due_In_Days'] < 10) & (df['Status'] == 'Pending')]['Amount'].sum()
    return {
        "Total_Pending_Amount": total_pending,
        "Amount_Due_Soon": due_soon,
        "Risk_Level": "High" if due_soon > 0.5 * total_pending else "Low"
    }

# Function to forecast future payments using Prophet
def forecast_future_payments(df):
    # Aggregate daily pending amounts
    df['Invoice_Date'] = pd.to_datetime(df['Invoice_Date'])
    daily_pending = df[df['Status'] == 'Pending'].groupby('Invoice_Date')['Amount'].sum().reset_index()
    daily_pending.columns = ['ds', 'y']

    # Train Prophet model
    model = Prophet()
    model.fit(daily_pending)

    # Forecast for the next 30 days
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    return forecast, model

# Main Streamlit app
def main():
    st.title("AI-Powered Invoice Analysis")
    st.write("This app analyzes pre-processed invoice data (from ABBYY) and provides actionable insights.")

    # Load data
    df = load_data()
    st.write("### Invoice Data Preview")
    st.dataframe(df.head())

    # Prioritize payments
    st.write("### Payment Prioritization")
    prioritized_df = prioritize_payments(df)
    st.dataframe(prioritized_df[['Invoice_ID', 'Supplier_Name', 'Due_Date', 'Amount', 'Priority']])

    # Detect anomalies
    st.write("### Anomaly Detection")
    anomalies = detect_anomalies(df)
    if anomalies:
        for anomaly in anomalies:
            st.warning(anomaly)
    else:
        st.success("No anomalies detected.")

    # Forecast cash flow risks
    st.write("### Cash Flow Risk Forecast")
    cash_flow_risks = forecast_cash_flow_risks(df)
    st.write(f"Total Pending Amount: **{cash_flow_risks['Total_Pending_Amount']}**")
    st.write(f"Amount Due Soon (within 10 days): **{cash_flow_risks['Amount_Due_Soon']}**")
    st.write(f"Risk Level: **{cash_flow_risks['Risk_Level']}**")

    # Forecast future payments
    st.write("### Future Payment Forecasting")
    if st.button("Run Forecasting"):
        with st.spinner("Forecasting future payments..."):
            forecast, model = forecast_future_payments(df)
            st.write("Forecasted Daily Payments for the Next 30 Days:")
            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30))

            # Plot forecast
            fig = model.plot(forecast)
            st.pyplot(fig)

if __name__ == "__main__":
    main()
