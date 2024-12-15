import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from datetime import timedelta

# App Title
st.title("📈 Stock Price Prediction with Streamlit & ML")

# Sidebar: User Input
st.sidebar.header("Stock Input")
try:
    stock_ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., Reliance.NS):", value="Reliance.NS")
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-12"))
    if start_date >= end_date:
        st.sidebar.error("Start date must be before end date.")
except Exception as e:
    st.sidebar.error(f"Invalid input: {e}")

# Load Stock Data
@st.cache_data
def load_stock_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            raise ValueError("No data found for the given ticker and date range.")
        data = data.resample('M').last()  # Resample data to monthly frequency
        return data
    except Exception as e:
        st.error(f"Failed to load stock data: {e}")
        return pd.DataFrame()

stock_data = load_stock_data(stock_ticker, start_date, end_date)

if not stock_data.empty:
    # Display Data
    st.subheader("Stock Data")
    st.write(stock_data.head(20))

    # Line Chart of Closing Prices
    st.subheader("Stock Closing Prices")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(stock_data.index, stock_data['Close'], color='blue', linewidth=2)
    ax.set_title(f"{stock_ticker} Stock Closing Prices (Monthly)", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Closing Price (USD)", fontsize=12)
    ax.grid(True)
    st.pyplot(fig)

    # Scale Data
    scaler = MinMaxScaler()
    scaled_close = scaler.fit_transform(stock_data[['Close']])

    # Split Data into Training and Testing Sets
    train_size = int(len(scaled_close) * 0.8)
    train_data, test_data = scaled_close[:train_size], scaled_close[train_size:]

    # Model Definitions
    models = {
        'SVR': SVR(kernel='rbf', C=1e3, gamma=0.1),
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeRegressor(random_state=42)
    }

    # Function to Train and Predict
    def train_and_predict(model, X_train, y_train, X_test):
        try:
            model.fit(X_train, y_train)
            return model.predict(X_test)
        except Exception as e:
            st.error(f"Model training failed: {e}")
            return np.zeros_like(X_test[:, 0])

    # Prepare Data
    X_train = np.arange(train_size).reshape(-1, 1)
    y_train = train_data.flatten()
    X_test = np.arange(train_size, len(scaled_close)).reshape(-1, 1)

    # Predict and Evaluate Models
    predictions = {}
    errors = {}
    r2_scores = {}

    st.subheader("Model Comparison")

    for model_name, model in models.items():
        try:
            predictions[model_name] = train_and_predict(model, X_train, y_train, X_test)
            predicted_prices = scaler.inverse_transform(predictions[model_name].reshape(-1, 1))
            actual_prices = scaler.inverse_transform(test_data)

            # Calculate Metrics
            error = sqrt(mean_squared_error(actual_prices, predicted_prices))
            r2 = r2_score(actual_prices, predicted_prices)
            errors[model_name] = error
            r2_scores[model_name] = r2

            # Display Metrics
            st.write(f"**{model_name}:**")
            st.write(f"- RMSE: {error:.2f}")
            st.write(f"- R² Score: {r2:.2f}")

            # Plot Predictions
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            ax2.plot(stock_data.index[train_size:], actual_prices, label="Actual", color="green")
            ax2.plot(stock_data.index[train_size:], predicted_prices, label=f"Predicted ({model_name})", color="red")
            ax2.set_title(f"Actual vs Predicted Prices ({model_name})", fontsize=16)
            ax2.set_xlabel("Date", fontsize=12)
            ax2.set_ylabel("Closing Price (USD)", fontsize=12)
            ax2.legend()
            st.pyplot(fig2)
        except Exception as e:
            st.error(f"Error processing model {model_name}: {e}")

    # Future Predictions
    st.subheader("Future Price Predictions")
    future_days = 12  # Predict for next 12 months
    future_dates = [stock_data.index[-1] + timedelta(days=i*30) for i in range(1, future_days + 1)]
    future_X = np.arange(len(scaled_close), len(scaled_close) + future_days).reshape(-1, 1)

    for model_name, model in models.items():
        try:
            future_preds = model.predict(future_X)
            future_prices = scaler.inverse_transform(future_preds.reshape(-1, 1))

            # Plot Future Predictions
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            ax3.plot(stock_data.index, stock_data['Close'], label="Historical Data", color='green')
            ax3.plot(future_dates, future_prices, label=f"Future Predictions ({model_name})", color='blue')
            ax3.set_title(f"Future Stock Price Prediction ({model_name})", fontsize=16)
            ax3.set_xlabel("Date", fontsize=12)
            ax3.set_ylabel("Closing Price (USD)", fontsize=12)
            ax3.legend()
            st.pyplot(fig3)
        except Exception as e:
            st.error(f"Failed to predict future prices with {model_name}: {e}")

    # Display Summary Table
    st.subheader("Model Performance Summary")
    performance_df = pd.DataFrame({"RMSE": errors, "R² Score": r2_scores})
    st.dataframe(performance_df)
else:
    st.error("No data to display. Please adjust the input parameters.")
