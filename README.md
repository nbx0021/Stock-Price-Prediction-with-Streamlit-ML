# 📈 Stock Market Prediction using Streamlit and Machine Learning

## Overview

This application predicts **stock market prices** using historical data and machine learning models. It is built with **Streamlit** for the user interface and employs models such as **SVR, Linear Regression, Random Forest, and Decision Tree** for stock price forecasting.

The app displays:
1. Monthly stock data and historical trends.
2. Model comparison with accuracy metrics (RMSE and R² scores).
3. Predicted vs actual stock prices.
4. Future predictions for the stock price.

---

## Features

- **Stock Data Visualization**: View historical stock prices at monthly intervals.
- **Model Comparison**: Compare predictions from four machine learning models:
   - SVR (Support Vector Regression)
   - Linear Regression
   - Random Forest Regressor
   - Decision Tree Regressor
- **Prediction Accuracy**: RMSE and R² values for each model.
- **Future Forecast**: Predict stock prices for the next 30 periods.
- **Interactive UI**: Input stock ticker symbols, date ranges, and view results in real time.

---

## Screenshots

### 1. Historical Stock Data and Closing Prices
![Historical Stock Data](screenshots/historical_data.png)

---

### 2. Actual vs Predicted Prices
![Actual vs Predicted](screenshots/actual_vs_predicted.png)

---

### 3. Model Accuracy Metrics
![Model Accuracy](screenshots/model_accuracy.png)

---

### 4. Future Stock Price Predictions
![Future Predictions](screenshots/future_predictions.png)

---

## Setup Instructions

Follow these steps to run the app locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/nbx0021/Stock-Price-Prediction-with-Streamlit-ML.git
   cd stock-prediction-streamlit
   ```

2. **Install Dependencies**:
   Make sure you have Python installed. Then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the App**:
   Execute the Streamlit app with:
   ```bash
   streamlit run app.py
   ```

4. **Access the App**:
   Open the provided URL in your browser (default is `http://localhost:8501`).

---

## Requirements

- Python 3.8 or higher
- Libraries:
   - `streamlit`
   - `yfinance`
   - `matplotlib`
   - `pandas`
   - `scikit-learn`
   - `numpy`

---

## How It Works

1. **Fetch Historical Data**: The app retrieves stock data from Yahoo Finance at monthly intervals.
2. **Train Models**: Four machine learning models are trained on the historical data.
3. **Evaluate Accuracy**: RMSE and R² scores are displayed for comparison.
4. **Visualize Results**: The app generates:
   - Historical stock price trends
   - Actual vs predicted charts
   - Future forecasts

---

## Contributing

Contributions are welcome! Fork this repository, make changes, and submit a pull request.

---

## License

This project is licensed under the **MIT License**.

