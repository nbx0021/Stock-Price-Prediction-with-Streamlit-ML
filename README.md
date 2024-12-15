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

![Historical Stock Data and Closing Prices](https://github.com/user-attachments/assets/8e6da584-81fe-4f08-9a85-43e2dcc3199a)


![Historical Stock Data and Closing Prices1](https://github.com/user-attachments/assets/86d6da94-d06f-403b-ae4e-6c2a5a5170ec)


---

### 2. Actual vs Predicted Prices
![Actual vs Predicted Prices](https://github.com/user-attachments/assets/e048c278-de15-4ead-8049-06996668787c)


![Actual vs Predicted Prices1](https://github.com/user-attachments/assets/57e72fca-5fe4-40ed-af17-84fb0f09d554)

---

### 3. Model Accuracy Metrics

![Model Accuracy Metrics](https://github.com/user-attachments/assets/d7277361-0664-4ca7-942f-48b6c4210502)

---

### 4. Future Stock Price Predictions

![Future Stock Price Predictions](https://github.com/user-attachments/assets/e7983bd2-e5d2-447f-ac96-04dc05042222)


![Future Stock Price Predictions1](https://github.com/user-attachments/assets/3a00546b-33f7-4811-872f-f3a7a579e430)
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

