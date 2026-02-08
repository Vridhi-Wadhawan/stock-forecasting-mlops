import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from google import genai
import os
import threading
import time
import subprocess

# Set Plot Style for better visuals
plt.style.use('ggplot')
sns.set_style('whitegrid')

# --- GLOBAL CONFIGURATION ---
FEATURE_COLS = [
    'Log_Returns', 'lag_1', 'lag_2', 'lag_5', 'lag_10', 'lag_21',
    'MA_5', 'MA_10', 'MA_21', 'VOL_5', 'VOL_10', 'VOL_21',
    'RSI_14', 'EMA_10', 'EMA_20'
]

# --- Static Model Performance & Historical Data ---
MODEL_METRICS = {
    # Using the R2 scores you provided in the previous message
    '1D_R2': 0.72,
    '30D_R2': 0.65,
    'Retrain_Threshold': 0.60}

@st.cache_data
def load_historical_predictions():
    """Loads historical model performance data (Actual vs. Predicted) for plotting."""
    try:
        # Using index_col=0 to correctly read the index column
        df_hist = pd.read_csv("historical_predictions.csv", index_col=0, parse_dates=True)
        return df_hist
    except FileNotFoundError:
        st.warning("Historical predictions file (historical_predictions.csv) not found. Past performance chart skipped.")
        return None

HISTORICAL_PREDICTIONS = load_historical_predictions()

# --- Load Models & Scaler ---
@st.cache_resource
def load_resources():
    try:
        xgb_1d = pickle.load(open("xgb_1d.pkl", "rb"))
        xgb_30d = pickle.load(open("xgb_30d.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))

        # --- Extract Feature Importance ---
        importance_scores = xgb_30d.feature_importances_
        feature_importance = dict(zip(FEATURE_COLS, importance_scores))

        return xgb_1d, xgb_30d, scaler, feature_importance

    except FileNotFoundError:
        st.error("Model files (xgb_1d.pkl, xgb_30d.pkl) or scaler (scaler.pkl) not found. Please upload them.")
        return None, None, None, None

xgb_1d, xgb_30d, scaler, FEATURE_IMPORTANCE = load_resources()

# --- Preprocessing Function ---
def preprocess_new_data(df, scaler):
    """
    Preprocess raw Yahoo stock data and return the last row of ML-ready features, price, and the DataFrame.
    """
    df = df.copy()
    df = df.sort_index()

    # Log Returns
    df['Log_Returns'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))

    # Lags
    for lag in [1, 2, 5, 10, 21]:
        df[f'lag_{lag}'] = df['Log_Returns'].shift(lag)

    # Moving Averages (Price & Volume)
    for window in [5, 10, 21]:
        df[f'MA_{window}'] = df['Adj Close'].rolling(window).mean()
        df[f'VOL_{window}'] = df['Volume'].rolling(window).mean()

    # RSI (14)
    def calculate_rsi(series, window=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(com=window-1, min_periods=window).mean()
        avg_loss = loss.ewm(com=window-1, min_periods=window).mean()
        # Handle division by zero for initial periods (where avg_loss might be 0)
        rs = avg_gain.divide(avg_loss.replace(0, np.nan)).fillna(0)
        return 100 - (100 / (1 + rs))

    df['RSI_14'] = calculate_rsi(df['Adj Close'])

    # EMA
    df['EMA_10'] = df['Adj Close'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df['Adj Close'].ewm(span=20, adjust=False).mean()

    # Clean NaNs
    df = df.dropna()

    if df.empty:
        return None, None, None

    X_all = df[FEATURE_COLS]

    # Scale features
    try:
        X_scaled_all = scaler.transform(X_all)
    except Exception as e:
        st.error(f"Scaling error: {e}. Ensure features match those used in training.")
        return None, None, None

    # Get last row for prediction
    last_features_scaled = X_scaled_all[-1].reshape(1, -1)
    last_price = df['Adj Close'].iloc[-1].item()

    return last_features_scaled, last_price, df

# --- AI Advice Function (LLM-Powered Narrative) ---
def generate_llm_insight(ticker, last_price, pred_price_1d, pred_price_30d, change_1d, change_30d, feature_importance, r2_score):
    """
    Calls the LLM to generate a narrative summary based on all model outputs.
    """

    # 1. Prepare the data payload for the LLM
    data_summary = {
        "Ticker": ticker,
        "Last_Price": f"₹{last_price:.2f}",
        "Predicted_1D_Price": f"₹{pred_price_1d:.2f} ({change_1d:.2f}%)",
        "Predicted_30D_Price": f"₹{pred_price_30d:.2f} ({change_30d:.2f}%)",
        "Model_R2_Score": f"{r2_score:.2f}",
        "Top_Feature_Importance": dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)[:3])
    }

    # 2. Construct the system prompt
    prompt = f"""
    You are a highly experienced Financial Analyst providing a daily market commentary.
    Your task is to analyze the provided stock forecast data for **{ticker}** and write a concise, professional, and actionable insight summary in 5-7 sentences.

    Follow these rules:
    1.  **Start** with a clear verdict.
    2.  **Highlight** the most important factor driving the 1-day and the 30-day prediction (use the 'Top_Feature_Importance' data).
    3.  **Include** a risk or caution note based on the Model_R2_Score (If R2 is < 0.70, mention it).
    4.  **Do NOT** include any introductory or concluding phrases like 'Based on the data' or 'In conclusion'. Just provide the commentary.
    5.  Use a professional and objective tone.

    Data for Analysis (JSON): {data_summary}
    """

    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")

        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt         )

        return "Success", response.text

    except Exception as e:
        print(f"LLM API Error/Missing Key: {e}")
        verdict = "Bullish" if change_30d > 0.5 else "Bearish" if change_30d < -0.5 else "Neutral"
        fallback_advice = f"**{verdict}** based on XGBoost (30-day forecast: {change_30d:.2f}%). LLM insight generation failed; check API key or connectivity."
        return "Caution", fallback_advice

# --- Streamlit App Layout ---
st.title(" Stock Price Forecasting App")
st.markdown("This app uses **XGBoost** models and **Gemini AI** to forecast and analyze stock price movements.")

# Sidebar only for inputs
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Enter Stock Ticker (Yahoo Finance format):", value="HDFCBANK.NS")
    st.caption("Examples: RELIANCE.NS, TCS.NS, INFY.NS")
    predict_btn = st.button("Generate Forecast")

    st.markdown("---")
    st.info("The Model is trained based on closing price for HDFC Bank from Jan 2020 - Nov 2025. Data Drift not taken into consideration due to short tenure.")


# Main Execution
if predict_btn:
    if not (xgb_1d and xgb_30d and scaler and FEATURE_IMPORTANCE):
        st.stop()

    with st.spinner(f"Fetching data and calculating indicators for {ticker}..."):
        try:
            # Fetch Data
            end_date = datetime.today()
            start_date = end_date - timedelta(days=5*365)
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)

            # --- CRITICAL FIX: Rename 'Close' to 'Adj Close' ---
            if 'Close' in df.columns:
                df.rename(columns={'Close': 'Adj Close'}, inplace=True)

            # Data validation checks
            if df.empty:
                st.error(f"No data found for {ticker}. Check the ticker symbol.")
            elif len(df) < 30:
                st.error(f"Only {len(df)} days of data were retrieved for {ticker}. Need more historical data (50+ days recommended) to calculate all indicators.")

            else:
                last_features, last_price, df_processed = preprocess_new_data(df, scaler)

                if last_features is not None:

                    # Predict
                    pred_ret_1d = xgb_1d.predict(last_features, validate_features=False)[0]
                    pred_ret_30d = xgb_30d.predict(last_features, validate_features=False)[0]

                    # Convert to Price
                    pred_price_1d = last_price * np.exp(pred_ret_1d)
                    pred_price_30d = last_price * np.exp(pred_ret_30d)

                    # Calculate Change (Percentage)
                    change_1d = (pred_price_1d - last_price) / last_price * 100
                    change_30d = (pred_price_30d - last_price) / last_price * 100

                    # --- Display AI Advice (LLM Feature) ---
                    current_r2 = MODEL_METRICS['30D_R2']
                    emoji, advice_text = generate_llm_insight(
                        ticker,
                        last_price,
                        pred_price_1d,
                        pred_price_30d,
                        change_1d,
                        change_30d,
                        FEATURE_IMPORTANCE,
                        current_r2
                    )
                    st.subheader(f"Insights {emoji}")
                    st.markdown(f"{advice_text}")
                    st.markdown("---")

                    # --- Display Clean Metrics ---
                    st.subheader(f"Forecast Results: {ticker}")
                    st.metric(label=f"Last Close Price ({df_processed.index[-1].strftime('%Y-%m-%d')})", value=f"₹{last_price:.2f}")

                    col1, col2 = st.columns(2)

                    col1.metric(
                        label="1-Day Predicted Price",
                        value=f"₹{pred_price_1d:.2f}",
                        delta=f"{change_1d:.2f}%",
                        delta_color="normal"
                    )

                    col2.metric(
                        label="30-Day Predicted Price",
                        value=f"₹{pred_price_30d:.2f}",
                        delta=f"{change_30d:.2f}%",
                        delta_color="normal"
                    )

                    st.markdown("---")

                    # --- Visualizations (Using Tabs) ---
                    tab1, tab2, tab3 = st.tabs([" Live Forecast & Trend", " Model Performance & History", " Data Analysis"])

                    with tab1: # Live Forecast & Trend
                        st.subheader("Historical Price Trend & Live Forecast")
                        chart_data = df_processed['Adj Close'].tail(500)

                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(chart_data.index, chart_data.values, label='Adj Close', color='#1f77b4', linewidth=2)

                        ax.axhline(last_price, color='red', linestyle='--', alpha=0.8, label='Last Close')
                        ax.axhline(pred_price_30d, color='green', linestyle=':', alpha=0.8, label='30-Day Target')

                        ax.set_title(f"{ticker} Adjusted Close Price (Last 500 Days)", fontsize=14)
                        ax.set_xlabel("Date")
                        ax.set_ylabel("Price (INR)")
                        ax.legend()
                        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                        st.pyplot(fig)

                    with tab2: # Model Performance & History (Data Drift Added Here)
                        st.subheader("Model Accuracy and Health Check")

                        # R2 Metrics and Warning
                        colA, colB = st.columns(2)
                        colA.metric("1-Day Model R² (Test)", f"{MODEL_METRICS['1D_R2']:.2f}")
                        colB.metric("30-Day Model R² (Test)", f"{MODEL_METRICS['30D_R2']:.2f}")

                        if MODEL_METRICS['30D_R2'] < MODEL_METRICS['Retrain_Threshold']:
                            st.warning(f" **WARNING:** 30-Day Model R² is below the {MODEL_METRICS['Retrain_Threshold']:.2f} threshold. **Retraining is Recommended!**")
                        else:
                            st.success(" Model R² is healthy and above the retraining threshold.")
                        st.caption("R² is the coefficient of determination on the test data.")

                        st.markdown("---")

                        # --- START: Data Drift Comparison Chart ---
                        st.subheader("Data Drift Check: Log Returns Distribution")
                        st.caption("Comparison of recent Log Returns vs. previous period to detect distribution shifts.")

                        # Define periods (90 trading days is approx. 4.5 months)
                        DAYS_COMPARE = 90

                        # Data extraction
                        log_returns = df_processed['Log_Returns'].dropna()

                        if len(log_returns) > DAYS_COMPARE * 2:
                            # Split into two periods
                            recent_returns = log_returns.iloc[-DAYS_COMPARE:]
                            previous_returns = log_returns.iloc[-(DAYS_COMPARE * 2):-DAYS_COMPARE]

                            fig_drift, ax_drift = plt.subplots(figsize=(10, 5))

                            # Plot distributions
                            sns.kdeplot(recent_returns, ax=ax_drift, label=f'Recent ({recent_returns.index[0].strftime("%Y-%m-%d")} to Today)', color='red', fill=True, alpha=0.3)
                            sns.kdeplot(previous_returns, ax=ax_drift, label=f'Previous ({previous_returns.index[0].strftime("%Y-%m-%d")} to {previous_returns.index[-1].strftime("%Y-%m-%d")})', color='blue', fill=True, alpha=0.3)

                            ax_drift.set_title(f"Data Drift: Log Returns Distribution Comparison", fontsize=14)
                            ax_drift.set_xlabel("Log Returns")
                            ax_drift.set_ylabel("Density (KDE)")
                            ax_drift.legend()
                            ax_drift.grid(True, which='both', linestyle='--', linewidth=0.5)
                            st.pyplot(fig_drift)

                        else:
                            st.info(f"Not enough data points available (need > {DAYS_COMPARE*2} points) to perform a meaningful drift comparison.")

                        st.markdown("---")
                        # --- END: Data Drift Comparison Chart ---

                        # Historical Accuracy Chart
                        if HISTORICAL_PREDICTIONS is not None:
                            st.subheader("Actual vs. Predicted Price (Historical Test Data)")

                            df_plot = HISTORICAL_PREDICTIONS.tail(120)

                            fig_comp, ax_comp = plt.subplots(figsize=(10, 5))

                            ax_comp.plot(df_plot.index, df_plot['Actual_Price'], label='Actual Price', color='black', alpha=0.7)
                            ax_comp.plot(df_plot.index, df_plot['Predicted_Price_1D'], label='1-Day Predicted', color='red', linestyle='--', alpha=0.7)
                            ax_comp.plot(df_plot.index, df_plot['Predicted_Price_30D'], label='30-Day Predicted', color='green', linestyle=':', alpha=0.7)

                            ax_comp.set_title(f"Model Performance: Actual vs. Predicted Prices (Last 120 Historical Predictions)", fontsize=14)
                            ax_comp.set_xlabel("Date")
                            ax_comp.set_ylabel("Price (INR)")
                            ax_comp.legend()
                            ax_comp.grid(True, which='both', linestyle='--', linewidth=0.5)
                            st.pyplot(fig_comp)
                        else:
                            st.info("Historical predictions data not available for plotting.")

                    with tab3: # Data Analysis
                        st.subheader("Historical Log Returns Distribution")
                        fig_hist, ax_hist = plt.subplots(figsize=(10, 5))

                        sns.histplot(df_processed['Log_Returns'].dropna(), bins=50, kde=True, ax=ax_hist, color='skyblue', edgecolor='black')

                        ax_hist.axvline(df_processed['Log_Returns'].iloc[-1].item(), color='red', linestyle='--', label='Last Log Return')
                        ax_hist.set_title(f"{ticker} Daily Log Returns Distribution", fontsize=14)
                        ax_hist.set_xlabel("Log Returns")
                        ax_hist.set_ylabel("Frequency")
                        ax_hist.legend()
                        ax_hist.grid(True, which='both', linestyle='--', linewidth=0.5)
                        st.pyplot(fig_hist)


                else:
                    st.error("Not enough data to calculate all required technical indicators (e.g., due to missing values).")

        except Exception as e:
            st.error(f"An error occurred: {e}")

else:
    st.info("Enter a ticker symbol and click 'Generate Forecast' to begin.")
