import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import load_model
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

st.set_page_config(page_title="Market Trend Analysis", layout="centered")
st.title("📊 Market Trend Analysis")

FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Adj_Close", "Volume",
    "MA_20", "Daily_Return", "Volatility_20", "MA_50",
    "High_Low_Range", "Momentum_5", "Close_Lag_1", "Close_Lag_7"
]

@st.cache_resource
def load_all_models():
    lstm_model = tf.keras.models.load_model(
    'lstm_marketmodel.keras',
    )   
    scaler = joblib.load(
        "scaler.pkl"
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        "intent_nlp_model"
    )
    nlp_model = DistilBertForSequenceClassification.from_pretrained(
        "intent_nlp_model"
    )
    nlp_model.eval()

    return lstm_model, scaler, tokenizer, nlp_model


lstm_model, scaler, tokenizer, nlp_model = load_all_models()

@st.cache_data
def load_data():
    df = pd.read_csv(
        "yahoo_clean_final.csv"
    )
    df["Date"] = pd.to_datetime(df["Date"])

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    return df


df = load_data()

ID2LABEL = {
    0: "mean", 1: "median", 2: "mode",
    3: "stats", 4: "trend",
    5: "volatility", 6: "forecast", 7: "year"
}


def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = nlp_model(**inputs)
    return ID2LABEL[torch.argmax(outputs.logits, dim=1).item()]


def refine_stat_intent(query, intent):
    q = query.lower()
    if "mean" in q or "average" in q:
        return "mean"
    if "median" in q:
        return "median"
    if "mode" in q or "frequent" in q:
        return "mode"
    return intent


def insight_engine(df):
    close = df["Close"]

    stats = {
        "mean": close.mean(),
        "median": close.median(),
        "mode": close.mode()[0],
    }

    recent_pct = (close.iloc[-1] - close.iloc[-30]) / close.iloc[-30]

    if recent_pct > 0.03:
        trend_text = "📈 **Upward trend** observed in recent period."
    elif recent_pct < -0.03:
        trend_text = "📉 **Downward trend** expected in near term."
    else:
        trend_text = "➡️ **Sideways movement**, market consolidation."

    vol = close.pct_change().std()
    volatility_text = (
        "⚠️ **High volatility**, increased risk."
        if vol > 0.025 else
        "✅ **Low volatility**, stable conditions."
    )

    year_outlook = (
        "The model is trained for short-term forecasting. "
        "Long-term yearly prediction is uncertain and directional."
    )

    return stats, trend_text, volatility_text, year_outlook


def plot_mean(df):
    fig, ax = plt.subplots()
    ax.plot(df["Date"], df["Close"])
    ax.axhline(df["Close"].mean(), linestyle="--", color="red")
    ax.set_title("Mean Price")
    return fig


def plot_median(df):
    fig, ax = plt.subplots()
    ax.plot(df["Date"], df["Close"])
    ax.axhline(df["Close"].median(), linestyle="--", color="green")
    ax.set_title("Median Price")
    return fig


def plot_mode(df):
    fig, ax = plt.subplots()
    ax.plot(df["Date"], df["Close"])
    ax.axhline(df["Close"].mode()[0], linestyle="--", color="purple")
    ax.set_title("Mode Price")
    return fig


def plot_trend(df):
    fig, ax = plt.subplots()
    ax.plot(df["Date"], df["Close"])
    ax.set_title("Market Trend")
    return fig


def plot_volatility(df):
    fig, ax = plt.subplots()
    ax.plot(df["Date"], df["Close"].pct_change())
    ax.axhline(0, linestyle="--")
    ax.set_title("Market Volatility")
    return fig


def prepare_lstm_input(df, scaler, lookback=60):
    scaled = scaler.transform(df[FEATURE_COLS].values)
    return np.expand_dims(scaled[-lookback:], axis=0)


def predict_next_price(df, model, scaler):
    X = prepare_lstm_input(df, scaler)
    scaled_pred = model.predict(X, verbose=0)

    dummy = np.zeros((1, len(FEATURE_COLS)))
    idx = FEATURE_COLS.index("Close")
    dummy[0, idx] = scaled_pred[0][0]

    price = scaler.inverse_transform(dummy)[0][idx]
    last = df["Close"].iloc[-1]

    direction = "📈 Upward" if price > last else "📉 Downward"
    return f"Predicted next close: **{price:.2f}** ({direction})", price


def plot_prediction(df, price):
    fig, ax = plt.subplots()
    ax.plot(df["Date"].iloc[-30:], df["Close"].iloc[-30:])
    ax.scatter(df["Date"].iloc[-1] + pd.Timedelta(days=1), price, color="red")
    ax.set_title("Next-Step Forecast")
    return fig

def chatbot_answer(query):
    intent = refine_stat_intent(query, predict_intent(query))
    stats, trend, volatility, year_outlook = insight_engine(df)

    q = query.lower()
    want_graph = any(w in q for w in ["graph", "plot", "chart", "show"])

    if intent == "mean":
        return f"Mean Close Price: {stats['mean']:.2f}", plot_mean(df) if want_graph else None

    if intent == "median":
        return f"Median Close Price: {stats['median']:.2f}", plot_median(df) if want_graph else None

    if intent == "mode":
        return f"Mode Close Price: {stats['mode']:.2f}", plot_mode(df) if want_graph else None

    if intent == "trend":
        return trend, plot_trend(df) if want_graph else None

    if intent == "volatility":
        return volatility, plot_volatility(df)

    if intent == "forecast":
        text, price = predict_next_price(df, lstm_model, scaler)
        return text, plot_prediction(df, price) if want_graph else None

    if intent == "year":
        return year_outlook, None

    return "I can help with trends, statistics, volatility, and forecasts.", None


st.markdown("""
This dashboard analyzes historical market data using deep learning (LSTM)
and NLP to provide insights, trends, volatility analysis, and forecasts.
""")

with st.expander("📂 Original Dataset"):
    st.dataframe(
        pd.read_csv(
            "yahoo_set.csv"
        ).head(100)
    )

with st.expander("🧹 Processed Dataset"):
    st.dataframe(df.head(100))

st.subheader("📈 Insights")

stats, trend_text, volatility_text, year_outlook = insight_engine(df)

choice = st.selectbox(
    "Select Insight",
    ["Mean", "Median", "Mode", "Trend", "Volatility", "Year Outlook"]
)

if choice == "Mean":
    st.write(stats["mean"])
elif choice == "Median":
    st.write(stats["median"])
elif choice == "Mode":
    st.write(stats["mode"])
elif choice == "Trend":
    st.markdown(trend_text)
elif choice == "Volatility":
    st.markdown(volatility_text)
elif choice == "Year Outlook":
    st.markdown(year_outlook)

st.subheader("📊 Visualizations")

graph = st.selectbox(
    "Select Graph",
    ["None", "Mean", "Median", "Mode", "Trend", "Volatility", "Forecast"]
)

if graph == "Mean":
    st.pyplot(plot_mean(df))
elif graph == "Median":
    st.pyplot(plot_median(df))
elif graph == "Mode":
    st.pyplot(plot_mode(df))
elif graph == "Trend":
    st.pyplot(plot_trend(df))
elif graph == "Volatility":
    st.pyplot(plot_volatility(df))
elif graph == "Forecast":
    t, p = predict_next_price(df, lstm_model, scaler)
    st.markdown(t)
    st.pyplot(plot_prediction(df, p))

st.subheader("🤖 Market Chatbot")

user_query = st.text_input("Ask a question")

if user_query:
    answer, fig = chatbot_answer(user_query)
    st.markdown(answer)
    if fig is not None:
        st.pyplot(fig)
