# Unique marker to force reload - 2025-05-05-004
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import requests
import time
import logging
import socket
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Initialize session state at the top
if "user_email" not in st.session_state:
    logger.info("Initializing user_email in session state to None")
    st.session_state.user_email = None
print(f"Initial user_email: {st.session_state.user_email}")
if "stock_lookups" not in st.session_state:
    logger.info("Initializing stock_lookups in session state to 0")
    st.session_state.stock_lookups = 0
if "cache" not in st.session_state:
    st.session_state.cache = {}
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "theme_styles" not in st.session_state:
    st.session_state.theme_styles = {
        "bg": "#FFFFFF",
        "text": "#1F2937",
        "plot_bg": "rgba(240, 240, 240, 0.5)",
        "grid": "rgba(200, 200, 200, 0.5)",
        "line": "#00A8E8",
        "sma20": "#FF6F61",
        "sma50": "#6B7280",
        "sma200": "#34D399",
        "bar_colors": ["#00C4B4", "#FF6F61", "#F4A261", "#34D399", "#6B7280", "#A78BFA", "#EC4899", "#EF4444"],
        "calc_header": "#00A8E8"
    }
if "api_requests_made" not in st.session_state:
    st.session_state.api_requests_made = 0
if "last_request_date" not in st.session_state:
    st.session_state.last_request_date = datetime.now().date()

# Clear cache on startup to avoid stale data
st.session_state.cache = {}

# Reset API request counter
st.session_state.api_requests_made = 0
st.session_state.last_request_date = datetime.now().date()

# Reddit API setup
reddit = praw.Reddit(
    client_id="V3rxmA_qYIzBNTNW79LWIg",
    client_secret="xKBA3Nx7f7VQS0fXnOgmJhYZOmGasA",
    user_agent="python:EquityScopeBot:v1.0"
)

# VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = "4OEV4A0EMNBMJP78"

# Test network connectivity to Alpha Vantage
def test_connectivity():
    try:
        socket.create_connection(("www.alphavantage.co", 443), timeout=5)
        return True, "Successfully connected to Alpha Vantage."
    except Exception as e:
        return False, f"Failed to connect to Alpha Vantage: {str(e)}"

# Validate Alpha Vantage API key
def validate_alpha_vantage_api_key():
    test_url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol=IBM&apikey={ALPHA_VANTAGE_API_KEY}"
    try:
        response = requests.get(test_url, timeout=30)
        logger.info(f"Alpha Vantage API key validation response: {response.status_code}")
        logger.debug(f"API key validation response content: {response.text}")
        if response.status_code == 200:
            data = response.json()
            if "Note" in data or "Information" in data:
                return False, "Rate limit exceeded or free tier limit reached."
            if "Error Message" in data or not data:
                return False, "Invalid Alpha Vantage API key."
            return True, "API key is valid."
        else:
            return False, f"HTTP Error {response.status_code}: Unable to validate API key."
    except Exception as e:
        logger.error(f"Error validating API key: {str(e)}")
        return False, f"Error validating API key: {str(e)}"

# Check connectivity and API key at startup with error handling
try:
    conn_success, conn_message = test_connectivity()
    if not conn_success:
        st.error(f"Connectivity Issue: {conn_message}")
        st.markdown("**Next Steps**: Ensure you're connected to the internet, disable any VPN/firewall, or fix DNS issues (e.g., use Google's DNS: 8.8.8.8).")
except Exception as e:
    st.error(f"Error checking connectivity: {str(e)}")

try:
    is_valid, validation_message = validate_alpha_vantage_api_key()
    if not is_valid:
        st.error(f"Alpha Vantage API Key Issue: {validation_message}")
    else:
        st.success("Alpha Vantage API key validated successfully!")
except Exception as e:
    st.error(f"Error validating API key: {str(e)}")
# Retry decorator with exponential backoff
def retry_with_backoff(func, max_attempts=5, initial_delay=5, backoff_factor=2):
    def wrapper(*args, **kwargs):
        delay = initial_delay
        for attempt in range(max_attempts):
            try:
                result = func(*args, **kwargs)
                if result is None:
                    raise ValueError("API returned None")
                return result
            except Exception as e:
                error_msg = str(e).lower()
                logger.warning(f"Attempt {attempt+1} failed for {func.__name__}: {str(e)}")
                if "429" in error_msg or "too many requests" in error_msg:
                    logger.info(f"Rate limit hit for {func.__name__}, waiting {delay} seconds...")
                elif "json" in error_msg or "expecting value" in error_msg:
                    logger.info(f"JSON parsing issue for {func.__name__}, possibly empty response...")
                if attempt == max_attempts - 1:
                    logger.error(f"Max attempts reached for {func.__name__}: {str(e)}")
                    return None, str(e)
                time.sleep(delay)
                delay *= backoff_factor
        return None, "Max retries exceeded"
    return wrapper

# Alpha Vantage request counter
def check_alpha_vantage_limit():
    today = datetime.now().date()
    if st.session_state.last_request_date != today:
        st.session_state.api_requests_made = 0
        st.session_state.last_request_date = today
    if st.session_state.api_requests_made >= 25:  # Free tier limit
        return False, "Daily limit of 25 requests reached for Alpha Vantage (free tier)."
    return True, ""

# Cache stock info
@retry_with_backoff
def get_stock_info_alpha_vantage(ticker):
    can_proceed, limit_message = check_alpha_vantage_limit()
    if not can_proceed:
        return None, limit_message
    logger.info(f"Fetching stock info for {ticker} from Alpha Vantage")
    try:
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url, timeout=30)
        logger.debug(f"Stock info response status for {ticker}: {response.status_code}")
        logger.debug(f"Stock info response content: {response.text}")
        st.session_state.api_requests_made += 1
        if response.status_code == 200:
            data = response.json()
            if "Symbol" in data:
                return data, None
            elif "Note" in data or "Information" in data:
                return None, "Alpha Vantage rate limit exceeded or free tier limit reached."
            else:
                return None, "No data returned from Alpha Vantage."
        else:
            return None, f"Alpha Vantage HTTP Error {response.status_code}"
    except Exception as e:
        logger.error(f"Alpha Vantage error for {ticker}: {str(e)}")
        return None, f"Alpha Vantage error: {str(e)}"

@retry_with_backoff
def get_stock_info_yfinance(ticker):
    logger.info(f"Fetching stock info for {ticker} from yfinance")
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info or "symbol" not in info:
            return None, "No data returned from yfinance."
        return info, None
    except Exception as e:
        logger.error(f"yfinance error for {ticker}: {str(e)}")
        return None, f"yfinance error: {str(e)}"

def get_stock_info(ticker):
    cache_key = f"info_{ticker}"
    if cache_key not in st.session_state.cache:
        # Try Alpha Vantage first
        info, error = get_stock_info_alpha_vantage(ticker)
        if info:
            st.session_state.cache[cache_key] = info
        else:
            logger.warning(f"Alpha Vantage failed for {ticker}: {error}")
            # Fallback to yfinance
            info, error = get_stock_info_yfinance(ticker)
            if info:
                st.session_state.cache[cache_key] = info
            else:
                logger.error(f"yfinance also failed for {ticker}: {error}")
                return None, f"Could not fetch stock info: {error}"
    return st.session_state.cache[cache_key], None

# Cache stock history
@retry_with_backoff
def get_stock_history_alpha_vantage(ticker, period):
    can_proceed, limit_message = check_alpha_vantage_limit()
    if not can_proceed:
        return None, limit_message
    logger.info(f"Fetching stock history for {ticker} - {period} from Alpha Vantage")
    try:
        time_series_map = {
            "1d": ("TIME_SERIES_INTRADAY", "5min", "1D"),
            "5d": ("TIME_SERIES_DAILY", None, "1W"),
            "1mo": ("TIME_SERIES_DAILY", None, "1M"),
            "6mo": ("TIME_SERIES_DAILY", None, "6M"),
            "1y": ("TIME_SERIES_DAILY", None, "1Y"),
            "5y": ("TIME_SERIES_WEEKLY", None, "5Y"),
            "10y": ("TIME_SERIES_WEEKLY", None, "10Y"),
            "max": ("TIME_SERIES_WEEKLY", None, "All")
        }
        function, interval, display_period = time_series_map.get(period, ("TIME_SERIES_DAILY", None, period))
        if interval:
            url = f"https://www.alphavantage.co/query?function={function}&symbol={ticker}&interval={interval}&apikey={ALPHA_VANTAGE_API_KEY}"
        else:
            url = f"https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url, timeout=30)
        logger.debug(f"Stock history response status for {ticker} ({period}): {response.status_code}")
        logger.debug(f"Stock history response content: {response.text}")
        st.session_state.api_requests_made += 1
        if response.status_code == 200:
            data = response.json()
            time_series_key = f"Time Series ({interval})" if interval else "Time Series" if function == "TIME_SERIES_DAILY" else "Weekly Time Series"
            if time_series_key in data:
                df = pd.DataFrame.from_dict(data[time_series_key], orient="index")
                df.index = pd.to_datetime(df.index)
                df = df.rename(columns={
                    "4. close": "Close",
                    "1. open": "Open",
                    "2. high": "High",
                    "3. low": "Low",
                    "5. volume": "Volume"
                })
                df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
                return df.sort_index(), None
            elif "Note" in data or "Information" in data:
                return None, "Alpha Vantage rate limit exceeded or free tier limit reached."
            else:
                return None, "No data returned from Alpha Vantage."
        else:
            return None, f"Alpha Vantage HTTP Error {response.status_code}"
    except Exception as e:
        logger.error(f"Alpha Vantage history error for {ticker}: {str(e)}")
        return None, f"Alpha Vantage error: {str(e)}"

@retry_with_backoff
def get_stock_history_yfinance(ticker, period):
    logger.info(f"Fetching stock history for {ticker} - {period} from yfinance")
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            return None, "No data returned from yfinance."
        if "Adj Close" in hist.columns:
            hist = hist.drop(columns=["Adj Close"])
        return hist, None
    except Exception as e:
        logger.error(f"yfinance history error for {ticker}: {str(e)}")
        return None, f"yfinance error: {str(e)}"

def get_stock_history(ticker, period):
    cache_key = f"history_{ticker}_{period}"
    if cache_key not in st.session_state.cache:
        # Try Alpha Vantage first
        hist, error = get_stock_history_alpha_vantage(ticker, period)
        if hist is not None:
            st.session_state.cache[cache_key] = hist
        else:
            logger.warning(f"Alpha Vantage failed for {ticker} history: {error}")
            # Fallback to yfinance
            hist, error = get_stock_history_yfinance(ticker, period)
            if hist is not None:
                st.session_state.cache[cache_key] = hist
            else:
                logger.error(f"yfinance also failed for {ticker} history: {error}")
                return None, f"Could not fetch stock history: {error}"
    return st.session_state.cache[cache_key], None
# Streamlit page setup
st.set_page_config(
    page_title="EquityScope: Stock Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)
st.title("📈 EquityScope: Stock Analyzer")

# Theme toggle
st.sidebar.title("Settings")
theme = st.sidebar.radio("Select Theme:", ["Light", "Dark"], index=0 if st.session_state.theme == "light" else 1)
st.session_state.theme = theme.lower()

# Define theme styles
light_theme = {
    "bg": "#FFFFFF",
    "text": "#111827",
    "plot_bg": "rgba(240, 240, 240, 0.5)",
    "grid": "rgba(150, 150, 150, 0.5)",
    "line": "#00A8E8",
    "sma20": "#FF6F61",
    "sma50": "#6B7280",
    "sma200": "#34D399",
    "bar_colors": ["#00C4B4", "#FF6F61", "#F4A261", "#34D399", "#6B7280", "#A78BFA", "#EC4899", "#EF4444"],
    "calc_header": "#00A8E8"
}
dark_theme = {
    "bg": "#1F2937",
    "text": "#F3F4F6",
    "plot_bg": "rgba(31, 41, 55, 0.8)",
    "grid": "rgba(107, 114, 128, 0.5)",
    "line": "#60A5FA",
    "sma20": "#F87171",
    "sma50": "#9CA3AF",
    "sma200": "#34D399",
    "bar_colors": ["#2DD4BF", "#F87171", "#FBBF24", "#34D399", "#9CA3AF", "#C4B5FD", "#F472B6", "#F87171"],
    "calc_header": "#60A5FA"
}
st.session_state.theme_styles = light_theme if st.session_state.theme == "light" else dark_theme

# Apply CSS
st.markdown(f"""
<style>
    .stApp {{
        background-color: {st.session_state.theme_styles['bg']} !important;
        color: {st.session_state.theme_styles['text']} !important;
    }}
    .stMarkdown, .stRadio > label, .stAlert, .company-info, .description, .metric-value, .calc-header, .st-expander {{
        color: {st.session_state.theme_styles['text']} !important;
        opacity: 1 !important;
    }}
    .calc-header {{
        color: {st.session_state.theme_styles['calc_header']} !important;
        font-size: 18px;
        font-weight: bold;
        margin-top: 10px;
    }}
    [data-testid="stTextInput"] label, [data-testid="stTextInput"] div p {{
        color: {st.session_state.theme_styles['text']} !important;
        opacity: 1 !important;
    }}
    [data-testid="stSidebar"] {{
        background-color: { '#F9FAFB' if st.session_state.theme == 'light' else '#374151' } !important;
    }}
    [data-testid="stSidebar"] div, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] h1 {{
        color: {st.session_state.theme_styles['text']} !important;
        opacity: 1 !important;
    }}
    [data-testid="stSidebar"] .stRadio > label, [data-testid="stSidebar"] .stRadio > label p, [data-testid="stSidebar"] .stRadio > div p {{
        color: {st.session_state.theme_styles['text']} !important;
        opacity: 1 !important;
    }}
    [data-testid="stDataFrame"] a {{
        color: {st.session_state.theme_styles['text']} !important;
        text-decoration: underline !important;
    }}
    .stApp[style*="background-color: #FFFFFF"] .js-plotly-plot .plotly .ticktext {{
        fill: #000000 !important;
        color: #000000 !important;
    }}
    .stApp[style*="background-color: #FFFFFF"] .js-plotly-plot .plotly .g-xtitle,
    .stApp[style*="background-color: #FFFFFF"] .js-plotly-plot .plotly .g-ytitle {{
        fill: #000000 !important;
        color: #000000 !important;
    }}
</style>
""", unsafe_allow_html=True)

# Debug Section
st.sidebar.subheader("Debug Info")
if st.sidebar.checkbox("Show Debug Logs"):
    st.sidebar.markdown("Check your terminal for detailed logs.")
st.sidebar.markdown(f"Alpha Vantage Requests Made Today: {st.session_state.api_requests_made}/25")

# Manual cache reset
if st.sidebar.button("Clear Cache"):
    st.session_state.cache = {}
    st.session_state.api_requests_made = 0
    st.experimental_rerun()

# Company Info
st.subheader("🔍 Company Information")
stock_ticker = st.text_input("Enter a stock ticker (e.g., AAPL):", value="AAPL").strip().upper()

def market_cap_display(market_cap):
    if isinstance(market_cap, (int, float)):
        if market_cap >= 1_000_000_000_000:
            return f"${market_cap / 1_000_000_000_000:.2f}T"
        elif market_cap >= 1_000_000_000:
            return f"${market_cap / 1_000_000_000:.2f}B"
        else:
            return f"${market_cap / 1_000_000:.2f}M"
    return "N/A"

if stock_ticker:
    logger.info(f"Attempting stock lookup for {stock_ticker}, current lookups: {st.session_state.stock_lookups}")
    if st.session_state.stock_lookups >= 3:
        st.error("You have reached the limit of 3 free stock lookups. Please check back later or contact support for more options.")
        logger.warning("Lookup limit reached during stock query")
    else:
        with st.spinner("Fetching company info..."):
            stock_info, error = get_stock_info(stock_ticker)
            if stock_info:
                st.session_state.stock_lookups += 1
                logger.info(f"Lookup successful, incremented to: {st.session_state.stock_lookups}")
                company_name = stock_info.get("Name", stock_info.get("longName", "N/A"))
                sector = stock_info.get("Sector", stock_info.get("sector", "N/A"))
                industry = stock_info.get("Industry", stock_info.get("industry", "N/A"))
                market_cap = float(stock_info.get("MarketCapitalization", stock_info.get("marketCap", 0)))
                summary = stock_info.get("Description", stock_info.get("longBusinessSummary", "No description available."))
                website = stock_info.get("Website", stock_info.get("website", "N/A"))
                st.markdown(
                    f'<p class="company-info"><strong>Company:</strong> {company_name} | '
                    f'<strong>Sector:</strong> {sector} | '
                    f'<strong>Industry:</strong> {industry} | '
                    f'<strong>Market Cap:</strong> {market_cap_display(market_cap)} | '
                    f'<strong>Website:</strong> <a href="{website}" target="_blank">{website}</a></p>',
                    unsafe_allow_html=True
                )
                st.markdown(f'<p class="description">{summary}</p>', unsafe_allow_html=True)
            else:
                st.error(f"Could not fetch company info for {stock_ticker}: {error}")
                st.markdown("**Next Steps**: Check the terminal logs for detailed error messages. Ensure you're connected to the internet, or try clearing the cache using the button in the sidebar.")
# Key Metrics
st.subheader("📊 Key Metrics")
if stock_ticker:
    with st.spinner("Fetching key metrics..."):
        stock_info = get_stock_info(stock_ticker)
        if stock_info:
            metrics = {}
            metrics["P/E Ratio"] = float(stock_info.get("PERatio", stock_info.get("trailingPE", "N/A")))
            metrics["P/B Ratio"] = float(stock_info.get("PriceToBookRatio", stock_info.get("priceToBook", "N/A")))
            metrics["Dividend Yield"] = float(stock_info.get("DividendYield", stock_info.get("dividendYield", 0))) * 100
            metrics["Beta"] = float(stock_info.get("Beta", stock_info.get("beta", "N/A")))
            metrics["ROE"] = float(stock_info.get("ReturnOnEquityTTM", stock_info.get("returnOnEquity", "N/A"))) * 100
            metrics["Debt/Equity"] = float(stock_info.get("DebtToEquity", stock_info.get("debtToEquity", "N/A")))
            metrics["P/S Ratio"] = float(stock_info.get("PriceToSalesRatioTTM", stock_info.get("priceToSalesTrailing12Months", "N/A")))
            metrics_display = []
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not pd.isna(value):
                    if key in ["Dividend Yield", "ROE"]:
                        metrics_display.append(f"{key}: {value:.2f}%")
                    else:
                        metrics_display.append(f"{key}: {value:.2f}")
                else:
                    metrics_display.append(f"{key}: N/A")
            st.markdown(
                f'<p class="metric-value">{" | ".join(metrics_display)}</p>',
                unsafe_allow_html=True
            )
        else:
            st.warning("Could not fetch key metrics.")

# Price History
st.subheader("📈 Price History")
time_frame_options = ["1D", "1W", "1M", "6M", "1Y", "5Y", "10Y", "All"]
time_frame_map = {"1D": "1d", "1W": "5d", "1M": "1mo", "6M": "6mo", "1Y": "1y", "5Y": "5y", "10Y": "10y", "All": "max"}
selected_time_frame = st.selectbox("Select time frame:", time_frame_options, index=4)
selected_period = time_frame_map[selected_time_frame]

if stock_ticker:
    with st.spinner(f"Fetching price history for {selected_time_frame}..."):
        hist = get_stock_history(stock_ticker, period=selected_period)
        if hist is not None and not hist.empty:
            axis_text_color = "#000000" if st.session_state.theme == "light" else st.session_state.theme_styles['text']
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], mode="lines", name="Close Price", line=dict(color=st.session_state.theme_styles["line"])))
            fig.add_trace(go.Bar(x=hist.index, y=hist["Volume"], name="Volume", yaxis="y2", opacity=0.3, marker_color=st.session_state.theme_styles["sma50"]))
            fig.update_layout(
                title=dict(text=f"{stock_ticker} Price and Volume ({selected_time_frame})", font=dict(color=st.session_state.theme_styles["text"])),
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                yaxis2=dict(title="Volume", overlaying="y", side="right", titlefont=dict(color=axis_text_color), tickfont=dict(color=axis_text_color)),
                plot_bgcolor=st.session_state.theme_styles["plot_bg"],
                paper_bgcolor=st.session_state.theme_styles["bg"],
                font=dict(family="Arial", size=12, color=axis_text_color),
                legend=dict(font=dict(color=axis_text_color)),
                xaxis=dict(
                    title=dict(font=dict(color=axis_text_color)),
                    tickfont=dict(family="Arial", size=12, color=axis_text_color),
                    gridcolor=st.session_state.theme_styles["grid"]
                ),
                yaxis=dict(
                    title=dict(font=dict(color=axis_text_color)),
                    tickfont=dict(family="Arial", size=12, color=axis_text_color),
                    gridcolor=st.session_state.theme_styles["grid"]
                )
            )
            fig.update_xaxes(tickfont=dict(family="Arial", size=12, color=axis_text_color))
            fig.update_yaxes(tickfont=dict(family="Arial", size=12, color=axis_text_color))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Could not fetch price history for {stock_ticker}. Check logs for details.")

# Moving Averages
st.subheader("📈 Moving Averages")
if selected_time_frame in ["1D", "1W", "1M"]:
    st.warning("Note: 50-day and 200-day SMAs may be less reliable for short time frames.")
if stock_ticker:
    with st.spinner(f"Calculating moving averages..."):
        hist = get_stock_history(stock_ticker, period=selected_period)
        if hist is not None and not hist.empty:
            sma_20 = hist["Close"].rolling(window=20).mean()
            sma_50 = hist["Close"].rolling(window=50).mean()
            sma_200 = hist["Close"].rolling(window=200).mean()
            axis_text_color = "#000000" if st.session_state.theme == "light" else st.session_state.theme_styles['text']
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], mode="lines", name="Close Price", line=dict(color=st.session_state.theme_styles["line"])))
            fig.add_trace(go.Scatter(x=hist.index, y=sma_20, mode="lines", name="20-day SMA", line=dict(color=st.session_state.theme_styles["sma20"])))
            fig.add_trace(go.Scatter(x=hist.index, y=sma_50, mode="lines", name="50-day SMA", line=dict(color=st.session_state.theme_styles["sma50"])))
            fig.add_trace(go.Scatter(x=hist.index, y=sma_200, mode="lines", name="200-day SMA", line=dict(color=st.session_state.theme_styles["sma200"])))
            fig.update_layout(
                title=dict(text=f"{stock_ticker} Moving Averages ({selected_time_frame})", font=dict(color=st.session_state.theme_styles["text"])),
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                showlegend=True,
                legend=dict(font=dict(color=axis_text_color)),
                plot_bgcolor=st.session_state.theme_styles["plot_bg"],
                paper_bgcolor=st.session_state.theme_styles["bg"],
                font=dict(family="Arial", size=12, color=axis_text_color),
                xaxis=dict(
                    title=dict(font=dict(color=axis_text_color)),
                    tickfont=dict(family="Arial", size=12, color=axis_text_color),
                    gridcolor=st.session_state.theme_styles["grid"]
                ),
                yaxis=dict(
                    title=dict(font=dict(color=axis_text_color)),
                    tickfont=dict(family="Arial", size=12, color=axis_text_color),
                    gridcolor=st.session_state.theme_styles["grid"]
                )
            )
            fig.update_xaxes(tickfont=dict(family="Arial", size=12, color=axis_text_color))
            fig.update_yaxes(tickfont=dict(family="Arial", size=12, color=axis_text_color))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Moving Averages Explanation**: SMAs smooth price data to identify trends. A golden cross (50-day SMA crossing above 200-day SMA) is bullish, while a death cross (50-day SMA crossing below 200-day SMA) is bearish.")
        else:
            st.warning(f"Could not fetch price history for moving averages.")
# Technical Indicators Graph
st.subheader("📉 Technical Indicators Graph")
if selected_time_frame in ["1D", "1W", "1M"]:
    st.warning("Note: Bollinger Bands and MACD may be less reliable for short time frames.")
if stock_ticker:
    with st.spinner(f"Calculating technical indicators..."):
        hist = get_stock_history(stock_ticker, period=selected_period)
        if hist is not None and not hist.empty:
            window = 20
            sma = hist["Close"].rolling(window=window).mean()
            std = hist["Close"].rolling(window=window).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            exp12 = hist["Close"].ewm(span=12, adjust=False).mean()
            exp26 = hist["Close"].ewm(span=26, adjust=False).mean()
            macd = exp12 - exp26
            signal = macd.ewm(span=9, adjust=False).mean()
            histogram = macd - signal
            axis_text_color = "#000000" if st.session_state.theme == "light" else st.session_state.theme_styles['text']
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=(f"{stock_ticker} Price with Bollinger Bands", "MACD"), row_heights=[0.7, 0.3])
            fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], mode="lines", name="Close Price", line=dict(color=st.session_state.theme_styles["line"])), row=1, col=1)
            fig.add_trace(go.Scatter(x=hatmospheric pressureist.index, y=upper_band, mode="lines", name="Upper Band", line=dict(color=st.session_state.theme_styles["sma50"], dash="dash")), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=lower_band, mode="lines", name="Lower Band", line=dict(color=st.session_state.theme_styles["sma50"], dash="dash")), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=macd, mode="lines", name="MACD", line=dict(color=st.session_state.theme_styles["line"])), row=2, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=signal, mode="lines", name="Signal Line", line=dict(color=st.session_state.theme_styles["sma20"])), row=2, col=1)
            fig.add_trace(go.Bar(x=hist.index, y=histogram, name="Histogram", marker_color=st.session_state.theme_styles["sma50"]), row=2, col=1)
            fig.update_layout(
                height=600,
                title=dict(text=f"{stock_ticker} Technical Indicators ({selected_time_frame})", font=dict(color=st.session_state.theme_styles["text"])),
                showlegend=True,
                legend=dict(font=dict(color=axis_text_color)),
                xaxis2_title="Date",
                yaxis_title="Price (USD)",
                yaxis2_title="MACD",
                plot_bgcolor=st.session_state.theme_styles["plot_bg"],
                paper_bgcolor=st.session_state.theme_styles["bg"],
                font=dict(family="Arial", size=12, color=axis_text_color),
                xaxis=dict(
                    title=dict(font=dict(color=axis_text_color)),
                    tickfont=dict(family="Arial", size=12, color=axis_text_color),
                    gridcolor=st.session_state.theme_styles["grid"]
                ),
                yaxis=dict(
                    title=dict(font=dict(color=axis_text_color)),
                    tickfont=dict(family="Arial", size=12, color=axis_text_color),
                    gridcolor=st.session_state.theme_styles["grid"]
                ),
                xaxis2=dict(
                    title=dict(font=dict(color=axis_text_color)),
                    tickfont=dict(family="Arial", size=12, color=axis_text_color),
                    gridcolor=st.session_state.theme_styles["grid"]
                ),
                yaxis2=dict(
                    title=dict(font=dict(color=axis_text_color)),
                    tickfont=dict(family="Arial", size=12, color=axis_text_color),
                    gridcolor=st.session_state.theme_styles["grid"]
                )
            )
            fig.update_xaxes(tickfont=dict(family="Arial", size=12, color=axis_text_color))
            fig.update_yaxes(tickfont=dict(family="Arial", size=12, color=axis_text_color))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Bollinger Bands**: Prices near the upper band may indicate overbought conditions; near the lower band, oversold.")
            st.markdown("**MACD**: MACD line crossing above the Signal line is bullish; below is bearish.")
        else:
            st.warning(f"Could not fetch price history for technical indicators.")

# Technical Indicators (RSI)
st.subheader("📉 Technical Indicators")
if selected_time_frame in ["1D", "1W"]:
    st.warning("Note: RSI may be less reliable for very short time frames.")
if stock_ticker:
    with st.spinner(f"Calculating RSI..."):
        hist = get_stock_history(stock_ticker, period=selected_period)
        if hist is not None and not hist.empty:
            delta = hist["Close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            latest_rsi = rsi.iloc[-1] if not rsi.empty else None
            if latest_rsi is not None and not pd.isna(latest_rsi):
                rsi_display = f"{latest_rsi:.2f}"
                rsi_status = "Overbought (sell signal)" if latest_rsi > 70 else "Oversold (buy signal)" if latest_rsi < 30 else "Neutral"
                st.markdown(f'<p class="metric-value">14-Day RSI: {rsi_display} ({rsi_status})</p>', unsafe_allow_html=True)
                st.markdown("**RSI**: Above 70 indicates overbought; below 30 indicates oversold.")
            else:
                st.warning("Could not calculate RSI (insufficient data).")
        else:
            st.warning(f"Could not fetch price history for RSI.")
# Valuation Section
st.subheader("💰 Valuation")
st.markdown("Estimate the fair value of the stock using multiple valuation methods.")
if stock_ticker:
    with st.spinner("Calculating valuations..."):
        stock_info = get_stock_info(stock_ticker)
        stock = yf.Ticker(stock_ticker)
        try:
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow
            financials = stock.financials
        except Exception as e:
            logger.error(f"Failed to fetch financial statements for {stock_ticker}: {str(e)}")
            balance_sheet = None
            cash_flow = None
            financials = None
            st.warning(f"Could not fetch financial statements for {stock_ticker}.")
        if stock_info:
            current_price = float(stock_info.get("PreviousClose", stock_info.get("regularMarketPrice", stock_info.get("regularMarketPreviousClose", 0))))
            eps = float(stock_info.get("EPS", stock_info.get("trailingEps", 0)))
            forward_eps = float(stock_info.get("ForwardEPS", stock_info.get("forwardEps", eps)))
            shares_outstanding = int(stock_info.get("SharesOutstanding", stock_info.get("sharesOutstanding", 0)))
            book_value = float(stock_info.get("BookValue", stock_info.get("bookValue", 0)))
            dividend_rate = float(stock_info.get("DividendRate", stock_info.get("dividendRate", 0)))
            five_year_avg_dividend = float(stock_info.get("FiveYearAvgDividendYield", stock_info.get("fiveYearAvgDividendYield", 0))) / 100
            pe_ratio = float(stock_info.get("PERatio", stock_info.get("trailingPE", 0)))
            growth_rate = min(float(stock_info.get("EarningsGrowth", stock_info.get("earningsGrowth", 0.20))), 0.25)
            beta = float(stock_info.get("Beta", stock_info.get("beta", 1.0)))
            discount_rate = 0.05 + beta * 0.02
            perpetual_growth = 0.03
            total_debt = float(stock_info.get("TotalDebt", stock_info.get("totalDebt", 0)))
            total_cash = float(stock_info.get("TotalCash", stock_info.get("totalCash", 0)))
            revenue = float(stock_info.get("RevenueTTM", stock_info.get("totalRevenue", 0)))
            valuations = []
            calculation_details = []
            # Intrinsic Value (Simple EPS Multiplier)
            try:
                intrinsic_value = eps * 15
                intrinsic_value = f"${intrinsic_value:.2f}" if isinstance(intrinsic_value, (int, float)) else "N/A"
                calc = f"""
                **What We Did** 📊
                - **EPS**: ${eps:.2f}
                - **Multiplier**: 15
                - **Per Share**: {intrinsic_value}
                """
                valuations.append({
                    "Method": "Intrinsic Value (EPS Multiplier)",
                    "Intrinsic Value per Share": intrinsic_value,
                    "Description": "EPS multiplied by 15."
                })
                calculation_details.append(("Intrinsic Value (EPS Multiplier)", calc))
            except:
                valuations.append({
                    "Method": "Intrinsic Value (EPS Multiplier)",
                    "Intrinsic Value per Share": "N/A",
                    "Description": "EPS multiplied by 15."
                })
                calculation_details.append(("Intrinsic Value (EPS Multiplier)", "**Why It’s Missing** 🚫\nError in calculation."))
            # DCF
            try:
                if cash_flow is not None and not cash_flow.empty and forward_eps and shares_outstanding:
                    fcf = forward_eps * shares_outstanding * 0.6
                    growth_rates = [growth_rate * (1 - 0.05 * t) for t in range(5)]
                    cash_flows = [fcf * (1 + g) ** t for t, g in enumerate(growth_rates, 1)]
                    terminal_value = cash_flows[-1] * (1 + perpetual_growth) / (discount_rate - perpetual_growth)
                    cash_flows_with_terminal = cash_flows + [terminal_value]
                    pv_cash_flows = [cf / (1 + discount_rate) ** t for t, cf in enumerate(cash_flows_with_terminal, 1)]
                    enterprise_value = sum(pv_cash_flows)
                    net_debt = total_debt - total_cash
                    equity_value = max(enterprise_value - net_debt, 0)
                    dcf_value = equity_value / shares_outstanding
                    dcf_value = f"${dcf_value:.2f}" if isinstance(dcf_value, (int, float)) else "N/A"
                    calc = f"""
                    **What We Did** 📊
                    - **Free Cash Flow**: ${fcf:,.2f}
                    - **Growth**: {growth_rate*100:.1f}% initially
                    - **Equity Value**: ${equity_value:,.2f}
                    - **Per Share**: {dcf_value}
                    """
                else:
                    dcf_value = "N/A"
                    calc = "**Why It’s Missing** 🚫\nNo cash flow data."
                valuations.append({
                    "Method": "Discounted Cash Flow (DCF)",
                    "Intrinsic Value per Share": dcf_value,
                    "Description": "Discounts future cash flows."
                })
                calculation_details.append(("Discounted Cash Flow (DCF)", calc))
            except:
                valuations.append({
                    "Method": "Discounted Cash Flow (DCF)",
                    "Intrinsic Value per Share": "N/A",
                    "Description": "Discounts future cash flows."
                })
                calculation_details.append(("Discounted Cash Flow (DCF)", "**Why It’s Missing** 🚫\nError in calculation."))
            # DDM
            try:
                if dividend_rate and dividend_rate > 0 and forward_eps:
                    expected_dividend = dividend_rate * (1 + 0.10)
                    ddm_value = expected_dividend / (discount_rate - perpetual_growth)
                    ddm_value = f"${ddm_value:.2f}" if isinstance(ddm_value, (int, float)) else "N/A"
                    calc = f"""
                    **What We Did** 📊
                    - **Dividend**: ${dividend_rate:.2f}
                    - **Per Share**: {ddm_value}
                    """
                else:
                    ddm_value = "N/A"
                    calc = "**Why It’s Missing** 🚫\nNo dividends."
                valuations.append({
                    "Method": "Dividend Discount Model (DDM)",
                    "Intrinsic Value per Share": ddm_value,
                    "Description": "Values stock via dividends."
                })
                calculation_details.append(("Dividend Discount Model (DDM)", calc))
            except:
                valuations.append({
                    "Method": "Dividend Discount Model (DDM)",
                    "Intrinsic Value per Share": "N/A",
                    "Description": "Values stock via dividends."
                })
                calculation_details.append(("Dividend Discount Model (DDM)", "**Why It’s Missing** 🚫\nError in calculation."))
            # RIM
            try:
                if book_value and forward_eps and shares_outstanding:
                    roe = forward_eps / book_value if book_value != 0 else 0
                    retention_ratio = 1 - (dividend_rate / forward_eps if dividend_rate and forward_eps else 0)
                    residual_income = forward_eps - (discount_rate * book_value)
                    rim_value = book_value + (residual_income * retention_ratio * (1 + growth_rate) / (discount_rate - perpetual_growth))
                    rim_value = f"${rim_value:.2f}" if isinstance(rim_value, (int, float)) else "N/A"
                    calc = f"""
                    **What We Did** 📊
                    - **Book Value**: ${book_value:.2f}
                    - **Per Share**: {rim_value}
                    """
                else:
                    rim_value = "N/A"
                    calc = "**Why It’s Missing** 🚫\nNo book value or EPS."
                valuations.append({
                    "Method": "Residual Income Model (RIM)",
                    "Intrinsic Value per Share": rim_value,
                    "Description": "Uses book value and income."
                })
                calculation_details.append(("Residual Income Model (RIM)", calc))
            except:
                valuations.append({
                    "Method": "Residual Income Model (RIM)",
                    "Intrinsic Value per Share": "N/A",
                    "Description": "Uses book value and income."
                })
                calculation_details.append(("Residual Income Model (RIM)", "**Why It’s Missing** 🚫\nError in calculation."))
            # Graham
            try:
                if forward_eps and growth_rate:
                    graham_value = forward_eps * (10 + 2.5 * growth_rate * 100)
                    graham_value = f"${graham_value:.2f}" if isinstance(graham_value, (int, float)) else "N/A"
                    calc = f"""
                    **What We Did** 📊
                    - **EPS**: ${forward_eps:.2f}
                    - **Per Share**: {graham_value}
                    """
                else:
                    graham_value = "N/A"
                    calc = "**Why It’s Missing** 🚫\nNo EPS or growth data."
                valuations.append({
                    "Method": "Graham Method",
                    "Intrinsic Value per Share": graham_value,
                    "Description": "EPS with growth multiplier."
                })
                calculation_details.append(("Graham Method", calc))
            except:
                valuations.append({
                    "Method": "Graham Method",
                    "Intrinsic Value per Share": "N/A",
                    "Description": "EPS with growth multiplier."
                })
                calculation_details.append(("Graham Method", "**Why It’s Missing** 🚫\nError in calculation."))
            # Comps (using P/E and P/S)
            try:
                if pe_ratio and revenue and shares_outstanding:
                    industry_pe = 25.0  # Static average for simplicity
                    comps_pe_value = eps * industry_pe
                    industry_ps = 5.0  # Static average for simplicity
                    comps_ps_value = (revenue * industry_ps) / shares_outstanding
                    comps_value = (comps_pe_value + comps_ps_value) / 2
                    comps_value = f"${comps_value:.2f}" if isinstance(comps_value, (int, float)) else "N/A"
                    calc = f"""
                    **What We Did** 📊
                    - **P/E Value**: ${comps_pe_value:.2f}
                    - **P/S Value**: ${comps_ps_value:.2f}
                    - **Average**: {comps_value}
                    """
                else:
                    comps_value = "N/A"
                    calc = "**Why It’s Missing** 🚫\nNo P/E, revenue, or shares data."
                valuations.append({
                    "Method": "Comparable Company Analysis (Comps)",
                    "Intrinsic Value per Share": comps_value,
                    "Description": "Uses industry P/E and P/S ratios."
                })
                calculation_details.append(("Comparable Company Analysis (Comps)", calc))
            except:
                valuations.append({
                    "Method": "Comparable Company Analysis (Comps)",
                    "Intrinsic Value per Share": "N/A",
                    "Description": "Uses industry P/E and P/S ratios."
                })
                calculation_details.append(("Comparable Company Analysis (Comps)", "**Why It’s Missing** 🚫\nError in calculation."))
            # Valuation Table and Chart
            valuation_df = pd.DataFrame(valuations)
            st.markdown(f"**Current Market Price**: ${current_price:.2f}")
            st.markdown("### Valuation Estimates")
            st.dataframe(valuation_df[["Method", "Intrinsic Value per Share", "Description"]], width=1200)
            st.markdown("### Valuation Comparison Chart")
            chart_data = valuation_df[valuation_df["Intrinsic Value per Share"] != "N/A"]
            if not chart_data.empty:
                methods = chart_data["Method"].tolist()
                intrinsic_values = [float(val.replace("$", "")) for val in chart_data["Intrinsic Value per Share"]]
                chart_methods = methods + ["Current Price"]
                chart_values = intrinsic_values + [current_price]
                axis_text_color = "#000000" if st.session_state.theme == "light" else st.session_state.theme_styles['text']
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=chart_methods,
                        y=chart_values,
                        marker_color=st.session_state.theme_styles["bar_colors"][:len(chart_methods)],
                        text=[f"${v:.2f}" for v in chart_values],
                        textposition="auto"
                    )
                )
                fig.add_hline(
                    y=current_price,
                    line_dash="dash",
                    line_color=st.session_state.theme_styles["bar_colors"][-1],
                    annotation_text="Current Price",
                    annotation_position="top right"
                )
                fig.update_layout(
                    title=dict(text=f"{stock_ticker} Intrinsic Value vs. Current Price", font=dict(color=st.session_state.theme_styles["text"])),
                    xaxis_title="Valuation Method",
                    yaxis_title="Price per Share (USD)",
                    showlegend=False,
                    plot_bgcolor=st.session_state.theme_styles["plot_bg"],
                    paper_bgcolor=st.session_state.theme_styles["bg"],
                    font=dict(family="Arial", size=14, color=axis_text_color),
                    height=500,
                    xaxis=dict(
                        title=dict(font=dict(color=axis_text_color)),
                        tickfont=dict(family="Arial", size=14, color=axis_text_color),
                        tickangle=45,
                        gridcolor=st.session_state.theme_styles["grid"]
                    ),
                    yaxis=dict(
                        title=dict(font=dict(color=axis_text_color)),
                        tickfont=dict(family="Arial", size=14, color=axis_text_color),
                        gridcolor=st.session_state.theme_styles["grid"]
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Insufficient data for valuation chart.")
            st.markdown("### How Intrinsic Values Are Calculated")
            for method, calc in calculation_details:
                with st.expander(f"{method} Calculation"):
                    st.markdown(f'<span class="calc-header">{method}</span>', unsafe_allow_html=True)
                    st.markdown(calc)
        else:
            st.warning("Could not fetch data for valuations.")
# Learning Section
st.subheader("📚 Learn")
st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}">Test your investing knowledge with interactive quizzes.</p>', unsafe_allow_html=True)
quiz_level = st.selectbox("Select Quiz Level:", ["Beginner", "Intermediate", "Expert"])

quizzes = {
    "Beginner": [
        {"question": "What is a stock?", "options": ["A loan to a company", "Ownership in a company", "A type of bond"], "answer": "Ownership in a company"},
        {"question": "What does P/E ratio measure?", "options": ["Profit margin", "Price per earnings", "Portfolio value"], "answer": "Price per earnings"},
        {"question": "What is a dividend?", "options": ["A loan repayment", "A share of profits paid to shareholders", "A stock split"], "answer": "A share of profits paid to shareholders"},
        {"question": "What is a bull market?", "options": ["Falling prices", "Rising prices", "Stable prices"], "answer": "Rising prices"}
    ],
    "Intermediate": [
        {"question": "What is a golden cross?", "options": ["50-day SMA crossing above 200-day SMA", "A sharp price drop", "A dividend increase"], "answer": "50-day SMA crossing above 200-day SMA"},
        {"question": "What does RSI above 70 indicate?", "options": ["Oversold", "Overbought", "Neutral"], "answer": "Overbought"},
        {"question": "What is beta?", "options": ["A measure of debt", "A measure of stock volatility", "A type of option"], "answer": "A measure of stock volatility"},
        {"question": "What does a high P/B ratio suggest?", "options": ["Undervalued stock", "Overvalued stock", "Low debt"], "answer": "Overvalued stock"}
    ],
    "Expert": [
        {"question": "What is the DCF valuation method?", "options": ["Dividend discount model", "Discounted cash flow", "Debt-to-equity calculation"], "answer": "Discounted cash flow"},
        {"question": "What does a PEG ratio below 1 suggest?", "options": ["Overvalued stock", "Undervalued stock", "High debt"], "answer": "Undervalued stock"},
        {"question": "What is the purpose of Bollinger Bands?", "options": ["Measure earnings", "Identify overbought/oversold conditions", "Calculate dividends"], "answer": "Identify overbought/oversold conditions"},
        {"question": "What does a high debt/equity ratio indicate?", "options": ["Low risk", "High financial leverage", "High dividends"], "answer": "High financial leverage"}
    ]
}

if quiz_level:
    st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}"><b>{quiz_level} Quiz</b></p>', unsafe_allow_html=True)
    score = 0
    total_questions = len(quizzes[quiz_level])
    for i, q in enumerate(quizzes[quiz_level], 1):
        st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}"><b>Question {i}: {q["question"]}</b></p>', unsafe_allow_html=True)
        answer = st.radio(f"Select an answer for question {i}:", q["options"], key=f"quiz_{quiz_level}_{i}")
        if st.button(f"Check Answer {i}", key=f"check_{quiz_level}_{i}"):
            if answer == q["answer"]:
                st.success("Correct!")
                score += 1
            else:
                st.error(f"Incorrect. The answer is: {q['answer']}.")
    if st.button("Submit Quiz", key=f"submit_{quiz_level}"):
        st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}">Your Score: {score}/{total_questions} ({score/total_questions*100:.1f}%)</p>', unsafe_allow_html=True)
# Portfolio
st.subheader("💼 Portfolio")
st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}">Track your investments and analyze sentiment.</p>', unsafe_allow_html=True)

@retry_with_backoff
def get_sentiment(ticker):
    cache_key = f"sentiment_{ticker}"
    if cache_key not in st.session_state.cache:
        logger.info(f"Fetching sentiment for {ticker}")
        try:
            subreddits = ["wallstreetbets", "stocks"]
            posts = []
            for subreddit in subreddits:
                for submission in reddit.subreddit(subreddit).search(ticker, limit=5):
                    text = submission.title + " " + (submission.selftext[:200] if submission.selftext else "")
                    score = analyzer.polarity_scores(text)["compound"]
                    posts.append(score)
            if posts:
                avg_score = np.mean(posts)
                sentiment = "Positive" if avg_score > 0.05 else "Negative" if avg_score < -0.05 else "Neutral"
                st.session_state.cache[cache_key] = f"{sentiment} ({avg_score:.2f})"
            else:
                st.session_state.cache[cache_key] = "N/A"
        except Exception as e:
            logger.error(f"Failed to fetch sentiment for {ticker}: {str(e)}")
            st.session_state.cache[cache_key] = "N/A"
    return st.session_state.cache[cache_key]

portfolio_input = st.text_input("Enter tickers (comma-separated, e.g., AAPL,MSFT):", value="AAPL,MSFT").strip().upper()
shares_input = st.text_input("Enter shares for each ticker (comma-separated, e.g., 100,50):", value="100,50").strip()
purchase_price_input = st.text_input("Enter purchase price per share for each ticker (comma-separated, e.g., 150.00,300.00):", value="150.00,300.00").strip()

if portfolio_input and shares_input and purchase_price_input:
    tickers = [t.strip() for t in portfolio_input.split(",")]
    try:
        shares = [int(s.strip()) for s in shares_input.split(",")]
        purchase_prices = [float(p.strip()) for p in purchase_price_input.split(",")]
        if len(tickers) != len(shares) or len(tickers) != len(purchase_prices):
            st.error("Number of tickers, shares, and purchase prices must match.")
        else:
            portfolio_data = []
            with st.spinner("Fetching portfolio data..."):
                for ticker, share, purchase_price in zip(tickers, shares, purchase_prices):
                    stock_info = get_stock_info(ticker)
                    if stock_info:
                        current_price = float(stock_info.get("PreviousClose", stock_info.get("regularMarketPrice", stock_info.get("regularMarketPreviousClose", 0))))
                        pe_ratio = float(stock_info.get("PERatio", stock_info.get("trailingPE", "N/A")))
                        pb_ratio = float(stock_info.get("PriceToBookRatio", stock_info.get("priceToBook", "N/A")))
                        sentiment = get_sentiment(ticker)
                        gain_loss = (current_price - purchase_price) * share
                        portfolio_data.append({
                            "Ticker": ticker,
                            "Shares": share,
                            "Purchase Price": f"${purchase_price:.2f}",
                            "Current Price": f"${current_price:.2f}",
                            "Value": f"${current_price * share:.2f}",
                            "Gain/Loss": f"${gain_loss:.2f}",
                            "P/E Ratio": f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else pe_ratio,
                            "P/B Ratio": f"{pb_ratio:.2f}" if isinstance(pb_ratio, (int, float)) else pb_ratio,
                            "Sentiment": sentiment
                        })
                    else:
                        st.warning(f"Could not fetch data for {ticker}.")
            if portfolio_data:
                st.markdown("### Portfolio Summary")
                portfolio_df = pd.DataFrame(portfolio_data)
                st.dataframe(portfolio_df, use_container_width=True)
                csv = portfolio_df.to_csv(index=False)
                st.download_button(
                    label="Download Portfolio Summary as CSV",
                    data=csv,
                    file_name=f"portfolio_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                # Portfolio Allocation Chart
                st.markdown("### Portfolio Allocation")
                values = [float(item["Value"].replace("$", "")) for item in portfolio_data]
                labels = [item["Ticker"] for item in portfolio_data]
                fig = go.Figure(data=[
                    go.Pie(labels=labels, values=values, textinfo='label+percent', marker=dict(colors=st.session_state.theme_styles["bar_colors"][:len(labels)]))
                ])
                fig.update_layout(
                    title=dict(text="Portfolio Allocation by Value", font=dict(color=st.session_state.theme_styles["text"])),
                    plot_bgcolor=st.session_state.theme_styles["plot_bg"],
                    paper_bgcolor=st.session_state.theme_styles["bg"],
                    font=dict(family="Arial", size=12, color=st.session_state.theme_styles["text"])
                )
                st.plotly_chart(fig, use_container_width=True)
                # Portfolio Simulator
                st.markdown("### Portfolio Simulator")
                st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}">Estimate your portfolio’s growth over time.</p>', unsafe_allow_html=True)
                current_age = st.number_input("Enter your current age:", min_value=18, max_value=100, value=30, step=1)
                retirement_age = 65
                years_to_retirement = max(retirement_age - current_age, 1)
                growth_rate = st.slider("Expected Annual Growth Rate (%):", 0.0, 20.0, 10.3, 1.0)
                years = st.slider("Investment Horizon (Years):", 1, 50, years_to_retirement, 1)
                total_value = sum(float(item["Value"].replace("$", "")) for item in portfolio_data)
                future_value = total_value * (1 + growth_rate / 100) ** years
                retirement_value = total_value * (1 + growth_rate / 100) ** years_to_retirement
                st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}">Current Portfolio Value: ${total_value:,.2f}</p>', unsafe_allow_html=True)
                st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}">Future Value ({years} years, {growth_rate}% growth): ${future_value:,.2f}</p>', unsafe_allow_html=True)
                st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}">Value at Retirement (Age {retirement_age}, {years_to_retirement} years, {growth_rate}% growth): ${retirement_value:,.2f}</p>', unsafe_allow_html=True)
                # Growth Over Time Chart
                years_range = range(years + 1)
                growth_values = [total_value * (1 + growth_rate / 100) ** t for t in years_range]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(years_range), y=growth_values, mode="lines", name="Portfolio Value", line=dict(color=st.session_state.theme_styles["line"])))
                fig.update_layout(
                    title=dict(text="Portfolio Growth Over Time", font=dict(color=st.session_state.theme_styles["text"])),
                    xaxis_title="Years",
                    yaxis_title="Portfolio Value (USD)",
                    plot_bgcolor=st.session_state.theme_styles["plot_bg"],
                    paper_bgcolor=st.session_state.theme_styles["bg"],
                    font=dict(family="Arial", size=12, color=st.session_state.theme_styles["text"]),
                    xaxis=dict(gridcolor=st.session_state.theme_styles["grid"]),
                    yaxis=dict(gridcolor=st.session_state.theme_styles["grid"])
                )
                st.plotly_chart(fig, use_container_width=True)
    except ValueError:
        st.error("Invalid input for shares or purchase prices. Please enter numbers (e.g., 100,50 for shares; 150.00,300.00 for prices).")
# Reddit Sentiment
st.subheader("🗣️ Reddit Sentiment")
st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}">Analyze sentiment from Reddit posts.</p>', unsafe_allow_html=True)
sentiment_ticker = st.text_input("Enter a ticker for sentiment analysis:", value="AAPL").strip().upper()

if sentiment_ticker:
    with st.spinner(f"Fetching Reddit posts for {sentiment_ticker}..."):
        try:
            subreddits = ["wallstreetbets", "stocks"]
            posts = []
            sentiment_scores = []
            for subreddit in subreddits:
                for submission in reddit.subreddit(subreddit).search(sentiment_ticker, limit=5):
                    text = submission.title + " " + (submission.selftext[:200] if submission.selftext else "")
                    score = analyzer.polarity_scores(text)["compound"]
                    sentiment = "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"
                    sentiment_scores.append(score)
                    posts.append({
                        "Title": submission.title,
                        "URL": submission.url,
                        "Sentiment": f"{sentiment} ({score:.2f})",
                        "Date": datetime.utcfromtimestamp(submission.created_utc).strftime("%Y-%m-%d"),
                        "Subreddit": subreddit
                    })
            if posts:
                avg_score = np.mean(sentiment_scores)
                overall_sentiment = "Positive" if avg_score > 0.05 else "Negative" if avg_score < -0.05 else "Neutral"
                st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}">Overall Sentiment: {overall_sentiment} ({avg_score:.2f})</p>', unsafe_allow_html=True)
                st.markdown("### Recent Reddit Posts")
                reddit_df = pd.DataFrame(posts)
                st.dataframe(reddit_df, use_container_width=True)
                csv = reddit_df.to_csv(index=False)
                st.download_button(
                    label="Download Reddit Sentiment as CSV",
                    data=csv,
                    file_name=f"reddit_sentiment_{sentiment_ticker}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                # Sentiment Distribution
                st.markdown("### Sentiment Distribution")
                sentiment_counts = reddit_df["Sentiment"].apply(lambda x: x.split(" (")[0]).value_counts()
                fig = go.Figure(data=[
                    go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values, textinfo='label+percent', marker=dict(colors=st.session_state.theme_styles["bar_colors"][:len(sentiment_counts)]))
                ])
                fig.update_layout(
                    title=dict(text="Sentiment Distribution", font=dict(color=st.session_state.theme_styles["text"])),
                    plot_bgcolor=st.session_state.theme_styles["plot_bg"],
                    paper_bgcolor=st.session_state.theme_styles["bg"],
                    font=dict(family="Arial", size=12, color=st.session_state.theme_styles["text"])
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No Reddit posts found for {sentiment_ticker}.")
        except Exception as e:
            logger.error(f"Error fetching Reddit posts for {sentiment_ticker}: {str(e)}")
            st.error(f"Error fetching Reddit posts: {str(e)}.")
# Stock News
st.subheader("📰 Stock News")
st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}">Latest news articles related to the stock.</p>', unsafe_allow_html=True)

@retry_with_backoff
def fetch_alpha_vantage_news(ticker):
    logger.info(f"Fetching news for {ticker} from Alpha Vantage")
    try:
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url).json()
        if "feed" in response and response["feed"]:
            news_data = []
            for item in response["feed"][:5]:
                title = item.get("title", "N/A")
                link = item.get("url", "#")
                publisher = item.get("source", "Unknown")
                date = item.get("time_published", "N/A")
                if date != "N/A":
                    date = datetime.strptime(date, "%Y%m%dT%H%M%S").strftime("%Y-%m-%d")
                summary = item.get("summary", "No summary available.")
                if len(summary) > 100:
                    summary = summary[:100] + "..."
                news_data.append({
                    "Title": title,
                    "URL": link,
                    "Publisher": publisher,
                    "Date": date,
                    "Summary": summary
                })
            return news_data
        return None
    except Exception as e:
        logger.error(f"Failed to fetch news from Alpha Vantage for {ticker}: {str(e)}")
        return None

@retry_with_backoff
def fetch_yfinance_news(ticker):
    logger.info(f"Fetching news for {ticker} from yfinance")
    try:
        stock = yf.Ticker(ticker)
        news_items = stock.news
        if news_items:
            news_data = []
            for item in news_items[:5]:
                title = item.get("title", "N/A")
                link = item.get("link", "#")
                publisher = item.get("publisher", "Unknown")
                timestamp = item.get("providerPublishTime", 0)
                date = datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d") if timestamp else "N/A"
                summary = item.get("summary", "No summary available.")
                if len(summary) > 100:
                    summary = summary[:100] + "..."
                news_data.append({
                    "Title": title,
                    "URL": link,
                    "Publisher": publisher,
                    "Date": date,
                    "Summary": summary
                })
            return news_data
        return None
    except Exception as e:
        logger.error(f"Failed to fetch news from yfinance for {ticker}: {str(e)}")
        return None

if stock_ticker:
    with st.spinner(f"Fetching news for {stock_ticker}..."):
        news_data = fetch_alpha_vantage_news(stock_ticker)
        if not news_data:
            news_data = fetch_yfinance_news(stock_ticker)
        if news_data:
            st.markdown("### Recent News Articles")
            news_df = pd.DataFrame(news_data)
            st.dataframe(news_df, use_container_width=True)
            csv = news_df.to_csv(index=False)
            st.download_button(
                label="Download Stock News as CSV",
                data=csv,
                file_name=f"stock_news_{stock_ticker}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.warning(f"No news found for {stock_ticker}. Check logs for details.")
# Stock Screener
st.subheader("🔎 Stock Screener")
st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}">Find stocks based on multiple financial metrics.</p>', unsafe_allow_html=True)

stock_list = [
    {"Ticker": "AAPL", "Company Name": "Apple Inc.", "Sector": "Technology", "Market Cap": 2.5e12, "P/E Ratio": 28.5, "P/B Ratio": 36.0, "Dividend Yield": 0.5, "Beta": 1.2, "Revenue Growth": 5.0, "Debt/Equity": 1.8, "ROE": 150.0, "P/S Ratio": 7.0, "PEG Ratio": 2.1},
    {"Ticker": "MSFT", "Company Name": "Microsoft Corporation", "Sector": "Technology", "Market Cap": 2.1e12, "P/E Ratio": 35.0, "P/B Ratio": 12.0, "Dividend Yield": 0.8, "Beta": 0.9, "Revenue Growth": 15.0, "Debt/Equity": 0.5, "ROE": 40.0, "P/S Ratio": 10.0, "PEG Ratio": 1.8},
    {"Ticker": "GOOGL", "Company Name": "Alphabet Inc.", "Sector": "Technology", "Market Cap": 1.8e12, "P/E Ratio": 25.0, "P/B Ratio": 6.5, "Dividend Yield": 0.0, "Beta": 1.0, "Revenue Growth": 10.0, "Debt/Equity": 0.1, "ROE": 25.0, "P/S Ratio": 6.0, "PEG Ratio": 1.5},
    {"Ticker": "JPM", "Company Name": "JPMorgan Chase & Co.", "Sector": "Financials", "Market Cap": 5.0e11, "P/E Ratio": 12.0, "P/B Ratio": 1.5, "Dividend Yield": 2.5, "Beta": 1.1, "Revenue Growth": 3.0, "Debt/Equity": 1.2, "ROE": 15.0, "P/S Ratio": 3.0, "PEG Ratio": 1.0},
    {"Ticker": "XOM", "Company Name": "Exxon Mobil Corporation", "Sector": "Energy", "Market Cap": 4.0e11, "P/E Ratio": 15.0, "P/B Ratio": 2.0, "Dividend Yield": 5.0, "Beta": 0.9, "Revenue Growth": -5.0, "Debt/Equity": 0.8, "ROE": 12.0, "P/S Ratio": 1.5, "PEG Ratio": 1.2}
]

stocks_df = pd.DataFrame(stock_list)
stocks_df["Market Cap"] = stocks_df["Market Cap"].apply(market_cap_display)
stocks_df["Dividend Yield"] = stocks_df["Dividend Yield"].apply(lambda x: f"{x:.1f}%")
stocks_df["Revenue Growth"] = stocks_df["Revenue Growth"].apply(lambda x: f"{x:.1f}%")
stocks_df["Debt/Equity"] = stocks_df["Debt/Equity"].apply(lambda x: f"{x:.2f}")
stocks_df["ROE"] = stocks_df["ROE"].apply(lambda x: f"{x:.1f}%")
stocks_df["P/S Ratio"] = stocks_df["P/S Ratio"].apply(lambda x: f"{x:.2f}")
stocks_df["PEG Ratio"] = stocks_df["PEG Ratio"].apply(lambda x: f"{x:.2f}")

# Filter options
st.markdown("### Filter Stocks")
sector_filter = st.multiselect("Sector:", stocks_df["Sector"].unique(), default=stocks_df["Sector"].unique())
market_cap_filter = st.slider("Market Cap (Billions):", min_value=0.0, max_value=3.0, value=(0.0, 3.0), step=0.1)
pe_ratio_filter = st.slider("P/E Ratio:", min_value=0.0, max_value=50.0, value=(0.0, 50.0), step=1.0)
dividend_yield_filter = st.slider("Dividend Yield (%):", min_value=0.0, max_value=10.0, value=(0.0, 10.0), step=0.1)
beta_filter = st.slider("Beta:", min_value=0.0, max_value=2.0, value=(0.0, 2.0), step=0.1)
roe_filter = st.slider("ROE (%):", min_value=0.0, max_value=200.0, value=(0.0, 200.0), step=1.0)

# Apply filters
filtered_df = stocks_df.copy()
if sector_filter:
    filtered_df = filtered_df[filtered_df["Sector"].isin(sector_filter)]
filtered_df = filtered_df[
    (filtered_df["Market Cap"].str.extract(r'(\d+\.?\d*)')[0].astype(float) >= market_cap_filter[0] * 1e9) &
    (filtered_df["Market Cap"].str.extract(r'(\d+\.?\d*)')[0].astype(float) <= market_cap_filter[1] * 1e9) &
    (filtered_df["P/E Ratio"].astype(float) >= pe_ratio_filter[0]) &
    (filtered_df["P/E Ratio"].astype(float) <= pe_ratio_filter[1]) &
    (filtered_df["Dividend Yield"].str.rstrip('%').astype(float) >= dividend_yield_filter[0]) &
    (filtered_df["Dividend Yield"].str.rstrip('%').astype(float) <= dividend_yield_filter[1]) &
    (filtered_df["Beta"].astype(float) >= beta_filter[0]) &
    (filtered_df["Beta"].astype(float) <= beta_filter[1]) &
    (filtered_df["ROE"].str.rstrip('%').astype(float) >= roe_filter[0]) &
    (filtered_df["ROE"].str.rstrip('%').astype(float) <= roe_filter[1])
]

# Display results
if not filtered_df.empty:
    st.markdown("### Matching Stocks")
    st.dataframe(filtered_df, use_container_width=True)
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download Stock Screener Results as CSV",
        data=csv,
        file_name=f"stock_screener_results_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
else:
    st.warning("No stocks match the selected filters.")
# Footer
st.markdown("---")
st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}">Developed by EquityScope Team | Version: 1.0.0 | Last Updated: {datetime.now().strftime("%Y-%m-%d")}</p>', unsafe_allow_html=True)
st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}">Data sourced from Alpha Vantage, Yahoo Finance, and Reddit API. For support, contact us at support@equityscope.com.</p>', unsafe_allow_html=True)
st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}">Disclaimer: This app is for educational purposes only. Investing involves risks. Consult a financial advisor before making decisions.</p>', unsafe_allow_html=True)