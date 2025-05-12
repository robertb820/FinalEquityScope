import yfinance as yf

ticker = "AAPL"
stock = yf.Ticker(ticker)
try:
    info = stock.info
    print("Stock Info:", info)
    history = stock.history(period="1mo")
    print("History:", history)
except Exception as e:
    print(f"Error: {e}")