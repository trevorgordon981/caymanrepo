from flask import Flask, render_template, request
import requests
import os
from jinja2 import Environment, FileSystemLoader
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import base64
import yfinance as yf

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Alpha Vantage API key (replace with your own)
API_KEY = "GTU4V5Y3SCLWFC2C"

# Function to get stock data
def get_stock_data(ticker):
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker.upper()}&apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    if 'Global Quote' in data:
        stock_data = {
            'symbol': ticker.upper(),
            'price': data['Global Quote']['05. price'],
            'open': data['Global Quote']['02. open'],
            'high': data['Global Quote']['03. high'],
            'low': data['Global Quote']['04. low'],
            'volume': data['Global Quote']['06. volume']
        }
        stock_data.update(get_moving_averages(ticker))
        return stock_data
    else:
        return None
    
# Function to get dividend data
def get_dividend_data(ticker):
    stock = yf.Ticker(ticker)
    dividends = stock.dividends  # Retrieve dividend data
    
    # Convert to DataFrame for easier handling
    dividend_df = dividends.reset_index()
    dividend_df.columns = ['Date', 'Dividend']
    
    return dividend_df.tail(5)  # Return the most recent 5 dividends

# Function to get moving averages
def get_moving_averages(ticker):
    moving_averages = {}
    for time_period in [50, 200]:
        url = f"https://www.alphavantage.co/query?function=SMA&symbol={ticker.upper()}&interval=daily&time_period={time_period}&series_type=close&apikey={API_KEY}"
        response = requests.get(url)
        data = response.json()
        if 'Technical Analysis: SMA' in data:
            sma_data = data['Technical Analysis: SMA']
            most_recent_date = list(sma_data.keys())[0]
            moving_averages[f'{time_period}_day'] = sma_data[most_recent_date]['SMA']
        else:
            moving_averages[f'{time_period}_day'] = "N/A"
    return moving_averages

# Function to get historical data for the graph
def get_historical_data(ticker, outputsize="compact"):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker.upper()}&outputsize={outputsize}&apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    
    # Debugging: Print the raw response
    print("Alpha Vantage API Response:", data)
    
    if 'Time Series (Daily)' in data:
        return pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index', dtype=float)
    else:
        return None



# Function to generate the graph
def generate_graph(ticker, timeframe):
    outputsize = "full" if timeframe in ["2 years", "5 years", "10 years"] else "compact"
    data = get_historical_data(ticker, outputsize)
    
    if data is None or data.empty:
        print(f"No data available for {ticker.upper()} with timeframe {timeframe}.")
        return None

    data.index = pd.to_datetime(data.index)
    data = data.sort_index()

    end_date = data.index.max()
    if timeframe == "5 days":
        start_date = end_date - pd.Timedelta(days=5)
    elif timeframe == "1 month":
        start_date = end_date - pd.DateOffset(months=1)
    elif timeframe == "6 months":
        start_date = end_date - pd.DateOffset(months=6)
    elif timeframe == "1 year":
        start_date = end_date - pd.DateOffset(years=1)
    elif timeframe == "2 years":
        start_date = end_date - pd.DateOffset(years=2)
    elif timeframe == "5 years":
        start_date = end_date - pd.DateOffset(years=5)
    elif timeframe == "10 years":
        start_date = end_date - pd.DateOffset(years=10)
    else:
        start_date = end_date

    filtered_data = data.loc[start_date:end_date]

    # Ensure filtered data is not empty
    if filtered_data.empty:
        print(f"No filtered data available for {ticker.upper()} in the selected timeframe.")
        return None

    plt.figure(figsize=(10, 6))
    plt.plot(filtered_data.index, filtered_data['5. adjusted close'], label="Adjusted Close Price")
    plt.title(f"{ticker.upper()} Stock Price ({timeframe})")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    graph_url = base64.b64encode(buf.getvalue()).decode('utf8')
    buf.close()
    plt.close()
    return graph_url


# Home route
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        action = request.form.get("action")
        ticker = request.form["ticker"]
        
        if action == "price":
            stock_data = get_stock_data(ticker)
            dividends = get_dividend_data(ticker)  # Fetch dividend data
            if stock_data:
                return render_template("stock.html", stock_data=stock_data, dividends=dividends)
            else:
                error_message = f"No data found for ticker: {ticker.upper()}"
                return render_template("home.html", error_message=error_message)
        
        elif action == "graph":
            timeframe = request.form["timeframe"]
            return graph(timeframe)
    
    return render_template("home.html")


# Route for graph
@app.route("/graph/<timeframe>", methods=["GET", "POST"])
def graph(timeframe):
    ticker = request.args.get("ticker", "AAPL").upper()  # Ensure ticker is uppercase
    graph_url = generate_graph(ticker, timeframe)
    if graph_url:
        return render_template("graph.html", graph_url=graph_url, ticker=ticker, timeframe=timeframe)
    else:
        error_message = f"Failed to generate graph for {ticker.upper()}."
        return render_template("home.html", error_message=error_message)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
