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
    stock = yf.Ticker(ticker)
    try:
        price = stock.history(period="1d")['Close'].iloc[-1]
        return {
            'symbol': ticker.upper(),
            'price': f"{price:.2f}",  # Format to 2 decimal places
            'open': "N/A",  # Replace with actual data if needed
            'high': "N/A",
            'low': "N/A",
            'volume': "N/A"
        }
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None


    
# Function to get dividend data
def get_dividend_data(ticker):
    stock = yf.Ticker(ticker)
    dividends = stock.dividends
    
    if dividends.empty:
        print(f"No dividend data available for {ticker}.")
        return None
    
    dividend_df = dividends.reset_index()
    dividend_df.columns = ['Date', 'Dividend']
    return dividend_df.tail(5)  # Return the last 5 diviends


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

# Function to get historical data using Yahoo Finance
def get_historical_data_yf(ticker):
    stock = yf.Ticker(ticker)
    hist_data = stock.history(period="max")  # Fetch maximum available historical data
    
    if hist_data.empty:
        print(f"No historical data available for {ticker.upper()}.")
        return None
    
    return hist_data




# Function to generate the graph
# Function to generate the graph using Yahoo Finance data
def generate_graph_yf(ticker, timeframe):
    data = get_historical_data_yf(ticker)
    
    if data is None:
        return None

    # Filter data based on the selected timeframe
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

    # Plot the graph
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_data.index, filtered_data['Close'], label="Close Price")  # Use 'Close' column
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
        ticker = request.form["ticker"].strip()
        
        if action == "price":
            stock_data = get_stock_data(ticker)
            dividends = get_dividend_data(ticker)
            if stock_data and dividends is not None:
                return render_template("stock.html", stock_data=stock_data, dividends=dividends)
            else:
                error_message = f"Info is not attainable for ticker: {ticker.upper()}"
                return render_template("home.html", error_message=error_message)
        
        elif action == "graph":
            timeframe = request.form["timeframe"]
            return graph(timeframe)
    
    return render_template("home.html")


# Route for graph
@app.route("/graph/<timeframe>", methods=["GET", "POST"])
def graph(timeframe):
    ticker = request.args.get("ticker")
    
    if not ticker:
        error_message = "Please provide a stock ticker to generate the graph."
        return render_template("home.html", error_message=error_message)
    
    ticker = ticker.upper()
    graph_url = generate_graph_yf(ticker, timeframe)
    if graph_url:
        return render_template("graph.html", graph_url=graph_url, ticker=ticker, timeframe=timeframe)
    else:
        error_message = f"Failed to generate graph for {ticker}."
        return render_template("home.html", error_message=error_message)





if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
