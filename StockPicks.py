from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Function to get stock data with moving averages
def get_stock_data_with_moving_averages(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1y")  # Fetch past 1 year's data

        if data.empty:
            return None

        # Calculate 50-day and 200-day moving averages
        data['50_day_ma'] = data['Close'].rolling(window=50).mean()
        data['200_day_ma'] = data['Close'].rolling(window=200).mean()

        # Get the latest price and moving averages
        latest_data = data.iloc[-1]
        return {
            'symbol': ticker.upper(),
            'price': f"{latest_data['Close']:.2f}",
            '50_day_ma': f"{latest_data['50_day_ma']:.2f}" if not pd.isna(latest_data['50_day_ma']) else "N/A",
            '200_day_ma': f"{latest_data['200_day_ma']:.2f}" if not pd.isna(latest_data['200_day_ma']) else "N/A"
        }
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None


# Home route
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        ticker = request.form["ticker"].strip()
        stock_data = get_stock_data_with_moving_averages(ticker)
        if stock_data:
            return render_template("stock.html", stock_data=stock_data)
        else:
            error_message = f"Info is not attainable for ticker: {ticker.upper()}"
            return render_template("home.html", error_message=error_message)
    return render_template("home.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
