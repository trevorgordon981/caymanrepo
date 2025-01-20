from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Function to get stock data
def get_stock_data(ticker, period="1y"):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)

        if data.empty:
            return None

        # Calculate moving averages
        data['50_day_ma'] = data['Close'].rolling(window=50).mean()
        data['200_day_ma'] = data['Close'].rolling(window=200).mean()

        stock_info = {
            'symbol': ticker.upper(),
            'price': f"{data['Close'].iloc[-1]:.2f}",
            '50_day_ma': f"{data['50_day_ma'].iloc[-1]:.2f}" if not pd.isna(data['50_day_ma'].iloc[-1]) else "N/A",
            '200_day_ma': f"{data['200_day_ma'].iloc[-1]:.2f}" if not pd.isna(data['200_day_ma'].iloc[-1]) else "N/A"
        }

        return stock_info

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None


# Home route
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        ticker = request.form["ticker"].strip()
        stock_data = get_stock_data(ticker)
        if stock_data:
            return render_template("stock.html", stock_data=stock_data)
        else:
            error_message = f"Info is not attainable for ticker: {ticker.upper()}"
            return render_template("home.html", error_message=error_message)
    return render_template("home.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
