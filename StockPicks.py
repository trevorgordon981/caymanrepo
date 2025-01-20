from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Function to get stock data and optionally generate a graph
def get_stock_data(ticker, generate_graph=False, period="1y"):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)

        if data.empty:
            return None, None if generate_graph else None

        # Calculate moving averages
        data['50_day_ma'] = data['Close'].rolling(window=50).mean()
        data['200_day_ma'] = data['Close'].rolling(window=200).mean()

        stock_info = {
            'symbol': ticker.upper(),
            'price': f"{data['Close'].iloc[-1]:.2f}",
            '50_day_ma': f"{data['50_day_ma'].iloc[-1]:.2f}" if not pd.isna(data['50_day_ma'].iloc[-1]) else "N/A",
            '200_day_ma': f"{data['200_day_ma'].iloc[-1]:.2f}" if not pd.isna(data['200_day_ma'].iloc[-1]) else "N/A"
        }

        if generate_graph:
            # Plot the stock data
            plt.figure(figsize=(10, 5))
            plt.plot(data.index, data['Close'], label="Closing Price", color="blue")
            plt.plot(data.index, data['50_day_ma'], label="50-Day MA", color="green")
            plt.plot(data.index, data['200_day_ma'], label="200-Day MA", color="red")
            plt.title(f"{ticker.upper()} Stock Price ({period})")
            plt.xlabel("Date")
            plt.ylabel("Price (USD)")
            plt.legend()
            plt.grid()

            # Convert the plot to a PNG image and encode it
            img = BytesIO()
            plt.savefig(img, format="png")
            img.seek(0)
            graph_url = base64.b64encode(img.getvalue()).decode()
            plt.close()

            return stock_info, graph_url

        return stock_info

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None, None if generate_graph else None


# Home route
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        ticker = request.form["ticker"].strip()
        stock_data, graph_url = get_stock_data(ticker, generate_graph=True)
        if stock_data:
            return render_template("graph.html", ticker=ticker, graph_url=graph_url, timeframe="1 Year")
        else:
            error_message = f"Info is not attainable for ticker: {ticker.upper()}"
            return render_template("home.html", error_message=error_message)
    return render_template("home.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
