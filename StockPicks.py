from flask import Flask, render_template, request
import requests
import os
from jinja2 import Environment, FileSystemLoader

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Set up Jinja2 environment
script_dir = os.path.dirname(__file__)  # Directory of the script
templates_path = os.path.join(script_dir, 'templates')
env = Environment(loader=FileSystemLoader(templates_path))

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
        return stock_data
    else:
        return None

# Home route
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        ticker = request.form["ticker"]
        stock_data = get_stock_data(ticker)
        if stock_data:
            return render_template("stock.html", stock_data=stock_data)
        else:
            error_message = f"No data found for ticker: {ticker.upper()}"
            return render_template("home.html", error_message=error_message)
    return render_template("home.html")

# Test route
@app.route("/test")
def test():
    try:
        return render_template("home.html")
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    print("Current Working Directory:", os.getcwd())
    app.run(host="0.0.0.0", port=8080, debug=True)
