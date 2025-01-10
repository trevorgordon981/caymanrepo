from flask import Flask, render_template, request, redirect, url_for
import requests

app = Flask(__name__)

# Alpha Vantage API key (replace with your own)
API_KEY = "GTU4V5Y3SCLWFC2C"

# Function to get stock data
def get_stock_data(ticker):
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    if 'Global Quote' in data:
        stock_data = {
            'symbol': ticker,
            'price': data['Global Quote']['05. price'],
            'open': data['Global Quote']['02. open'],
            'high': data['Global Quote']['03. high'],
            'low': data['Global Quote']['04. low'],
            'volume': data['Global Quote']['06. volume']
        }
        return stock_data
    else:
        return None

# Define routes
@app.route("/", endpoint="home")
def home():
    return "Hello, Flask is running your StockTicker app!"

@app.route("/stock", methods=['GET', 'POST'])
def stock_form():
    if request.method == 'POST':
        ticker = request.form['ticker']
        stock_data = get_stock_data(ticker)
        if stock_data:
            return render_template('stock.html', stock_data=stock_data)
        else:
            return render_template('error.html', error_message='Invalid ticker symbol')
    else:
        # Redirect to 'home' if the request is not POST
        return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

