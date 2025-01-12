import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import base64

# Function to get historical data for the graph
def get_historical_data(ticker, outputsize="compact"):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker.upper()}&outputsize={outputsize}&apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    if 'Time Series (Daily)' in data:
        return pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index', dtype=float)
    else:
        return None

# Function to generate the graph
def generate_graph(ticker, timeframe):
    # Fetch historical data
    outputsize = "full" if timeframe in ["2 years", "5 years", "10 years"] else "compact"
    data = get_historical_data(ticker, outputsize)
    if data is None:
        return None

    # Convert index to datetime and sort
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()

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

    # Plot the graph
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_data.index, filtered_data['5. adjusted close'], label="Adjusted Close Price")
    plt.title(f"{ticker.upper()} Stock Price ({timeframe})")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    # Save the graph to a buffer
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    graph_url = base64.b64encode(buf.getvalue()).decode('utf8')
    buf.close()
    plt.close()
    return graph_url

# Route for graph
@app.route("/graph/<timeframe>", methods=["GET"])
def graph(timeframe):
    ticker = request.args.get("ticker", "AAPL")
    graph_url = generate_graph(ticker, timeframe)
    if graph_url:
        return render_template("graph.html", graph_url=graph_url, ticker=ticker, timeframe=timeframe)
    else:
        error_message = f"Failed to generate graph for {ticker.upper()}."
        return render_template("home.html", error_message=error_message)
