
import pandas as pd
import numpy as np

def load_historical_data(symbol='AAPL', start_date='2020-01-01', end_date='2023-01-01'):
    """
    Loads historical stock data. For demonstration, generates dummy data.
    In a real application, this would fetch data from a financial API.
    """
    print(f"Loading historical data for {symbol} from {start_date} to {end_date}")

    date_range = pd.date_range(start=start_date, end=end_date, freq='B') # Business days
    num_days = len(date_range)

    # Generate dummy prices
    np.random.seed(42)
    open_prices = np.random.uniform(100, 200, num_days).cumsum() + 100
    close_prices = open_prices + np.random.uniform(-5, 5, num_days)
    high_prices = np.maximum(open_prices, close_prices) + np.random.uniform(0, 5, num_days)
    low_prices = np.minimum(open_prices, close_prices) - np.random.uniform(0, 5, num_days)
    volume = np.random.randint(1_000_000, 10_000_000, num_days)

    df = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volume
    }, index=date_range)

    return df

if __name__ == "__main__":
    data = load_historical_data()
    print("Sample Historical Data:")
    print(data.head())
    print(data.tail())
