"""financial_tools.py: Contains functions for various financial calculations and data retrieval (mocked)."""

def calculate_roi(initial_investment: float, final_value: float) -> float:
    """Calculates the Return on Investment (ROI)."""
    if initial_investment == 0:
        return 0.0
    return ((final_value - initial_investment) / initial_investment) * 100

def calculate_compound_interest(principal: float, rate: float, time_years: float, compounds_per_period: int) -> float:
    """Calculates compound interest for a given principal, rate, time, and compounding frequency."""
    # A = P(1 + r/n)^(nt)
    amount = principal * (1 + rate / compounds_per_period)**(compounds_per_period * time_years)
    return amount - principal

def get_stock_price(ticker: str) -> float:
    """Mocks fetching a real-time stock price for a given ticker."""
    mock_prices = {
        "AAPL": 175.00,
        "GOOGL": 140.50,
        "MSFT": 380.25,
        "AMZN": 150.70,
        "TSLA": 200.00
    }
    return mock_prices.get(ticker.upper(), 0.0)

def analyze_market_trend(data_points: list[float], window_size: int = 3) -> str:
    """Analyzes a simple market trend using a moving average on historical data points."""
    if len(data_points) < window_size:
        return "Not enough data to determine a trend with the given window size."
    
    moving_averages = []
    for i in range(len(data_points) - window_size + 1):
        window = data_points[i:i + window_size]
        moving_averages.append(sum(window) / window_size)
    
    if len(moving_averages) < 2:
        return "Trend cannot be determined with current data."
    
    # Compare the last two moving averages to determine trend
    if moving_averages[-1] > moving_averages[-2]:
        return "Upward trend detected."
    elif moving_averages[-1] < moving_averages[-2]:
        return "Downward trend detected."
    else:
        return "Stable trend detected."