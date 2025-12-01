import numpy as np

def calculate_sharpe_ratio(returns, risk_free_rate, std_dev):
    """
    Calculates the Sharpe Ratio.

    Args:
        returns (float): Portfolio's average annual return.
        risk_free_rate (float): Annual risk-free rate.
        std_dev (float): Portfolio's annual standard deviation of returns.

    Returns:
        float: The Sharpe Ratio.
    """
    if std_dev == 0:
        return 0.0 # Or handle as an error
    return (returns - risk_free_rate) / std_dev

def monte_carlo_simulation(initial_price, drift, volatility, time_steps, num_simulations):
    """
    Performs a simple Monte Carlo simulation for stock price paths.

    Args:
        initial_price (float): Starting price of the asset.
        drift (float): Expected return (mean of daily returns).
        volatility (float): Standard deviation of daily returns.
        time_steps (int): Number of days to simulate.
        num_simulations (int): Number of simulation paths.

    Returns:
        numpy.ndarray: An array of simulated final prices for each path.
    """
    dt = 1 # daily steps
    simulations = np.zeros((time_steps + 1, num_simulations))
    simulations[0] = initial_price

    for t in range(1, time_steps + 1):
        # Brownian motion component
        rand = np.random.standard_normal(num_simulations)
        # Geometric Brownian Motion formula
        simulations[t] = simulations[t-1] * np.exp((drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * rand)

    return simulations[-1]

def calculate_portfolio_value(prices, quantities):
    """
    Calculates the total value of a portfolio given asset prices and quantities.

    Args:
        prices (list): List of asset prices.
        quantities (list): List of quantities for each asset.

    Returns:
        float: Total portfolio value.
    """
    if len(prices) != len(quantities):
        raise ValueError("Prices and quantities lists must have the same length.")
    return sum(p * q for p, q in zip(prices, quantities))
