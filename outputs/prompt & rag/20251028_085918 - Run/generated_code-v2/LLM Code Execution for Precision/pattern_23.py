import numpy as np

def calculate_sharpe_ratio(returns, risk_free_rate):
    """
    Calculates the Sharpe Ratio for a given set of returns.
    Assumes returns are daily, and annualizes the standard deviation for annual Sharpe Ratio.
    """
    if not isinstance(returns, np.ndarray):
        returns = np.array(returns)

    if len(returns) < 2:
        return 0.0 # Cannot calculate standard deviation with less than 2 data points

    excess_returns = returns - risk_free_rate
    mean_excess_return = np.mean(excess_returns)
    std_dev_excess_return = np.std(excess_returns, ddof=1) # Sample standard deviation

    if std_dev_excess_return == 0:
        return 0.0 # Avoid division by zero

    # Assuming daily returns, annualize Sharpe Ratio (sqrt(252) for daily to annual)
    annualized_sharpe_ratio = (mean_excess_return / std_dev_excess_return) * np.sqrt(252)
    return annualized_sharpe_ratio

def calculate_portfolio_return(weights, asset_returns):
    """
    Calculates the weighted return of a portfolio.
    weights: numpy array or list of weights for each asset.
    asset_returns: numpy array or list of returns for each asset.
    """
    if not isinstance(weights, np.ndarray):
        weights = np.array(weights)
    if not isinstance(asset_returns, np.ndarray):
        asset_returns = np.array(asset_returns)

    if len(weights) != len(asset_returns):
        raise ValueError("Weights and asset returns must have the same length.")

    if not np.isclose(np.sum(weights), 1.0):
        # Normalize weights if they don't sum to 1, or raise a warning/error
        # For this example, we'll normalize
        normalized_weights = weights / np.sum(weights)
    else:
        normalized_weights = weights

    portfolio_return = np.sum(normalized_weights * asset_returns)
    return portfolio_return

def calculate_portfolio_volatility(weights, cov_matrix):
    """
    Calculates the portfolio volatility (standard deviation).
    weights: numpy array or list of weights for each asset.
    cov_matrix: numpy array, covariance matrix of asset returns.
    """
    if not isinstance(weights, np.ndarray):
        weights = np.array(weights)
    if not isinstance(cov_matrix, np.ndarray):
        cov_matrix = np.array(cov_matrix)

    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)
    return portfolio_volatility

# Example usage (for testing purposes, not part of the main application flow)
if __name__ == "__main__":
    # Example Sharpe Ratio calculation
    daily_returns = [0.01, 0.005, -0.002, 0.015, 0.008]
    risk_free = 0.0001
    sharpe = calculate_sharpe_ratio(daily_returns, risk_free)
    print(f"Sharpe Ratio: {sharpe:.4f}")

    # Example Portfolio Return calculation
    asset_weights = [0.6, 0.4]
    asset_returns = [0.05, 0.08]
    port_return = calculate_portfolio_return(asset_weights, asset_returns)
    print(f"Portfolio Return: {port_return:.4%}")

    # Example Portfolio Volatility calculation
    weights_vol = np.array([0.5, 0.5])
    cov_matrix_vol = np.array([[0.0001, 0.00005], [0.00005, 0.0002]])
    port_vol = calculate_portfolio_volatility(weights_vol, cov_matrix_vol)
    print(f"Portfolio Volatility: {port_vol:.4f}")
