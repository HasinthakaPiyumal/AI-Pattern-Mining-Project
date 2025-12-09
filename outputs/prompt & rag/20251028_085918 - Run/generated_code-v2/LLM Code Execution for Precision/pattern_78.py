import numpy as np

def calculate_sharpe_ratio(returns, risk_free_rate, annualization_factor=12):
    """
    Calculates the Sharpe Ratio for a given series of returns.
    Assumes returns are monthly by default for annualization.
    """
    if len(returns) < 2:
        return 0.0 # Not enough data for meaningful std dev

    excess_returns = np.array(returns) - risk_free_rate
    avg_excess_return = np.mean(excess_returns)
    std_dev_excess_return = np.std(excess_returns, ddof=1) # Sample standard deviation

    if std_dev_excess_return == 0:
        return 0.0 # Avoid division by zero

    # Annualize if monthly returns
    annualized_avg_excess_return = avg_excess_return * annualization_factor
    annualized_std_dev_excess_return = std_dev_excess_return * np.sqrt(annualization_factor)

    return annualized_avg_excess_return / annualized_std_dev_excess_return

def project_returns(initial_investment, annual_return_rate, years):
    """
    Projects the future value of an investment.
    """
    future_value = initial_investment * ((1 + annual_return_rate) ** years)
    return future_value

def calculate_portfolio_value(holdings):
    """
    Calculates the total value of a portfolio from a list of holdings.
    holdings is expected to be a list of dicts, e.g., [{'name': 'Stock A', 'value': 10000}].
    """
    total_value = sum(asset['value'] for asset in holdings)
    return total_value

def calculate_portfolio_returns(asset_returns_list, weights=None):
    """
    Calculates the overall portfolio returns from a list of asset returns.
    asset_returns_list: list of lists, where each inner list contains monthly returns for an asset.
    weights: list of floats, weights for each asset. If None, equal weights are assumed.
    """
    if not asset_returns_list:
        return []

    num_assets = len(asset_returns_list)
    num_periods = len(asset_returns_list[0]) # Assuming all assets have same number of periods

    if weights is None:
        weights = [1.0 / num_assets] * num_assets
    elif len(weights) != num_assets:
        raise ValueError("Number of weights must match number of assets.")

    portfolio_returns = []
    for i in range(num_periods):
        period_return = sum(asset_returns_list[j][i] * weights[j] for j in range(num_assets))
        portfolio_returns.append(period_return)

    return portfolio_returns
