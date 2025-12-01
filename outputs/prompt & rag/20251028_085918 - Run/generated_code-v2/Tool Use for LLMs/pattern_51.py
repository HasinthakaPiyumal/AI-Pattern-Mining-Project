import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class FinancialDataHandler:
    """
    Simulates a financial data source.
    Holds mock transaction data and provides methods to query it.
    """
    def __init__(self):
        self.data = self._generate_mock_data()

    def _generate_mock_data(self):
        np.random.seed(42)
        num_transactions = 1000
        
        dates = [datetime.now() - timedelta(days=int(d)) for d in np.random.randint(1, 365, num_transactions)]
        amounts = np.random.normal(loc=1000, scale=500, size=num_transactions)
        amounts[amounts < 1] = 1 # Ensure positive amounts
        
        # Introduce some anomalies
        anomaly_indices = np.random.choice(num_transactions, 5, replace=False)
        amounts[anomaly_indices] = amounts[anomaly_indices] * np.random.uniform(5, 20, 5) # Abnormally large amounts

        types = np.random.choice(['debit', 'credit', 'transfer'], num_transactions, p=[0.5, 0.4, 0.1])
        accounts = np.random.randint(1001, 1010, num_transactions)

        df = pd.DataFrame({
            'transaction_id': range(num_transactions),
            'date': dates,
            'amount': amounts,
            'type': types,
            'account_id': accounts
        })
        df = df.sort_values(by='date').reset_index(drop=True)
        return df

    def get_transactions_by_account(self, account_id: int):
        """Retrieves transactions for a specific account."""
        return self.data[self.data['account_id'] == account_id]

    def get_transactions_above_threshold(self, threshold: float):
        """Retrieves transactions with amount above a certain threshold."""
        return self.data[self.data['amount'] > threshold]

    def get_all_data(self):
        """Returns the entire dataset."""
        return self.data

    def describe_data(self):
        """Provides a statistical description of the data."""
        return self.data.describe()