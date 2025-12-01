
import json
import os

class DataLoader:
    """Loads chat transcripts from a specified file path."""

    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_data(self) -> list:
        """Loads data from a JSON file.

        Returns:
            A list of dictionaries, where each dictionary represents a chat transcript.
        """
        if not os.path.exists(self.data_path):
            print(f"Warning: Data file not found at {self.data_path}. Returning empty list.")
            return []

        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} items from {self.data_path}.")
        return data
