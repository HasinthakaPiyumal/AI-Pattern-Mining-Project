
import os

def load_medical_passages(filepath: str) -> list[str]:
    """Loads medical passages from a given text file.

    Args:
        filepath: The path to the text file containing medical passages.

    Returns:
        A list of strings, where each string is a medical passage.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        passages = [line.strip() for line in f if line.strip()]
    return passages

def preprocess_passage(passage: str) -> str:
    """Performs basic preprocessing on a medical passage.

    Args:
        passage: The raw medical passage string.

    Returns:
        The preprocessed passage string.
    """
    # Example preprocessing: convert to lowercase and remove extra whitespace
    return passage.lower().strip()

