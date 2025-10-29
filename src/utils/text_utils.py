import re
# from langchain_text_splitters import RecursiveCharacterTextSplitter
def link_cleaner(text):
    """
    Cleans links from the text.

    Args:
        text (str): The input text.

    Returns:
        str: The text with links removed.
    """
    url_pattern = r'http\S+|www\S+|https\S+'
    return re.sub(url_pattern, '', text, flags=re.MULTILINE)

def quote_cleaner(text):
    """
    Cleans quotes from the text.

    Args:
        text (str): The input text.

    Returns:
        str: The text with quotes standardized.
    """
    # Standardize quotes
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return text

def remove_extra_whitespaces(text):
    """
    Removes extra whitespaces from the text.

    Args:
        text (str): The input text.

    Returns:
        str: The text with extra whitespaces removed.
    """
    return ' '.join(text.split())

def remove_emojis(text):
    """
    Removes emojis from the text.

    Args:
        text (str): The input text.

    Returns:
        str: The text with emojis removed.
    """
    return text.encode('ascii', 'ignore').decode('ascii')

def remove_symbols(text):
    """
    Removes special symbols from the text.

    Args:
        text (str): The input text.

    Returns:
        str: The text with special symbols removed.
    """
    return ''.join(char for char in text if char.isalnum() or char.isspace())



def text_cleaner(text):
    """
    Cleans the input text by removing links and standardizing quotes.

    Args:
        text (str): The input text.

    Returns:
        str: The cleaned text.
    """
    text = link_cleaner(text)
    text = quote_cleaner(text)
    # text = remove_extra_whitespaces(text)
    text = remove_emojis(text)
    text = remove_symbols(text)
    return text



# def chunk_text(text, chunk_size=10000, overlap=2000):
#     splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
#     return splitter.split_text(text)