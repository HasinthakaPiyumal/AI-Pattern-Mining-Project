import nltk
from nltk.tokenize import sent_tokenize

# Ensure you have the 'punkt' tokenizer data
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

def load_document(file_path: str) -> str:
    """
    Loads content from a text document.
    For a real-world application, this would handle various file types like PDF, DOCX.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File not found at {file_path}"
    except Exception as e:
        return f"Error loading document: {e}"

def chunk_text(text: str, max_chunk_length: int = 1000) -> list[str]:
    """
    Divides a long text into smaller, contextually relevant chunks.
    It first tokenizes by sentence and then combines sentences into chunks
    up to max_chunk_length (character count for simplicity here).
    In a real medical context, token limits (e.g., using tiktoken) would be preferred.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        # Check if adding the next sentence exceeds the max_chunk_length
        # Add 1 for the space between sentences
        if current_length + len(sentence) + (1 if current_chunk else 0) > max_chunk_length and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)
        else:
            current_chunk.append(sentence)
            current_length += len(sentence) + (1 if current_chunk else 0)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

