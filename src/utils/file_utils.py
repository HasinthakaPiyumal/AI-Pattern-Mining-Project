from pypdf import PdfReader
import os
from utils.text_utils import text_cleaner 
from utils.logger import get_logger

logger = get_logger(__name__)

def read_pdf_file(file_path):
    text_content = ""
    try:
        with open(file_path, "rb") as file:
            reader = PdfReader(file)
            for page in reader.pages:
                    text_content += page.extract_text() + "\n"
    except Exception as e:
        logger.error(f"Error extracting text from page {file_path}: {e}")
    return text_content

def read_text_file(file_path):
    text_content = ""
    try:
        with open(file_path, "r") as file:
            text_content = file.read()
    except Exception as e:
        logger.error(f"Error reading text file {file_path}: {e}")
    return text_content

def save_text_to_file(text, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as file:
        file.write(text)

def clean_file_content(file_path):
    raw_text = read_pdf_file(file_path)
    cleaned_text = text_cleaner(raw_text)
    return cleaned_text

def clean_and_save_file(file_path, output_path):
    cleaned_text = clean_file_content(file_path)
    save_text_to_file(cleaned_text, output_path)


def _collect_paths(folder_path, predicate=None):
    return [
        os.path.join(folder_path, name)
        for name in os.listdir(folder_path)
        if predicate is None or predicate(name)
    ]


def get_all_pdf_files(folder_path):
    return _collect_paths(folder_path, lambda name: name.lower().endswith(".pdf"))

def get_all_text_files(folder_path):
    return _collect_paths(folder_path, lambda name: name.lower().endswith(".txt"))

def get_all_json_files(folder_path):
    return _collect_paths(folder_path, lambda name: name.lower().endswith(".json"))

def get_all_files_in_folder(folder_path):
    return _collect_paths(folder_path)

def get_all_files_with_extension(folder_path, extension):
    ext = extension if extension.startswith(".") else f".{extension}"
    return _collect_paths(folder_path, lambda name: name.lower().endswith(ext.lower()))